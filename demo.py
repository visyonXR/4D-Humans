from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
from PIL import Image
import time
import psutil
import subprocess
import re

from hmr2.configs import get_config
from hmr2.models import HMR2
from hmr2.utils import recursive_to
from hmr2.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hmr2.utils.renderer import Renderer, cam_crop_to_full

LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

DEFAULT_CHECKPOINT='logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt'
parser = argparse.ArgumentParser(description='HMR2 demo code')
parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
parser.add_argument('--img_folder', type=str, default='example_data/images', help='Folder with input images')
parser.add_argument('--out_folder', type=str, default='demo_out', help='Output folder to save rendered results')
parser.add_argument('--side_view', dest='side_view', action='store_true', default=False, help='If set, render side view also')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')

args = parser.parse_args()

# Setup HMR2.0 model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model_cfg = str(Path(args.checkpoint).parent.parent / 'model_config.yaml')
model_cfg = get_config(model_cfg)
model = HMR2.load_from_checkpoint(args.checkpoint, strict=False, cfg=model_cfg).to(device)
model.eval()

# Load detector
from detectron2.config import LazyConfig
from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy
detectron2_cfg = LazyConfig.load(f"vendor/detectron2/projects/ViTDet/configs/COCO/cascade_mask_rcnn_vitdet_h_75ep.py")
detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
for i in range(3):
    detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
detector = DefaultPredictor_Lazy(detectron2_cfg)

# Setup the renderer
renderer = Renderer(model_cfg, faces=model.smpl.faces)

# Make output directory if it does not exist
os.makedirs(args.out_folder, exist_ok=True)

# Iterate over all images in folder
jpg_files = Path(args.img_folder).glob('*.jpg')
png_files = Path(args.img_folder).glob('*.png')
all_files = list(jpg_files) + list(png_files)

def get_gpu_memory_usage():
    gpu_info = subprocess.check_output(['nvidia-smi']).decode('utf-8')
    pattern = r"(\d+)\s*MiB\s*/\s*(\d+)\s*MiB\s*.*\s*(\d+)%\s*"
    match = re.search(pattern, gpu_info)
    if match:
        used_memory = int(match.group(1))
        total_memory = int(match.group(2))
        gpu_percent = round(((used_memory / total_memory) * 100), 1)
        return gpu_percent
    else:
        return 0

for img_path in all_files:
    start_time = time.time()
    img_cv2 = cv2.imread(str(img_path))

    # Detect humans in image
    det_out = detector(img_cv2)

    det_instances = det_out['instances']
    valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
    boxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()

    # Run HMR2.0 on all detected humans
    dataset = ViTDetDataset(model_cfg, img_cv2, boxes)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

    cpu_percent1 = psutil.cpu_percent()
    ram_percent1 = psutil.virtual_memory().percent
    gpu_percent1 = get_gpu_memory_usage()

    all_verts = []
    all_cam_t = []
    
    for batch in dataloader:
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = model(batch)

        pred_cam = out['pred_cam']
        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        img_size = batch["img_size"].float()
        render_size = img_size
        pred_cam_t = cam_crop_to_full(pred_cam, box_center, box_size, render_size).detach().cpu().numpy()

        # Render the result
        batch_size = batch['img'].shape[0]
        for n in range(batch_size):
            # Get filename from path img_path
            img_fn, _ = os.path.splitext(os.path.basename(img_path))
            person_id = int(batch['personid'][n])
            white_img = (torch.ones_like(batch['img'][n]).cpu() - DEFAULT_MEAN[:,None,None]/255) / (DEFAULT_STD[:,None,None]/255)
            input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:,None,None]/255) + (DEFAULT_MEAN[:,None,None]/255)
            input_patch = input_patch.permute(1,2,0).numpy()
            
            regression_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                    out['pred_cam_t'][n].detach().cpu().numpy(),
                                    batch['img'][n],
                                    mesh_base_color=LIGHT_BLUE,
                                    scene_bg_color=(1, 1, 1),
                                    )

            verts = out['pred_vertices'][n].detach().cpu().numpy()
            cam_t = pred_cam_t[n]

            all_verts.append(verts)
            all_cam_t.append(cam_t)

    cpu_percent2 = psutil.cpu_percent()
    ram_percent2 = psutil.virtual_memory().percent
    gpu_percent2 = get_gpu_memory_usage()

    if len(all_verts) > 0:
        misc_args = dict(
            mesh_base_color=LIGHT_BLUE,
            scene_bg_color=(1, 1, 1),
        )
        cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=render_size[n], **misc_args)

        # Overlay image
        input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
        input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
        input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]

        # convert to PIL image
        out_pil_img =  Image.fromarray((input_img_overlay*255).astype(np.uint8))

        out_pil_img.save(os.path.join(args.out_folder, f'{img_fn}_final.png'))

    end_time = time.time()
    processing_time = end_time - start_time

    cpu_percent3 = psutil.cpu_percent()
    ram_percent3 = psutil.virtual_memory().percent
    gpu_percent3 = get_gpu_memory_usage()

    print(f"Tiempo de procesamiento de la imagen: {processing_time:.2f} segundos")
    print(f"Estado del sistema - Uso de CPU: {cpu_percent1}% {cpu_percent2}% {cpu_percent3}%")
    print(f"Estado del sistema - Uso de GPU: {gpu_percent1}% {gpu_percent2}% {gpu_percent3}%")
    print(f"Estado del sistema - Uso de RAM: {ram_percent1}% {ram_percent2}% {ram_percent3}%")
