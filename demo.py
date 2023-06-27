from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
import pickle

import smplx

from utils import bvh, quat

from hmr2.configs import CACHE_DIR_4DHUMANS
from hmr2.models import HMR2, download_models, load_hmr2, DEFAULT_CHECKPOINT
from hmr2.utils import recursive_to
from hmr2.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hmr2.utils.renderer import Renderer, cam_crop_to_full

LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

def mirror_rot_trans(lrot, trans, names, parents):
    joints_mirror = np.array([(
        names.index("Left"+n[5:]) if n.startswith("Right") else (
        names.index("Right"+n[4:]) if n.startswith("Left") else
        names.index(n))) for n in names])

    mirror_pos = np.array([-1, 1, 1])
    mirror_rot = np.array([1, 1, -1, -1])
    grot = quat.fk_rot(lrot, parents)
    trans_mirror = mirror_pos * trans
    grot_mirror = mirror_rot * grot[:,joints_mirror]

    return quat.ik_rot(grot_mirror, parents), trans_mirror

def smpl2bvh(model_path:str, poses:str, output:str, mirror:bool,
             model_type="smpl", gender="MALE",
             num_betas=10, fps=60) -> None:
    """Save bvh file created by smpl parameters.

    Args:
        model_path (str): Path to smpl models.
        poses (str): Path to npz or pkl file.
        output (str): Where to save bvh.
        mirror (bool): Whether save mirror motion or not.
        model_type (str, optional): I prepared "smpl" only. Defaults to "smpl".
        gender (str, optional): Gender Information. Defaults to "MALE".
        num_betas (int, optional): How many pca parameters to use in SMPL. Defaults to 10.
        fps (int, optional): Frame per second. Defaults to 30.
    """

    names = [
        "Pelvis",
        "Left_hip",
        "Right_hip",
        "Spine1",
        "Left_knee",
        "Right_knee",
        "Spine2",
        "Left_ankle",
        "Right_ankle",
        "Spine3",
        "Left_foot",
        "Right_foot",
        "Neck",
        "Left_collar",
        "Right_collar",
        "Head",
        "Left_shoulder",
        "Right_shoulder",
        "Left_elbow",
        "Right_elbow",
        "Left_wrist",
        "Right_wrist",
        "Left_palm",
        "Right_palm",
    ]

    # I prepared smpl models only, 
    # but I will release for smplx models recently.
    model = smplx.create(model_path=model_path,
                        model_type=model_type,
                        gender=gender,
                        batch_size=1)

    parents = model.parents.detach().cpu().numpy()

    # You can define betas like this.(default betas are 0 at all.)
    rest = model(
        # betas = torch.randn([1, num_betas], dtype=torch.float32)
    )
    rest_pose = rest.joints.detach().cpu().numpy().squeeze()[:24,:]

    root_offset = rest_pose[0]
    offsets = rest_pose - rest_pose[parents]
    offsets[0] = root_offset
    offsets *= 100

    scaling = None

    # Pose setting.
    if poses.endswith(".npz"):
        poses = np.load(poses)
        rots = np.squeeze(poses["poses"], axis=0) # (N, 24, 3)
        trans = np.squeeze(poses["trans"], axis=0) # (N, 3)

    elif poses.endswith(".pkl"):
        with open(poses, "rb") as f:
            poses = pickle.load(f)
            rots = poses["smpl_poses"] # (N, 72)
            rots = rots.reshape(rots.shape[0], -1, 3) # (N, 24, 3)
            scaling = poses["smpl_scaling"]  # (1,)
            trans = poses["smpl_trans"]  # (N, 3)

    else:
        raise Exception("This file type is not supported!")

    if scaling is not None:
        trans /= scaling

    # to quaternion
    rots = quat.from_axis_angle(rots)

    order = "zyx"
    pos = offsets[None].repeat(len(rots), axis=0)
    positions = pos.copy()
    positions[:,0] += trans * 100
    rotations = np.degrees(quat.to_euler(rots, order=order))

    bvh_data ={
        "rotations": rotations,
        "positions": positions,
        "offsets": offsets,
        "parents": parents,
        "names": names,
        "order": order,
        "frametime": 1 / fps,
    }

    if not output.endswith(".bvh"):
        output = output + ".bvh"

    bvh.save(output, bvh_data)

    if mirror:
        rots_mirror, trans_mirror = mirror_rot_trans(
                rots, trans, names, parents)
        positions_mirror = pos.copy()
        positions_mirror[:,0] += trans_mirror
        rotations_mirror = np.degrees(
            quat.to_euler(rots_mirror, order=order))

        bvh_data ={
            "rotations": rotations_mirror,
            "positions": positions_mirror,
            "offsets": offsets,
            "parents": parents,
            "names": names,
            "order": order,
            "frametime": 1 / fps,
        }

        output_mirror = output + "_mirror.bvh"
        bvh.save(output_mirror, bvh_data)


def main():
    parser = argparse.ArgumentParser(description='HMR2 demo code')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--img_folder', type=str, default='example_data/images', help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='demo_out', help='Output folder to save rendered results')
    parser.add_argument('--side_view', dest='side_view', action='store_true', default=False, help='If set, render side view also')
    parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=False, help='If set, render all people together also')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')

    args = parser.parse_args()

    # Download and load checkpoints
    download_models(CACHE_DIR_4DHUMANS)
    model, model_cfg = load_hmr2(args.checkpoint)

    # Setup HMR2.0 model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # Load detector
    from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy
    from detectron2.config import LazyConfig
    import hmr2
    cfg_path = Path(hmr2.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = DefaultPredictor_Lazy(detectron2_cfg)

    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.smpl.faces)

    # Make output directory if it does not exist
    os.makedirs(args.out_folder, exist_ok=True)

    # Iterate over all images in folder
    for img_path in Path(args.img_folder).glob('*.png'):
        img_cv2 = cv2.imread(str(img_path))

        # Detect humans in image
        det_out = detector(img_cv2)

        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
        boxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()

        # Run HMR2.0 on all detected humans
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

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
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

            #Test
            '''
            body_pose = out['pred_smpl_params']['body_pose'].detach().cpu().numpy()
            global_orient = out['pred_smpl_params']['global_orient'].detach().cpu().numpy()
            smpl_poses_p = np.concatenate([global_orient, body_pose], axis=1)
            smpl_scaling_p = scaled_focal_length.detach().cpu().numpy()
            smpl_trans_p = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

            dataPose = {
                "smpl_poses": smpl_poses_p,
                "smpl_scaling": smpl_scaling_p,
                "smpl_trans": smpl_trans_p
            }

            with open(os.path.join(args.out_folder, f'test.pkl'), 'wb') as f:
                pickle.dump(dataPose, f)
            '''

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

                if args.side_view:
                    side_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                            out['pred_cam_t'][n].detach().cpu().numpy(),
                                            white_img,
                                            mesh_base_color=LIGHT_BLUE,
                                            scene_bg_color=(1, 1, 1),
                                            side_view=True)
                    final_img = np.concatenate([input_patch, regression_img, side_img], axis=1)
                else:
                    final_img = np.concatenate([input_patch, regression_img], axis=1)

                cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_{person_id}.png'), 255*final_img[:, :, ::-1])

                # Add all verts and cams to list
                verts = out['pred_vertices'][n].detach().cpu().numpy()
                cam_t = pred_cam_t_full[n]
                all_verts.append(verts)
                all_cam_t.append(cam_t)

                # smpl -> pkl
                smpl_poses = np.array(all_verts)  # (N, ...)
                smpl_scaling = scaled_focal_length.detach().cpu().numpy()  # No info
                smpl_trans = np.array(all_cam_t)  # (N, ...)

                poses = {
                    "smpl_poses": smpl_poses,
                    "smpl_scaling": smpl_scaling,
                    "smpl_trans": smpl_trans
                }

                with open(os.path.join(args.out_folder, f'{img_fn}_{person_id}.pkl'), 'wb') as f:
                    pickle.dump(poses, f)

                argsSMPL2BVH = {
                    "model_path" : "data/",
                    "model_type" : "smpl",
                    "gender" : "MALE",
                    "num_betas" : 10,
                    "poses" : os.path.join(args.out_folder, f'{img_fn}_{person_id}.pkl'),
                    "fps" : 60,
                    "output" : os.path.join(args.out_folder, f'{img_fn}_{person_id}.bvh'),
                    "mirror" : True
                }

                print(argsSMPL2BVH)

                smpl2bvh(
                    model_path=argsSMPL2BVH["model_path"],
                    model_type=argsSMPL2BVH["model_type"],
                    mirror=argsSMPL2BVH["mirror"],
                    gender=argsSMPL2BVH["gender"],
                    poses=argsSMPL2BVH["poses"],
                    num_betas=argsSMPL2BVH["num_betas"],
                    fps=argsSMPL2BVH["fps"],
                    output=argsSMPL2BVH["output"]
                )

                print("finished!")

                # Save all meshes to disk
                if args.save_mesh:
                    camera_translation = cam_t.copy()
                    tmesh = renderer.vertices_to_trimesh(verts, camera_translation, LIGHT_BLUE)
                    tmesh.export(os.path.join(args.out_folder, f'{img_fn}_{person_id}.obj'))

        # Render front view
        if args.full_frame and len(all_verts) > 0:
            misc_args = dict(
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[n], **misc_args)

            # Overlay image
            input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
            input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]

            cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_all.png'), 255*input_img_overlay[:, :, ::-1])

if __name__ == '__main__':
    main()