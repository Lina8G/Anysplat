from pathlib import Path
import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.misc.image_io import save_interpolated_video, save_interpolated_images
from src.model.ply_export import export_ply
from src.model.model.anysplat import AnySplat
from src.utils.image import process_image

def main():
    # Load the model from Hugging Face
    model = AnySplat.from_pretrained("lhjiang/anysplat")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Load Images
    image_folder = f"./examples/4_frames"

    images = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    images = [process_image(img_path) for img_path in images]
    images = torch.stack(images, dim=0).unsqueeze(0).to(device) # [1, K, 3, 448, 448]
    b, v, _, h, w = images.shape
    # Run Inference
    gaussians, pred_context_pose = model.inference((images+1)*0.5)
    pred_all_extrinsic = pred_context_pose['extrinsic']  # [1, K, 4, 4]
    pred_all_intrinsic = pred_context_pose['intrinsic']  # [1, K, 3, 3]

    # --- Pose sequencing (greedy nearest neighbor) ---
    # Only use the translation part for sequencing
    positions = pred_all_extrinsic[0, :, :3, 3].cpu().numpy()  # (K, 3)
    K = positions.shape[0]
    visited = np.zeros(K, dtype=bool)
    order = [0]
    visited[0] = True
    for _ in range(1, K):
        last = order[-1]
        dists = np.linalg.norm(positions - positions[last], axis=1)
        dists[visited] = np.inf
        next_idx = np.argmin(dists)
        order.append(next_idx)
        visited[next_idx] = True

    # Reorder images and poses
    images = images[:, order]
    pred_all_extrinsic = pred_all_extrinsic[:, order]
    pred_all_intrinsic = pred_all_intrinsic[:, order]

    # Save the rendering results as videos
    save_interpolated_video(pred_all_extrinsic, pred_all_intrinsic, b, h, w, gaussians, image_folder+'/result_videos', model.decoder)
    # Save the interpolated images (5 interpolated views between each pair of input views)
    save_interpolated_images(pred_all_extrinsic, pred_all_intrinsic, b, h, w, gaussians, image_folder+'/result_images', model.decoder, t=5)

if __name__ == "__main__":
    main()
