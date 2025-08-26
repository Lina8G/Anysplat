from pathlib import Path
import torch
import os
import sys
import numpy as np
import csv

# Metrics
import lpips
from torchmetrics.functional.image import peak_signal_noise_ratio as compute_psnr
from torchmetrics.functional.image import structural_similarity_index_measure as compute_ssim
import torchvision.utils as vutils

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model.model.anysplat import AnySplat
from src.utils.image import process_image

def main():
    model = AnySplat.from_pretrained("lhjiang/anysplat")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    lpips_model = lpips.LPIPS(net='alex').to(device)

    all_metrics = []  # Store per-image metrics

    for claim_num in range(1, 71):  # Change range as needed
        image_folder = f"/shared/xinyu_gu_car3d/claims/filtered/claim_{claim_num}"
        if not os.path.isdir(image_folder):
            print(f"Skipping claim_{claim_num}: folder does not exist.")
            continue

        image_paths = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder)
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if len(image_paths) < 2:
            print(f"Skipping claim_{claim_num}: not enough images.")
            continue

        images = [process_image(p) for p in image_paths]
        images = torch.stack(images, dim=0).unsqueeze(0).to(device)  # [1, K, 3, H, W]
        b, K, _, H, W = images.shape

        with torch.no_grad():
            gaussians, pred_context_pose = model.inference((images + 1) * 0.5)
            extrinsics = pred_context_pose['extrinsic'][:, :K]
            intrinsics = pred_context_pose['intrinsic'][:, :K]

            for i in range(K):
                ext = extrinsics[:, i]  # [1, 4, 4]
                intr = intrinsics[:, i]  # [1, 3, 3]

                # Render the image at this view
                output = model.decoder.forward(
                    gaussians,
                    ext.unsqueeze(1),  # [1, 1, 4, 4]
                    intr.unsqueeze(1),  # [1, 1, 3, 3]
                    torch.ones(1, 1, device=device) * 0.1,
                    torch.ones(1, 1, device=device) * 100,
                    (H, W)
                )
                pred = output.color[0][0].unsqueeze(0).clamp(0, 1)  # [1, 3, H, W]

                gt = ((images[0, i].unsqueeze(0) + 1) / 2).clamp(0, 1)  # [1, 3, H, W]

                psnr = compute_psnr(pred, gt, data_range=1.0).item()
                ssim = compute_ssim(pred, gt, data_range=1.0).item()

                lp = lpips_model(pred, gt).item()

                all_metrics.append({
                    'claim_id': claim_num,
                    'image_index': i,
                    'psnr': psnr,
                    'ssim': ssim,
                    'lpips': lp
                })

        print(f"[✓] Processed claim {claim_num}")

    # Save metrics
    output_file = "anysplat_eval_all_metrics.csv"
    with open(output_file, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['claim_id', 'image_index', 'psnr', 'ssim', 'lpips'])
        writer.writeheader()
        writer.writerows(all_metrics)

        # Compute global mean/std
        psnr = compute_psnr(pred, gt, data_range=1.0).item()
        ssim = compute_ssim(pred, gt, data_range=1.0).item()

        lpips_vals = [m['lpips'] for m in all_metrics]

        writer.writerow({})
        writer.writerow({'claim_id': 'mean', 'psnr': np.mean(psnr_vals), 'ssim': np.mean(ssim_vals), 'lpips': np.mean(lpips_vals)})
        writer.writerow({'claim_id': 'std', 'psnr': np.std(psnr_vals), 'ssim': np.std(ssim_vals), 'lpips': np.std(lpips_vals)})

    print(f"[✓] Saved evaluation metrics to {output_file}")

if __name__ == "__main__":
    main()
