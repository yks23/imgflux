import argparse
from PIL import Image
import os
import torch
from src.flux.xflux_pipeline import XFluxPipeline
from src.dataset.hoivideodataset import HOIVideoDatasetResizing
from pathlib import Path
import numpy as np
import cv2
def create_argparser():
    parser = argparse.ArgumentParser()

    # parser.add_argument(
    #     "--prompt", type=str, required=True,
    #     help="The input text prompt"
    # )
    parser.add_argument(
        "--neg_prompt", type=str, default="",
        help="The input text negative prompt"
    )
    parser.add_argument(
        "--img_prompt", type=str, default=None,
        help="Path to input image prompt"
    )
    parser.add_argument(
        "--neg_img_prompt", type=str, default=None,
        help="Path to input negative image prompt"
    )
    parser.add_argument(
        "--ip_scale", type=float, default=1.0,
        help="Strength of input image prompt"
    )
    parser.add_argument(
        "--neg_ip_scale", type=float, default=1.0,
        help="Strength of negative input image prompt"
    )
    parser.add_argument(
        "--local_path", type=str, default=None,
        help="Local path to the model checkpoint (Controlnet)"
    )
    parser.add_argument(
        "--repo_id", type=str, default=None,
        help="A HuggingFace repo id to download model (Controlnet)"
    )
    parser.add_argument(
        "--name", type=str, default=None,
        help="A filename to download from HuggingFace"
    )
    parser.add_argument(
        "--ip_repo_id", type=str, default=None,
        help="A HuggingFace repo id to download model (IP-Adapter)"
    )
    parser.add_argument(
        "--ip_name", type=str, default=None,
        help="A IP-Adapter filename to download from HuggingFace"
    )
    parser.add_argument(
        "--ip_local_path", type=str, default=None,
        help="Local path to the model checkpoint (IP-Adapter)"
    )
    parser.add_argument(
        "--lora_repo_id", type=str, default=None,
        help="A HuggingFace repo id to download model (LoRA)"
    )
    parser.add_argument(
        "--lora_name", type=str, default=None,
        help="A LoRA filename to download from HuggingFace"
    )
    parser.add_argument(
        "--lora_local_path", type=str, default=None,
        help="Local path to the model checkpoint (Controlnet)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use (e.g. cpu, cuda:0, cuda:1, etc.)"
    )
    parser.add_argument(
        "--offload", action='store_true', help="Offload model to CPU when not in use"
    )
    parser.add_argument(
        "--use_ip", action='store_true', help="Load IP model"
    )
    parser.add_argument(
        "--use_lora", action='store_true', help="Load Lora model"
    )
    parser.add_argument(
        "--use_controlnet", action='store_true', help="Load Controlnet model"
    )
    parser.add_argument(
        "--num_images_per_prompt", type=int, default=1,
        help="The number of images to generate per prompt"
    )
    parser.add_argument(
        "--image", type=str, default=None, help="Path to image"
    )
    parser.add_argument(
        "--lora_weight", type=float, default=0.9, help="Lora model strength (from 0 to 1.0)"
    )
    parser.add_argument(
        "--control_weight", type=float, default=0.8, help="Controlnet model strength (from 0 to 1.0)"
    )
    parser.add_argument(
        "--control_type", type=str, default="canny",
        choices=("canny", "openpose", "depth", "zoe", "hed", "hough", "tile"),
        help="Name of controlnet condition, example: canny"
    )
    parser.add_argument(
        "--model_type", type=str, default="flux-dev",
        choices=("flux-dev", "flux-dev-fp8", "flux-schnell"),
        help="Model type to use (flux-dev, flux-dev-fp8, flux-schnell)"
    )
    parser.add_argument(
        "--width", type=int, default=1024, help="The width for generated image"
    )
    parser.add_argument(
        "--height", type=int, default=1024, help="The height for generated image"
    )
    parser.add_argument(
        "--num_steps", type=int, default=25, help="The num_steps for diffusion process"
    )
    parser.add_argument(
        "--guidance", type=float, default=4, help="The guidance for diffusion process"
    )
    parser.add_argument(
        "--seed", type=int, default=123456789, help="A seed for reproducible inference"
    )
    parser.add_argument(
        "--true_gs", type=float, default=3.5, help="true guidance"
    )
    parser.add_argument(
        "--timestep_to_start_cfg", type=int, default=5, help="timestep to start true guidance"
    )
    parser.add_argument(
        "--save_path", type=str, default='results', help="Path to save"
    )
    return parser

def get_concat_img(infer,gt,normal,depth,segmask,output_path):
    infer=infer.permute(1,2,0).numpy().astype('uint8')
    gt=gt.permute(1,2,0).numpy().astype('uint8')
    normal=normal.permute(1,2,0).numpy().astype('uint8')
    depth=depth.permute(1,2,0).numpy().astype('uint8')
    segmask=segmask.permute(1,2,0).numpy().astype('uint8')
    img = np.concatenate((infer,gt,normal,depth,segmask), axis=1)
    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    

def main(args):
    dataset = HOIVideoDatasetResizing(
        data_root=Path("/data115/video-diff/workspace/HOI-DiffusionAsShader/"),
        caption_column=Path("data/ykspath/valid/val_prompts.txt"),
        video_column=Path("data/ykspath/valid/val_videos.txt"),
        tracking_column=Path("data/ykspath/valid/val_trackings.txt"),
        normal_column=Path("data/ykspath/valid/val_normals.txt"),
        depth_column=Path("data/ykspath/valid/val_depths.txt"),
        label_column=Path("data/ykspath/valid/val_labels.txt"),
        image_to_video=True,
        load_tensors=False,
        max_num_frames=72,
        frame_buckets=[8],
        height_buckets=[480],
        width_buckets=[640],
        loss_threshold=10,
        filter_file='/data115/video-diff/workspace/hamer/dexycb_filter_sorted.jsonl'
    )
    xflux_pipeline = XFluxPipeline('hoi', 'cuda', True, args.local_path)
    os.makedirs(args.save_path, exist_ok=True)
    for i in range(30):
        for j in range(1):
            with torch.no_grad():
                print(f"infer on {i} {j}")
                result= xflux_pipeline.infer_data_naive(
                    data=dataset[i], seed=42+j
                )
                cv2.imwrite(f'./{args.save_path}/{i}_infer_{j}.png', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    args = create_argparser().parse_args()
    main(args)
