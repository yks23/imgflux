import argparse
from PIL import Image
import os
import json
import torch
from src.flux.xflux_pipeline import XFluxPipeline,DoubleControlPipeline
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
def get_train_set():
    return  HOIVideoDatasetResizing(
        data_root=Path("/data115/video-diff/workspace/HOI-DiffusionAsShader/"),
        caption_column=Path("data/dexycb_filelist/val_prompts.txt"),
        video_column=Path("data/dexycb_filelist/val_videos.txt"),
        normal_column=Path("data/dexycb_filelist/val_normals.txt"),
        depth_column=Path("data/dexycb_filelist/val_depths.txt"),
        label_column=Path("data/dexycb_filelist/val_labels.txt"),
        image_to_video=True,
        load_tensors=False,
        max_num_frames=72,
        frame_buckets=[8],
        height_buckets=[480],
        width_buckets=[640],
        is_valid=True,
        used_condition={'depth','mask','hand','normal'}
    )
def get_valid_set():
    return HOIVideoDatasetResizing(
        data_root=Path("/data115/video-diff/workspace/HOI-DiffusionAsShader/"),
        caption_column=Path("data/dexycb_filelist/val_prompts.txt"),
        video_column=Path("data/dexycb_filelist/val_video_prev.txt"),
        normal_column=Path("data/dexycb_filelist/val_normals.txt"),
        depth_column=Path("data/dexycb_filelist/val_depths.txt"),
        label_column=Path("data/dexycb_filelist/val_labels.txt"),
        image_to_video=True,
        load_tensors=False,
        max_num_frames=72,
        frame_buckets=[8],
        height_buckets=[480],
        width_buckets=[640],
        is_valid=True,
        used_condition={'depth','mask','hand','normal'},
    )
def main(args):
    dataset= get_valid_set()
    xflux_pipeline = DoubleControlPipeline(model_type='flux-dev', device='cuda', offload=True,control_net=args.local_path)
    # print(xflux_pipeline.controlnet1.input_hint_block[0].weight[:,0:3,:,:].norm(),xflux_pipeline.controlnet1.input_hint_block[0].weight[:,3:6,:,:].norm(),xflux_pipeline.controlnet2.input_hint_block[0].weight[:,0:3,:,:].norm(),xflux_pipeline.controlnet2.input_hint_block[0].weight[:,3:6,:,:].norm())
    os.makedirs(args.save_path, exist_ok=True)
    with open("./valid/desp.json",'r') as f:
        descriptions=json.load(f)
    # 实验1，（1，1，-1，-1），在三个图片上进行五种场景的推断
    datalist=[(0,dataset[0],descriptions['0']),(3,dataset[3],descriptions['3']),(9,dataset[9],descriptions['9']),(32,dataset[32],descriptions['32'])]
    all_control_gs=[
        # 双重
        (1.0,1.0,0,0),
        (1.0,1.0,-0.3,-0.3),
        (1.0,1.0,-0.5,-0.5),
        (1.0,1.0,-0.7,-0.7),
        (1.0,1.0,-1,-1),
        # 单独
        (1,0,0,0),
        (0,1,0,0),
        (1,0,-1,0),
        (0,1,0,-1),
        (0,0,1,1),
        # # 物理
        # (1.0,0,-1,0),
        # (1.0,0,-0.7,0),
        # (1.0,0,-0.5,0),
        # (1.0,0,0,0),
        # # 语义
        # (0,1,0,-1),
        # (0,1,0,-0.7),
        # (0,1,0,-0.5),
        # (0,1,0,0),
        # # 默认
        # (0,0,0,1),
        # (0,0,1,0),
        # (0,0,1,1),
        # (0,0,0.5,0.5),
        # # 不均等
        # (0.7,1.0,0,0),
        # (0.3,1.0,0,0),
        # (0.3,1.0,-0.3,0),
        # (0.7,1.0,-0.7,0),
        # # 物理测试
        # (0.5,0,-0.5,0),
        # (0.7,0,-0.7,0),
        # (1.0, 1.0, 0,0)
        
    ]
    # for cont in all_control_gs:
    #     os.makedirs(f'./{args.save_path}/{str(cont)}/',exist_ok=True)
    #     os.makedirs(f'./{args.save_path}/validset1',exist_ok=True)
    #     os.makedirs(f'./{args.save_path}/validset2',exist_ok=True)
    os.makedirs(f'./{args.save_path}/validset3',exist_ok=True)
    # for i,data,prompts in datalist:
    #     for j,p in prompts.items():
    #         with torch.no_grad():
    #             print(f"infer on {i} {j}")
    #             data['prompt']=p
    #             for cont in all_control_gs:
    #                 result= xflux_pipeline.infer_data(
    #                             data=data, seed=42,control_gs=cont
    #                 )
    #                 cv2.imwrite(f'./{args.save_path}/{str(cont)}/data_{i}on{j}.png', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    for i  in range(100):
        for j in range(5):
            result= xflux_pipeline.infer_data(
                                    data=dataset[i*5+2], seed=42+j*3,control_gs=(1,1,0,0)
            )
            cv2.imwrite(f'./{args.save_path}/validset3/data_{i}_{j}.png', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    # for i  in range(100):
    #     result= xflux_pipeline.infer_data(
    #                             data=dataset[i*5+2], seed=42,control_gs=(1,1,-1,-1)
    #     )
    #     cv2.imwrite(f'./{args.save_path}/validset2/data_{i}.png', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    
    # for i,data in enumerate(dataset):
    #     if i!=3:
    #         continue
    #     if i%5!=1  or i==1:
    #         continue
    #     for j in [0,1,2,3,4]:
    #         for k in [1.0]:
    #             with torch.no_grad():
    #                 print(f"infer on {j}")
    #                 data['prompt']=descriptions[str(j)]
    #                 data['prompt']=data['prompt'].replace('black','white')
    #                 result= xflux_pipeline.infer_data(
    #                     data=data, seed=42+j,control_gs=k
    #                 )
    #                 cv2.imwrite(f'./{args.save_path}/{i}_infer_{j}_{k}.png', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    # for i,data in enumerate(dataset):
    #     with torch.no_grad():
    #                 result= xflux_pipeline.infer_data(
    #                     data=data, seed=42,control_gs=1.0
    #                 )
    #                 cv2.imwrite(f'./{args.save_path}/{i}_infer.png', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    #     if i == 300:
    #         break

if __name__ == "__main__":
    args = create_argparser().parse_args()
    main(args)
