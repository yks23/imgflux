import json
from pathlib import Path
from PIL import Image
import random
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as TT
from accelerate.logging import get_logger
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from transformers import AutoProcessor, LlavaNextForConditionalGeneration
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
import os
import sys
from src.dataset.hoi_utils import showHandJoints, convert_gray_to_color
from src.dataset.hoivideodataset import VideoDataset
import decord
class HOIVideoDatasetResizing(VideoDataset):
    def __init__(self, *args, **kwargs) -> None:
        self.tracking_column = kwargs.pop("tracking_column", None)
        self.normal_column = kwargs.pop("normal_column", None)
        self.depth_column = kwargs.pop("depth_column", None)
        self.label_column = kwargs.pop("label_column", None)
        self.device = 'cuda'
        self.llava_processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        self.llava_model = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf",
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        self.frame_interval = kwargs.pop("frame_interval", 8)  # 采样间隔
        super().__init__(*args, **kwargs)

    def _get_description_path(self, video_path: Path) -> Path:
        """获取描述文件的路径"""
        return video_path.parent / f"{video_path.stem}_descriptions.json"

    def _load_descriptions(self, video_path: Path) -> dict:
        """加载已有的描述文件"""
        desc_path = self._get_description_path(video_path)
        if desc_path.exists():
            with open(desc_path, "r") as f:
                return json.load(f)
        return {}

    def _save_descriptions(self, video_path: Path, descriptions: dict):
        """保存描述文件"""
        desc_path = self._get_description_path(video_path)
        with open(desc_path, "w") as f:
            json.dump(descriptions, f, indent=2)

    def generate_description(self, image):
        """使用 LLaVA v1.6 Mistral-7B 模型生成图像描述"""
        image = image.convert("RGB")
        prompt = "<image>\nDescribe this image in detail."
        inputs = self.llava_processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.llava_model.generate(
                **inputs,
                max_new_tokens=400,
                do_sample=True,
                temperature=0.3,
            )
        description = self.llava_processor.decode(outputs[0], skip_special_tokens=True)
        description = description.replace("Describe this image in detail.", "").strip()
        return description

    def _preprocess_video(self, path: Path, tracking_path: Path, normal_path: Path, depth_path: Path, label_path: Path) -> torch.Tensor:
        if self.load_tensors:
            return self._load_preprocessed_latents_and_embeds(path, tracking_path, normal_path, depth_path, label_path)
        else:
            # 读取视频帧
            video_reader = decord.VideoReader(uri=path.as_posix())
            video_num_frames = len(video_reader)
            nearest_frame_bucket = min(
                self.frame_buckets, key=lambda x: abs(x - min(video_num_frames, self.max_num_frames))
            )

            # 采样帧
            frame_indices = list(range(0, video_num_frames, self.frame_interval))
            frames = video_reader.get_batch(frame_indices)
            frames = frames[:nearest_frame_bucket].float()
            frames = frames.permute(0, 3, 1, 2).contiguous()  # (T, C, H, W)

            # 调整分辨率
            nearest_res = self._find_nearest_resolution(frames.shape[2], frames.shape[3])
            frames_resized = torch.stack([resize(frame, nearest_res) for frame in frames], dim=0)
            frames = torch.stack([self.video_transforms(frame) for frame in frames_resized], dim=0)
            self.frames=frames
            # 生成描述
            descriptions = self._load_descriptions(path)
            for idx, frame in zip(frame_indices, frames):
                if str(idx) not in descriptions:
                    pil_img = Image.fromarray(frame.permute(1, 2, 0).byte().cpu().numpy())
                    description = self.generate_description(pil_img)
                    descriptions[str(idx)] = description
            self._save_descriptions(path, descriptions)

            # 其他处理逻辑（跟踪、深度等）
            image = frames[:1].clone() if self.image_to_video else None
            tracking_frames, normal_frames, depth_frames, colored_masks, hand_keypoints = self._process_auxiliary_data(
                tracking_path, normal_path, depth_path, label_path, frame_indices, nearest_frame_bucket, nearest_res
            )

            return {
                "image": image,
                "frames": frames,
                "tracking_frames": tracking_frames,
                "normal_frames": normal_frames,
                "depth_frames": depth_frames,
                "colored_masks": colored_masks,
                "hand_keypoints": hand_keypoints,
                "descriptions": descriptions
            }

    def _process_auxiliary_data(self, tracking_path, normal_path, depth_path, label_path, frame_indices, nearest_frame_bucket, nearest_res):
        """处理辅助数据（跟踪、深度等）"""
        tracking_frames = self._load_and_process_tracking(tracking_path, frame_indices, nearest_frame_bucket, nearest_res)
        normal_frames = self._load_and_process_normal(normal_path, frame_indices, nearest_frame_bucket, nearest_res)
        depth_frames = self._load_and_process_depth(depth_path, frame_indices, nearest_frame_bucket, nearest_res)
        colored_masks, hand_keypoints = self._load_and_process_label(label_path, frame_indices, nearest_frame_bucket, nearest_res)
        return tracking_frames, normal_frames, depth_frames, colored_masks, hand_keypoints

    def _load_and_process_tracking(self, tracking_path, frame_indices, nearest_frame_bucket, nearest_res):
        """加载和处理跟踪数据"""
        if tracking_path is not None and random.random() < 0.8:
            tracking_reader = decord.VideoReader(uri=tracking_path.as_posix())
            tracking_frames = tracking_reader.get_batch(frame_indices[:nearest_frame_bucket])
            tracking_frames = tracking_frames[:nearest_frame_bucket].float()
            tracking_frames = tracking_frames.permute(0, 3, 1, 2).contiguous()
            tracking_frames_resized = torch.stack([resize(tracking_frame, nearest_res) for tracking_frame in tracking_frames], dim=0)
            tracking_frames = torch.stack([self.video_transforms(tracking_frame) for tracking_frame in tracking_frames_resized], dim=0)
        else:
            tracking_frames = torch.zeros_like(self.frames)
        return tracking_frames

    def _load_and_process_normal(self, normal_path, frame_indices, nearest_frame_bucket, nearest_res):
        """加载和处理法线数据"""
        if normal_path is not None and random.random() < 0.7:
            normal_reader = decord.VideoReader(uri=normal_path.as_posix())
            normal_frames = normal_reader.get_batch(frame_indices[:nearest_frame_bucket])
            normal_frames = normal_frames[:nearest_frame_bucket].float()
            normal_frames = normal_frames.permute(0, 3, 1, 2).contiguous()
            normal_frames_resized = torch.stack([resize(normal_frame, nearest_res) for normal_frame in normal_frames], dim=0)
            normal_frames = torch.stack([self.video_transforms(normal_frame) for normal_frame in normal_frames_resized], dim=0)
        else:
            normal_frames = torch.zeros_like(self.frames)
        return normal_frames

    def _load_and_process_depth(self, depth_path, frame_indices, nearest_frame_bucket, nearest_res):
        """加载和处理深度数据"""
        if depth_path is not None and random.random() < 0.8:
            depth_reader = decord.VideoReader(uri=depth_path.as_posix())
            depth_frames = depth_reader.get_batch(frame_indices[:nearest_frame_bucket])
            depth_frames = depth_frames[:nearest_frame_bucket].float()
            depth_frames = depth_frames.permute(0, 3, 1, 2).contiguous()
            depth_frames_resized = torch.stack([resize(depth_frame, nearest_res) for depth_frame in depth_frames], dim=0)
            depth_frames = torch.stack([self.video_transforms(depth_frame) for depth_frame in depth_frames_resized], dim=0)
        else:
            depth_frames = torch.zeros_like(self.frames)
        return depth_frames

    def _load_and_process_label(self, label_path, frame_indices, nearest_frame_bucket, nearest_res):
        """加载和处理标签数据"""
        if label_path is not None:
            label_files = []
            for file in os.listdir(label_path.as_posix()):
                if file.startswith("label"):
                    label_files.append(file)
            label_files.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))

            masks = []
            hand_keypoints = []
            colored_masks = []
            for index in frame_indices[:nearest_frame_bucket]:
                file = label_files[index]
                label = np.load(label_path.joinpath(file))
                masks.append(label["seg"])
                colored_masks.append(convert_gray_to_color(label["seg"]))
                hand_keypoints.append(showHandJoints(np.zeros(list(nearest_res) + [3], dtype=np.uint8), label["joint_2d"][0]))

            # 处理语义分割和手部关键点
            colored_masks = torch.from_numpy(np.stack(colored_masks, axis=0)).float()
            colored_masks = colored_masks.permute(0, 3, 1, 2).contiguous()
            colored_masks = torch.stack([resize(colored_mask, nearest_res, interpolation=InterpolationMode.NEAREST) for colored_mask in colored_masks], dim=0)
            colored_masks = torch.stack([self.video_transforms(colored_mask) for colored_mask in colored_masks], dim=0)

            hand_keypoints = torch.from_numpy(np.stack(hand_keypoints, axis=0)).float()
            hand_keypoints = hand_keypoints.permute(0, 3, 1, 2).contiguous()
            hand_keypoints = torch.stack([resize(hand_keypoint, nearest_res, interpolation=InterpolationMode.NEAREST) for hand_keypoint in hand_keypoints], dim=0)
            hand_keypoints = torch.stack([self.video_transforms(hand_keypoint) for hand_keypoint in hand_keypoints], dim=0)

            # 掩码深度和法线帧
            masks = torch.from_numpy(np.stack(masks, axis=0))
            masks = torch.stack([resize(mask.unsqueeze(0), nearest_res, interpolation=InterpolationMode.NEAREST) for mask in masks], dim=0)
            masks[masks > 0] = 1

            depth_frames = depth_frames * masks
            normal_frames = normal_frames * masks

            if random.random() > 0.8:
                colored_masks = torch.zeros_like(self.frames)
            if random.random() > 0.8:
                hand_keypoints = torch.zeros_like(self.frames)
        else:
            colored_masks = torch.zeros_like(self.frames)
            hand_keypoints = torch.zeros_like(self.frames)
        return colored_masks, hand_keypoints
    
def __main__():
dataset = HOIVideoDatasetResizing(
    data_root="/path/to/videos",
    dataset_file="metadata.csv",
    video_column="video_path",
    caption_column="caption",
    frame_interval=8,  # 采样间隔
    device="cuda" if torch.cuda.is_available() else "cpu"
)

    sample = dataset[0]
    print(sample["descriptions"])  # 输出帧描述