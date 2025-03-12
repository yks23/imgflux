import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn

class MultiModalDataset(Dataset):
    def __init__(self, filelist_path, base_dir, transform=None):
        """
        多模态数据集，加载 RGB、掩码、深度、描述、手部关键点、法线图。
        
        Args:
            filelist_path (str): 统一 filelist 文件路径，包含所有模态的路径。
            transform (callable, optional): 对图像的预处理变换。
        """
        self.transform = transform
        self.base_dir = base_dir
        with open(filelist_path, "r") as f:
            self.entries = [line.strip().split() for line in f.readlines()]
        
        for entry in self.entries:
            if len(entry) != 6:
                raise ValueError(f"filelist 每行应包含 6 列，实际为 {len(entry)}: {entry}")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        video_path, mask_path, depth_path, prompt_path, hand_path, normal_path = self.entries[idx]
        video_path = os.path.join(self.base_dir, video_path)
        mask_path = os.path.join(self.base_dir, mask_path)
        depth_path = os.path.join(self.base_dir, depth_path)
        prompt_path = os.path.join(self.base_dir, prompt_path)
        hand_path = os.path.join(self.base_dir, hand_path)
        normal_path = os.path.join(self.base_dir, normal_path)
        
        # 加载 RGB 视频帧
        video_img = cv2.imread(video_path)
        if video_img is None:
            raise ValueError(f"无法加载 RGB 图像: {video_path}")
        video_img = cv2.cvtColor(video_img, cv2.COLOR_BGR2RGB)
        
        # 加载掩码
        mask_img = cv2.imread(mask_path)
        if mask_img is None:
            raise ValueError(f"无法加载掩码图像: {mask_path}")
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)
        
        # 加载深度图
        depth_img = cv2.imread(depth_path)
        if depth_img is None:
            raise ValueError(f"无法加载深度图像: {depth_path}")
        depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2RGB)
        
        # 加载描述
        with open(prompt_path, "r") as f:
            prompt_text = f.read().strip()
        
        # 加载手部关键点图像（如果存在）
        hand_img = None
        if hand_path and os.path.exists(hand_path) and hand_path != '-':
            hand_img = cv2.imread(hand_path)
            if hand_img is not None:
                hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
        
        # 加载法线图（如果存在）
        normal_img = None
        if normal_path and os.path.exists(normal_path) and normal_path != '-':
            normal_img = cv2.imread(normal_path)
            if normal_img is not None:
                normal_img = cv2.cvtColor(normal_img, cv2.COLOR_BGR2RGB)
        
        # 应用图像变换
        if self.transform:
            video_img = self.transform(Image.fromarray(video_img))
            mask_img = self.transform(Image.fromarray(mask_img))
            depth_img = self.transform(Image.fromarray(depth_img))
            if hand_img is not None:
                hand_img = self.transform(Image.fromarray(hand_img))
            if normal_img is not None:
                normal_img = self.transform(Image.fromarray(normal_img))
        
        data = {
            "img": video_img,
            "mask": mask_img,
            "depth": depth_img,
            "prompt": prompt_text,
            "hand": hand_img if hand_img is not None else torch.zeros_like(video_img),
            "normal": normal_img if normal_img is not None else torch.zeros_like(video_img)
        }
        return data

def get_multiple_of_16_size(img):
    """计算调整后的尺寸，使宽高是 16 的倍数，同时尽量保持原始比例"""
    width, height = img.size  # PIL Image 的原始尺寸
    aspect_ratio = width / height
    
    # 以较小边为基础，调整到 16 的倍数
    if width < height:
        new_width = ((width // 16) + 1) * 16 if width % 16 != 0 else width
        new_height = int(new_width / aspect_ratio)
        new_height = ((new_height // 16) + 1) * 16 if new_height % 16 != 0 else new_height
    else:
        new_height = ((height // 16) + 1) * 16 if height % 16 != 0 else height
        new_width = int(new_height * aspect_ratio)
        new_width = ((new_width // 16) + 1) * 16 if new_width % 16 != 0 else new_width
    
    return (new_width, new_height)

# 定义动态调整大小的变换
class DynamicResize:
    def __init__(self):
        pass
    
    def __call__(self, img):
        target_size = get_multiple_of_16_size(img)
        return transforms.Resize(target_size)(img)

# 定义图像预处理变换
transform = transforms.Compose([
    transforms.Resize((480, 640), interpolation=transforms.InterpolationMode.LANCZOS),  # 调整大小为 16 的倍数
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])

def concatenate_and_save(batch, batch_idx, output_dir="./concat"):
    """
    将 batch 中的 depth、normal 和 hand 图像水平拼接，并保存到指定目录。
    
    Args:
        batch (dict): DataLoader 返回的批次数据，包含 'depth', 'normal', 'hand' 等键。
        batch_idx (int): 批次索引，用于命名文件。
        output_dir (str): 保存拼接图像的目录。
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取批次大小
    batch_size = batch["depth"].shape[0]

    for i in range(batch_size):
        # 提取单张图像（从张量转换为 NumPy 数组）
        depth_img = batch["depth"][i].permute(1, 2, 0).numpy()  # [H, W, C]
        normal_img = batch["normal"][i].permute(1, 2, 0).numpy()  # [H, W, C]
        hand_img = batch["hand"][i].permute(1, 2, 0).numpy()  # [H, W, C]

        # 反标准化（将值从 [-2, 2] 转换回 [0, 1]）
        depth_img = (depth_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
        normal_img = (normal_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
        hand_img = (hand_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))

        # 确保值在 [0, 1] 范围内
        depth_img = np.clip(depth_img, 0, 1)
        normal_img = np.clip(normal_img, 0, 1)
        hand_img = np.clip(hand_img, 0, 1)

        # 转换为 0-255 的 uint8 格式
        depth_img = (depth_img * 255).astype(np.uint8)
        normal_img = (normal_img * 255).astype(np.uint8)
        hand_img = (hand_img * 255).astype(np.uint8)

        # 水平拼接（确保尺寸一致）
        # 获取最小高度和宽度（假设所有图像经过相同变换，尺寸应为 480x640）
        min_height = min(depth_img.shape[0], normal_img.shape[0], hand_img.shape[0])
        min_width = min(depth_img.shape[1], normal_img.shape[1], hand_img.shape[1])

        # 裁剪或调整为相同尺寸
        depth_img = depth_img[:min_height, :min_width, :]
        normal_img = normal_img[:min_height, :min_width, :]
        hand_img = hand_img[:min_height, :min_width, :]

        # 水平拼接
        concatenated_img = np.hstack([depth_img, normal_img, hand_img])

        # 保存拼接后的图片
        filename = f"batch_{batch_idx}_{i}_concat.png"
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, cv2.cvtColor(concatenated_img, cv2.COLOR_RGB2BGR))
        print(f"已保存拼接图像到: {save_path}")

def get_dataloader(**kwargs):
    filelist_path = "./data/filelist.txt"  # 统一 filelist 文件路径
    dataset = MultiModalDataset(
        filelist_path=filelist_path,
        transform=transform,
        base_dir='/data115/video-diff/Kaisen.Yang/workspace/x-flux/data'
    )
    dataloader = DataLoader(
        dataset,
        **kwargs
        # batch_size=4,       # 批次大小
        # shuffle=True,       # 是否随机打乱
        # num_workers=2       # 多线程加载
    )
    return dataloader

def get_transform():
    return transform

def get_resize(h, w):
    return transforms.Compose([
        transforms.Resize((h, w), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

if __name__ == "__main__":
    # 创建 DataLoader
    dataloader = get_dataloader(batch_size=1, shuffle=False, num_workers=2)

    # 遍历 DataLoader 并拼接保存
    batch_idx = 0
    for batch in dataloader:
        concatenate_and_save(batch, batch_idx, output_dir="./concat")
        batch_idx += 1