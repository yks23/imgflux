import os
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaNextForConditionalGeneration
import shutil

class DataProcessor:
    def __init__(self, videolist_path, masklist_path, output_dir, k=5):
        """
        初始化数据处理器，使用 LLaVA v1.6 Mistral-7B 模型。
        
        Args:
            videolist_path (str): 视频列表文件路径。
            masklist_path (str): 掩码视频列表文件路径。
            output_dir (str): 输出目录。
            k (int): 等间距抽取的帧数，默认 5。
        """
        self.videolist_path = videolist_path
        self.masklist_path = masklist_path
        self.output_dir = output_dir
        self.k = k
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 加载 LLaVA v1.6 Mistral-7B 模型和处理器
        self.llava_processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        self.llava_model = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf",
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

    def read_list(self, file_path):
        """读取文件列表并返回路径列表"""
        with open(file_path, "r") as f:
            return [line.strip() for line in f.readlines()]

    def get_depth_path(self, mask_path):
        """将 masklist 中的 'masked_color.mp4' 替换为 'depth.mp4'"""
        return mask_path.replace("masked_color.mp4", "depth.mp4")

    def extract_frames(self, video_path, num_frames, output_folder, prefix):
        """从视频中等间距提取帧并保存，对掩码帧进行三通道二值化处理"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(1, total_frames // num_frames)
        frame_paths = []
        
        for i in range(num_frames):
            frame_idx = i * interval
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 如果是掩码帧，进行三通道二值化处理
            if prefix == "mask":
                # 检查三个通道是否都为 0
                mask = np.all(frame_rgb == 0, axis=2)  # [H, W]，True 表示全零（黑色）
                # 二值化：全零保持黑色 (0, 0, 0)，其他设为白色 (255, 255, 255)
                frame_rgb = np.where(mask[..., None], 0, 255).astype(np.uint8)
            
            frame_path = os.path.join(output_folder, f"{prefix}_{i:03d}.png")
            cv2.imwrite(frame_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            frame_paths.append(frame_path)
        
        cap.release()
        return frame_paths

    def generate_description(self, image_path):
        """使用 LLaVA v1.6 Mistral-7B 模型生成图像描述"""
        image = Image.open(image_path).convert("RGB")
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
    
    def process(self):
        """主处理函数"""
        video_paths = self.read_list(self.videolist_path)
        mask_paths = self.read_list(self.masklist_path)
        depth_paths = [self.get_depth_path(mask_path) for mask_path in mask_paths]
        
        base_data_dir = '/data115/video-diff/workspace/HOI-DiffusionAsShader/'
        video_paths = [os.path.join(base_data_dir, video_path) for video_path in video_paths]
        mask_paths = [os.path.join(base_data_dir, mask_path) for mask_path in mask_paths]
        depth_paths = [os.path.join(base_data_dir, depth_path) for depth_path in depth_paths]
        
        if not (len(video_paths) == len(mask_paths) == len(depth_paths)):
            raise ValueError("videolist, masklist 和 depth 列表长度不匹配")
        
        filelist_content = []
        
        for idx, (video_path, mask_path, depth_path) in enumerate(zip(video_paths, mask_paths, depth_paths)):
            if idx<=373:
                continue
            print(f"处理第 {idx} 个视频")
            subfolder_name = f"sample_{idx:03d}"
            output_folder = os.path.join(self.output_dir, subfolder_name)
            os.makedirs(output_folder, exist_ok=True)
            
            video_frames = self.extract_frames(video_path, self.k, output_folder, "video")
            mask_frames = self.extract_frames(mask_path, self.k, output_folder, "mask")
            depth_frames = self.extract_frames(depth_path, self.k, output_folder, "depth")
            
            for i, frame_path in enumerate(video_frames):
                description = self.generate_description(frame_path)
                prompt_file = os.path.join(output_folder, f"prompt_{i:03d}.txt")
                with open(prompt_file, "w") as f:
                    f.write(description)
            
            for i in range(self.k):
                filelist_entry = (
                    f"{output_folder}/video_{i:03d}.png,"
                    f"{output_folder}/mask_{i:03d}.png,"
                    f"{output_folder}/depth_{i:03d}.png,"
                    f"{output_folder}/prompt_{i:03d}.txt"
                )
                filelist_content.append(" ".join(filelist_entry))
        filelist_path = os.path.join(self.output_dir, "filelist.txt")
        with open(filelist_path, "w") as f:
            f.write("\n".join(filelist_content))
        
        print(f"处理完成，总 filelist 已保存至: {filelist_path}")

if __name__ == "__main__":
    videolist_path = "/data115/video-diff/workspace/HOI-DiffusionAsShader/data/dexycb_filelist/dexycb_depth_training_videos.txt"
    masklist_path = "/data115/video-diff/workspace/HOI-DiffusionAsShader/data/dexycb_filelist/dexycb_depth_training_depths.txt"
    output_dir = "/data115/video-diff/Kaisen.Yang/workspace/x-flux/data/processed_data"
    processor = DataProcessor(videolist_path, masklist_path, output_dir, k=8)
    processor.process()