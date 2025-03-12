import os
import cv2
import numpy as np

class NormalProcessor:
    def __init__(self, filelist_path, output_dir, normal_base_dir):
        """
        初始化法线视频处理器。
        
        Args:
            filelist_path (str): filelist 文件路径，包含法线视频的目录路径。
            output_dir (str): 输出目录。
            normal_base_dir (str): 法线视频的根目录。
        """
        self.filelist_path = filelist_path
        self.output_dir = output_dir
        self.normal_base_dir = normal_base_dir
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

    def read_filelist(self):
        """读取 filelist 文件，返回目录路径列表"""
        with open(self.filelist_path, "r") as f:
            return [line.strip() for line in f.readlines()]

    def generate_normal_video(self, normal_dir, output_path):
        """从指定目录读取法线视频并保存"""
        # 假设法线视频文件名为 normal.mp4
        normal_video_path = os.path.join(normal_dir, "video.mp4")
        if not os.path.exists(normal_video_path):
            print(f"错误: 未找到法线视频 {normal_video_path}")
            return None
        
        cap = cv2.VideoCapture(normal_video_path)
        if not cap.isOpened():
            print(f"错误: 无法打开法线视频 {normal_video_path}")
            return None
        
        # 获取视频属性
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # 默认 30 FPS
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 创建输出视频
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            frame_count += 1
        
        cap.release()
        out.release()
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"成功生成法线视频: {output_path}，帧数: {frame_count}")
            return output_path
        else:
            print(f"错误: 法线视频 {output_path} 生成失败或为空")
            return None

    def extract_first_frame(self, video_path, output_folder):
        """从视频中提取第一帧并保存为 normal.jpg"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")
        
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"无法读取视频首帧: {video_path}")
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_path = os.path.join(output_folder, "normal.jpg")
        cv2.imwrite(frame_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        
        cap.release()
        return frame_path

    def process(self):
        """主处理函数，从 filelist 读取目录路径，生成法线视频并截取第一帧"""
        sample_dirs = self.read_filelist()
        
        filelist_content = []
        
        for idx, sample_dir in enumerate(sample_dirs):
            # 转换为完整路径（如果不是绝对路径）
            if not os.path.isabs(sample_dir):
                sample_dir = os.path.join(self.normal_base_dir, sample_dir)
            sample_dir=sample_dir.replace("workspace/HOI-DiffusionAsShader/data/dexycb_videos",'workspace/DSINE/projects/dsine/dexycb_normal/dexycb_videos')
            sample_dir=os.path.dirname(sample_dir)
            # 使用 sample_00x 格式创建输出目录
            output_folder = os.path.join(self.output_dir, f"sample_{idx:03d}")
            os.makedirs(output_folder, exist_ok=True)
            
            if not os.path.exists(sample_dir):
                print(f"警告: 未找到法线目录 {sample_dir}，跳过 sample_{idx:03d}")
                continue
            
            # 生成法线视频
            normal_video_path = os.path.join(output_folder, "normal.mp4")
            print(f"生成法线视频: {normal_video_path}")
            video_path = self.generate_normal_video(sample_dir, normal_video_path)
            
            if video_path and os.path.exists(video_path):
                # 提取第一帧并保存为 normal.jpg
                normal_frame_path = self.extract_first_frame(video_path, output_folder)
                filelist_content.append(normal_frame_path)
            else:
                print(f"跳过 sample_{idx:03d}，因为法线视频生成失败")
        
        # 保存新的 filelist
        filelist_path = os.path.join(self.output_dir, "normal_filelist.txt")
        with open(filelist_path, "w") as f:
            f.write("\n".join(filelist_content))
        
        print(f"处理完成，法线 filelist 已保存至: {filelist_path}")

if __name__ == "__main__":
    filelist_path = "/data115/video-diff/workspace/HOI-DiffusionAsShader/data/dexycb_filelist/dexycb_depth_training_videos.txt"  # 假设文件名为 filelist.txt
    output_dir = "./processed_data_with_normal"
    normal_base_dir = "/data115/video-diff/workspace/HOI-DiffusionAsShader/"  # 法线视频根目录
    processor = NormalProcessor(filelist_path, output_dir, normal_base_dir)
    processor.process()