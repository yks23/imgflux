import os
import cv2
import numpy as np
import math
import sys

class HandPoseProcessor:
    def __init__(self, filelist_path, output_dir, dexycb_base_dir):
        """
        初始化手部关键点处理器。
        
        Args:
            filelist_path (str): filelist 文件路径，包含 DexYCB 数据集的目录路径。
            output_dir (str): 输出目录。
            dexycb_base_dir (str): DexYCB 数据集标注的根目录（未直接使用，因 filelist 已包含完整路径）。
        """
        self.filelist_path = filelist_path
        self.output_dir = output_dir
        self.dexycb_base_dir = dexycb_base_dir
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

    def read_filelist(self):
        """读取 filelist 文件，返回目录路径列表"""
        with open(self.filelist_path, "r") as f:
            return [line.strip() for line in f.readlines()]

    def showHandJoints(self, imgInOrg, gtIn):
        """绘制手部关键点和连接线"""
        imgIn = np.copy(imgInOrg)
        joint_color_code = [[139, 53, 255], [0, 56, 255], [43, 140, 237], [37, 168, 36], [147, 147, 0], [70, 17, 145]]
        limbs = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11], [11, 12],
                 [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
        limbs.reverse()
        PYTHON_VERSION = sys.version_info[0]
        gtIn = np.round(gtIn).astype(np.int32)

        try:
            if gtIn.shape[0] == 1:
                imgIn = cv2.circle(imgIn, center=(gtIn[0][0], gtIn[0][1]), radius=3, color=joint_color_code[0], thickness=-1)
            else:
                for joint_num in range(gtIn.shape[0]):
                    color_code_num = (joint_num // 4)
                    joint_color = list(map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num]))
                    cv2.circle(imgIn, center=(gtIn[joint_num][0], gtIn[joint_num][1]), radius=3, color=joint_color, thickness=-1)

                for limb_num in range(len(limbs)):
                    x1 = gtIn[limbs[limb_num][0], 1]
                    y1 = gtIn[limbs[limb_num][0], 0]
                    x2 = gtIn[limbs[limb_num][1], 1]
                    y2 = gtIn[limbs[limb_num][1], 0]
                    length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                    if length < 150 and length > 5:
                        deg = math.degrees(math.atan2(x1 - x2, y1 - y2))
                        polygon = cv2.ellipse2Poly((int((y1 + y2) / 2), int((x1 + x2) / 2)),
                                                   (int(length / 2), 6 if limb_num >= len(limbs) - 4 else 3),
                                                   int(deg), 0, 360, 1)
                        limb_color = list(map(lambda x: x + 35 * (limb_num % 4), joint_color_code[limb_num // 4]))
                        cv2.fillConvexPoly(imgIn, polygon, color=limb_color)
        except Exception as e:
            print(f"Error in drawing hand joints: {e}")
            return imgInOrg
        
        return imgIn

    def generate_hand_pose_video(self, frames_base, output_path):
        """从 DexYCB 数据集生成手部关键点视频"""
        hand_pose_files = [f for f in os.listdir(frames_base) if f.startswith('label')]
        hand_pose_files.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
        
        kps = []
        for file in hand_pose_files:
            label = np.load(os.path.join(frames_base, file))
            joint_2d = label['joint_2d'][0]  # 第一只手的 2D 关键点
            if (joint_2d == -1).any():
                kps.append(np.zeros((480, 640, 3), dtype=np.uint8))
            else:
                kps.append(self.showHandJoints(np.zeros((480, 640, 3), dtype=np.uint8), joint_2d))
        
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
        for frame in kps:
            out.write(frame)
        out.release()
        return output_path

    def extract_first_frame(self, video_path, output_folder):
        """从视频中提取第一帧并保存为 hand.jpg"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")
        
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"无法读取视频首帧: {video_path}")
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_path = os.path.join(output_folder, "hand.jpg")
        cv2.imwrite(frame_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        
        cap.release()
        return frame_path

    def process(self):
        """主处理函数，从 filelist 读取目录路径，生成手部关键点视频并截取第一帧"""
        sample_dirs = self.read_filelist()
        
        filelist_content = []
        
        for idx, sample_dir in enumerate(sample_dirs):
            # 转换为完整路径（如果不是绝对路径）
            if not os.path.isabs(sample_dir):
                sample_dir = os.path.join(self.dexycb_base_dir, sample_dir)
            sample_dir=os.path.dirname(sample_dir).replace("dexycb_depth",'dexycb')
            # 使用 sample_00x 格式创建输出目录
            output_folder = os.path.join(self.output_dir, f"sample_{idx:03d}")
            os.makedirs(output_folder, exist_ok=True)
            
            if not os.path.exists(sample_dir):
                print(f"警告: 未找到标注目录 {sample_dir}，跳过 sample_{idx:03d}")
                continue
            
            # 生成手部关键点视频
            hand_pose_video_path = os.path.join(output_folder, "hand_pose.mp4")
            self.generate_hand_pose_video(sample_dir, hand_pose_video_path)
            
            # 提取第一帧并保存为 hand.jpg
            hand_frame_path = self.extract_first_frame(hand_pose_video_path, output_folder)
            
            # 更新 filelist
            filelist_content.append(hand_frame_path)
        
        # 保存新的 filelist
        filelist_path = os.path.join(self.output_dir, "hand_filelist.txt")
        with open(filelist_path, "w") as f:
            f.write("\n".join(filelist_content))
        
        print(f"处理完成，手部关键点 filelist 已保存至: {filelist_path}")

if __name__ == "__main__":
    filelist_path = "/data115/video-diff/workspace/HOI-DiffusionAsShader/data/dexycb_filelist/dexycb_depth_training_depths.txt"  # 假设文件名为 filelist.txt
    output_dir = "./processed_data_with_hand"
    dexycb_base_dir = "/data115/video-diff/workspace/HOI-DiffusionAsShader/"  # DexYCB 数据集根目录
    processor = HandPoseProcessor(filelist_path, output_dir, dexycb_base_dir)
    processor.process()