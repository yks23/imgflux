import cv2
import numpy as np

def process_video(video_path, output_image_path):
    """
    打开 RGB 视频，截取第一帧，存储为图片，并输出图片参数。
    
    Args:
        video_path (str): 输入视频的文件路径。
        output_image_path (str): 输出图片的保存路径。
    
    Returns:
        dict: 包含图片参数的字典。
    """
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    # 检查视频是否成功打开
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    # 读取第一帧
    ret, frame = cap.read()
    if not ret:
        raise ValueError("无法读取视频的第一帧")
    
    # 释放视频对象
    cap.release()
    
    # OpenCV 的帧是 BGR 格式，转换为 RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 保存图片
    cv2.imwrite(output_image_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))  # 保存时转换回 BGR
    
    # 获取图片参数
    image_params = {
        "shape": frame_rgb.shape,          # 形状 (高度, 宽度, 通道数)
        "height": frame_rgb.shape[0],      # 高度
        "width": frame_rgb.shape[1],       # 宽度
        "channels": frame_rgb.shape[2],    # 通道数
        "dtype": str(frame_rgb.dtype),     # 数据类型
        "size_bytes": frame_rgb.nbytes     # 占用字节数
    }
    
    # 输出参数
    print("图片参数:")
    for key, value in image_params.items():
        print(f"{key}: {value}")
    
    return image_params

# 示例用法
if __name__ == "__main__":
    # 输入视频路径和输出图片路径
    video_path = "/data115/video-diff/workspace/HOI-DiffusionAsShader/data/dexycb_depth/20200709-subject-01/20200709_141754/836212060125/depth.mp4"      # 替换为你的视频路径
    output_image_path = "first_frame.jpg"  # 替换为输出图片路径
    
    try:
        params = process_video(video_path, output_image_path)
        print(f"第一帧已保存至: {output_image_path}")
    except Exception as e:
        print(f"错误: {e}")