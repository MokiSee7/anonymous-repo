import cv2
import os
import glob

# ========== 配置 ==========
datasets_root = "datasets"    # 所有数据集的根目录
frame_fps = 2           # 采样频率

# ========== 遍历所有数据集 ==========
dataset_dirs = [d for d in os.listdir(datasets_root) if os.path.isdir(os.path.join(datasets_root, d))]

for dataset in dataset_dirs:
    # 当前数据集的视频目录
    video_dir = os.path.join(datasets_root, dataset, "video")
    if not os.path.exists(video_dir):
        print(f"⚠️ 跳过 {dataset}, 因为没有 video/ 文件夹")
        continue

    # 获取当前数据集里的所有 mp4 文件
    video_paths = glob.glob(os.path.join(video_dir, "*.mp4"))
    print(f"📂 {dataset}: 找到 {len(video_paths)} 个视频")

    # 循环每个video进行抽帧
    for video_path in video_paths:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(datasets_root, dataset, "frames", video_name)
        # 对每个视频创建对应的输出文件夹
        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ 无法打开视频: {video_path}")
            continue

        # 读取视频的原始帧率
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        if original_fps <= 0:
            print(f"❌ 无法获取视频 FPS: {video_path}")
            cap.release()
            continue

        # 计算抽帧间隔
        frame_interval = max(1, int(round(original_fps / frame_fps)))
        frame_count = 0
        saved_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                output_path = os.path.join(output_dir, f"{video_name}_frame_{frame_count:04d}.jpg")
                cv2.imwrite(output_path, frame)
                saved_count += 1
            frame_count += 1

        cap.release()
        print(f"✅ {dataset}/{video_name}: 原始{original_fps}FPS, 抽帧 {saved_count} 张")

print("🎉 所有数据集抽帧完成！")