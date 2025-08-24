import argparse
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from configs.config import YOLO_MODEL_DIR, YouTube_DIR, URFD_DIR
from configs.logger import logger


def extract_keypoints_from_paths(img_paths: List[Path], model: YOLO, num_keypoints: int = 17 * 2) -> np.ndarray:
    """遍历给定的图像路径列表，提取关键点，并返回关键点数组"""
    keypoints_buffer = []

    for img_path in img_paths:
        print(f'Processing {img_path}')
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f'Warning: Could not read {img_path}')
            continue

        results = model(frame, verbose=False)[0]

        if len(results.keypoints.xy) > 0:
            keypoints = results.keypoints.xy[0].cpu().numpy().flatten()
            # conf 识别的置信度
            conf = results.keypoints.conf[0].cpu().numpy()

            for i, c in enumerate(conf):
                if c < 0.9:
                    print(f"\033[91mKeypoint {i}: {c:.3f}\033[0m")  # 红色
                else:
                    print(f"Keypoint {i}: {c:.3f}")  # 默认颜色
            if keypoints.shape[0] != num_keypoints:
                keypoints = np.pad(keypoints, (0, num_keypoints - keypoints.shape[0]))
        else:
            keypoints = np.zeros(num_keypoints, dtype=np.float32)

        keypoints_buffer.append(keypoints)

    if not keypoints_buffer:
        print("No keypoints extracted.")
        return None

    keypoints_array = np.array(keypoints_buffer, dtype=np.float32)
    return keypoints_array


def find_image_by_index(img_folder: Path, folder: str, idx: int, zero_pad_width: int = 3):
    """
    正确的文件名模式:
    <folder>-cam0-rgb-XXX.png / .jpg
    例如: adl-01-cam0-rgb-001.png
    """
    name = f"{folder}-cam0-rgb-{idx:0{zero_pad_width}d}"
    for ext in (".png", ".jpg", ".jpeg"):
        p = img_folder / f"{name}{ext}"
        if p.exists():
            return p
    return None


# 标签映射：-1 -> 0 (正常)，0/1 -> 1 (跌倒)
def map_label(raw_label: int):
    if raw_label in (0, 1):
        return 1
    if raw_label == -1:
        return 0
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Keypoint extraction from video datasets")
    parser.add_argument('--name', type=str, choices=['Youtube', 'URFD'], required=True,
                        help='Dataset name: Youtube or UR')
    parser.add_argument('--conf', type=str, choices=['xy', 'xyc'], default='xy',
                        help='Output format: "xy" for (x,y) only; "xyc" for (x,y,conf).')
    args = parser.parse_args()

    if args.name == 'Youtube':
        video_folders = {
            Path(YouTube_DIR) / 'videos' / 'falls': 1,
            Path(YouTube_DIR) / 'videos' / 'normal': 0,
            Path(YouTube_DIR) / 'videos' / 'no_fall_static': 0
        }
    elif args.name == 'URFD':
        img_folders = {
            Path(URFD_DIR) / 'img' / 'Fall': 1,
            Path(URFD_DIR) / 'img' / 'ADL': 0,
        }

        adls_csv = Path(URFD_DIR) / 'urfall-cam0-adls.csv'
        falls_csv = Path(URFD_DIR) / 'urfall-cam0-falls.csv'
    else:
        raise ValueError(f"Unknown dataset name: {args.name}")

    print(f'img_folders: {img_folders}')

    # 读取标签 CSV
    # print(f'adls_csv: {adls_csv}')
    # adls_df = pd.read_csv(adls_csv)
    print(f'falls_csv: {falls_csv}')
    falls_df = pd.read_csv(falls_csv)

    # 加载YOLO模型
    model = YOLO(YOLO_MODEL_DIR, task='pose')

    csv_records = []
    sequence_length = 20

    save_root = Path(URFD_DIR) / 'processed'

    # 获取 folder 列所有不重复项目
    # unique_folders = adls_df['folder'].unique()
    unique_folders = falls_df['folder'].unique()

    # 循环遍历
    for folder in unique_folders:
        print("正在处理:", folder)

        # 取出该 folder 的所有行
        # folder_df = adls_df[adls_df["folder"] == folder]
        folder_df = falls_df[falls_df["folder"] == folder]
        print(folder_df)

        img_folder = Path(URFD_DIR) / 'img' / 'Fall' / f"{folder}-cam0-rgb" / f"{folder}-cam0-rgb"
        # img_folder = Path(URFD_DIR) / 'img' / 'ADL' / f"{folder}-cam0-rgb" / f"{folder}-cam0-rgb"

        if not img_folder.exists():
            print(f"⚠️ 文件夹不存在: {img_folder}")
            continue

        img_files = []
        labels = []
        for row in folder_df.itertuples(index=False):
            img_index = int(getattr(row, "index"))
            raw_label = int(getattr(row, "label"))
            label = map_label(raw_label)
            labels.append(label)
            img_path = find_image_by_index(img_folder, folder, img_index)

            print(f'img_path: {img_path}')
            img_files.append(img_path)

        print(f'img_files: {len(img_files)}')
        print(f'labels: {len(labels)}')
        keypoints = extract_keypoints_from_paths(img_files, model)

        # 保存为 .npy 文件
        save_name = f"{folder}.npy"
        save_path = save_root / 'falls' / save_name
        # save_path = save_root / 'adls' / save_name
        np.save(save_path, keypoints)

        # 记录 CSV 元数据
        csv_records.append({
            "video": folder,
            "label": labels,
            "npy_path": str(save_path)
        })

        logger.debug(f"Saved keypoints to {save_path}")

    # 保存 CSV 文件
    csv_df = pd.DataFrame(csv_records)
    csv_save_path = save_root / Path(f"extracted_labels_falls_{args.name}.csv")

    csv_df.to_csv(csv_save_path, index=False)

    logger.info(f"Saved label CSV to {csv_save_path}")

