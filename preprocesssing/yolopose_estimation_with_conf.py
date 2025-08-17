import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

from configs.config import YOLO_MODEL_DIR, YouTube_DIR
from configs.logger import logger
from utils.video_to_keypoints import extract_keypoints_from_video_with_conf, normalize_keypoints_sequence_with_conf

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Keypoint extraction from video datasets")
    parser.add_argument('--name', type=str, choices=['Youtube'], required=True,
                        help='Dataset name: Youtube or UR')
    args = parser.parse_args()

    if args.name == 'Youtube':
        dataset_path = YouTube_DIR / 'videos' / 'falls'

        video_folders = {
            Path(YouTube_DIR) / 'videos' / 'falls': 1,
            Path(YouTube_DIR) / 'videos' / 'normal': 0,
            Path(YouTube_DIR) / 'videos' / 'no_fall_static': 0
        }

    else:
        raise ValueError(f"Unknown dataset name: {args.name}")

    sequence_length = 20
    keypoints_sequences = []
    labels = []
    csv_records = []
    save_root = Path(YouTube_DIR) / 'processed_with_conf'

    model = YOLO(YOLO_MODEL_DIR, task='pose')
    for folder_path, label in video_folders.items():
        if not folder_path.exists():
            logger.warning(f"Folder does not exist: {folder_path}")
            continue

        video_files = list(folder_path.glob("*.mp4"))
        if not video_files:
            logger.info(f"No .mp4 videos found in {folder_path}")
            continue

        dataset_name = folder_path.name
        logger.info(f"Processing {len(video_files)} videos in {dataset_name}")

        for video_idx, video_path in enumerate(tqdm(video_files, desc=f"Processing {dataset_name}")):
            try:
                keypoints = extract_keypoints_from_video_with_conf(str(video_path), model, sequence_length=sequence_length)
                print(f'keypoints shape: {keypoints.shape}')
                keypoints = normalize_keypoints_sequence_with_conf(keypoints)
                keypoints_sequences.append(keypoints)
                labels.append(label)

                # 保存为 .npy 文件
                save_name = f"{video_path.stem}.npy"
                save_path = save_root / 'data' /save_name
                np.save(save_path, keypoints)

                # 记录 CSV 元数据
                csv_records.append({
                    "video": video_path.name,
                    "index": video_idx,
                    "label": label,
                    "dataset": dataset_name,
                    "npy_path": str(save_path)
                })

                logger.debug(f"Saved keypoints to {save_path}")
            except Exception as e:
                logger.error(f"Failed to process {video_path.name}: {str(e)}")

    keypoints_sequences = np.array(keypoints_sequences)
    labels = np.array(labels)

    # 保存 CSV 文件
    csv_df = pd.DataFrame(csv_records)
    csv_save_path = save_root / Path(f"extracted_labels_{args.name}.csv")
    csv_df.to_csv(csv_save_path, index=False)

    logger.info(f"Saved label CSV to {csv_save_path}")
    logger.info(f"keypoints_sequences shape: {keypoints_sequences.shape}")
    logger.info(f"labels shape: {labels.shape}")
