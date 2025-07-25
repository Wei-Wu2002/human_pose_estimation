import os

import cv2
import numpy as np
from ultralytics import YOLO


def normalize_keypoints_sequence(sequence: np.ndarray) -> np.ndarray:
    """
    对单个关键点序列进行 Min-Max 归一化。
    输入: sequence.shape == (20, 34)
    返回: shape == (20, 34)
    """
    seq = sequence.reshape(20, 17, 2)

    # 分别归一化 x 和 y 坐标
    x_min, x_max = seq[..., 0].min(), seq[..., 0].max()
    y_min, y_max = seq[..., 1].min(), seq[..., 1].max()

    seq[..., 0] = (seq[..., 0] - x_min) / (x_max - x_min + 1e-8)
    seq[..., 1] = (seq[..., 1] - y_min) / (y_max - y_min + 1e-8)

    return seq.reshape(20, 34)


def extract_keypoints_from_video(video_path: str, model: YOLO, sequence_length: int = 20,
                                 output_path: str = 'keypoints.npy'):
    num_keypoints = 17 * 2
    frame_count = 0  # 初始化帧编号

    if not os.path.exists(video_path):
        raise FileNotFoundError(f'The video file {video_path} does not exist')

    cap = cv2.VideoCapture(video_path)
    keypoints_buffer = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Video terminado

        results = model(frame)[0]
        frame_count += 1
        # print(f"处理第 {frame_count} 帧")

        if len(results.keypoints.xy) > 0:
            keypoints = results.keypoints.xy[0].cpu().numpy().flatten()
            if keypoints.shape[0] != num_keypoints:
                keypoints = np.pad(keypoints, (0, num_keypoints - keypoints.shape[0]))
        else:
            continue

        keypoints_buffer.append(keypoints)

        if len(keypoints_buffer) == sequence_length:
            break

    cap.release()

    keypoints_buffer = np.array(keypoints_buffer, dtype=np.float32)
    # np.save(output_path, keypoints_buffer)
    # print(f'save to {output_path}')

    return keypoints_buffer
