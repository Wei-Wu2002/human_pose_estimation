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

    # 方案1：分别归一化 x 和 y 坐标
    x_min, x_max = seq[..., 0].min(), seq[..., 0].max()
    y_min, y_max = seq[..., 1].min(), seq[..., 1].max()

    seq[..., 0] = (seq[..., 0] - x_min) / (x_max - x_min + 1e-8)
    seq[..., 1] = (seq[..., 1] - y_min) / (y_max - y_min + 1e-8)

    return seq.reshape(20, 34)

def normalize_keypoints_sequence_with_conf(sequence: np.ndarray) -> np.ndarray:
    """
    对单个关键点序列进行 Min-Max 归一化 (仅对 x,y 进行归一化, conf 保留原值)。
    输入: sequence.shape == (20, 51)
    返回: shape == (20, 51)
    """
    seq = sequence.reshape(20, 17, 3)  # (T, num_joints, [x,y,conf])

    # 只对 x 和 y 进行归一化
    x_min, x_max = seq[..., 0].min(), seq[..., 0].max()
    y_min, y_max = seq[..., 1].min(), seq[..., 1].max()

    seq[..., 0] = (seq[..., 0] - x_min) / (x_max - x_min + 1e-8)
    seq[..., 1] = (seq[..., 1] - y_min) / (y_max - y_min + 1e-8)
    # seq[..., 2] = conf, 保持不变

    return seq.reshape(20, 51)

# def normalize_keypoints_sequence(sequence: np.ndarray, eps: float = 1e-8) -> np.ndarray:
#     """
#     以“根关节居中 + Frobenius 范数缩放”的方式标准化 2D 关键点序列。
#     输入:
#         sequence: shape = (T, 34) 或 (T, 17*2)，按 [x1,y1, x2,y2, ..., x17,y17]
#     输出:
#         same shape as input: (T, 34)
#     约定:
#         COCO 17 点，left_hip = 11, right_hip = 12，根关节 = (left_hip + right_hip)/2
#     """
#     T = sequence.shape[0]
#     K = 17
#     seq = sequence.reshape(T, K, 2).astype(np.float32)
#
#     LEFT_HIP, RIGHT_HIP = 11, 12
#
#     for t in range(T):
#         pts = seq[t]  # (17,2)
#         # 有效点：非全零
#         valid_mask = ~(np.isclose(pts[:, 0], 0.0) & np.isclose(pts[:, 1], 0.0))
#
#         if not valid_mask.any():
#             # 整帧都无效，跳过
#             continue
#
#         # 根关节：优先用骨盆（左右髋中点）；若无效则用所有有效点的质心
#         lh, rh = pts[LEFT_HIP], pts[RIGHT_HIP]
#         hips_valid = valid_mask[LEFT_HIP] and valid_mask[RIGHT_HIP]
#
#         if hips_valid:
#             root = 0.5 * (lh + rh)
#         else:
#             root = pts[valid_mask].mean(axis=0)
#
#         # 居中
#         centered = pts - root
#
#         # Frobenius 范数（对整帧 17×2）
#         frob = np.sqrt((centered**2).sum())
#
#         # 缩放（仅对有效点缩放；无效点保持 0）
#         if frob > eps:
#             centered /= frob
#
#         # 写回（无效点仍为 0）
#         centered[~valid_mask] = 0.0
#         seq[t] = centered
#
#     return seq.reshape(T, K * 2)



def extract_keypoints_from_video_with_conf(video_path: str, model: YOLO, sequence_length: int = 20,
                                           output_path: str = 'keypoints.npy'):
    num_keypoints = 17
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
            # xy: (17, 2), conf: (17,)
            xy = results.keypoints.xy[0].cpu().numpy()        # shape = (17, 2)
            conf = results.keypoints.conf[0].cpu().numpy()    # shape = (17,)

            # 拼接 xy 和 conf -> (17, 3)
            xyconf = np.concatenate([xy, conf[:, None]], axis=1)

            # 打印置信度
            for i, c in enumerate(conf):
                if c < 0.9:
                    print(f"\033[91mKeypoint {i}: {c:.3f}\033[0m")  # 红色
                else:
                    print(f"Keypoint {i}: {c:.3f}")

        else:
            continue

        keypoints_buffer.append(xyconf.flatten())  # shape = (51,)

        if len(keypoints_buffer) == sequence_length:
            break

    cap.release()

    keypoints_buffer = np.array(keypoints_buffer, dtype=np.float32)  # shape = (T, 51)
    # np.save(output_path, keypoints_buffer)
    # print(f'save to {output_path}')

    return keypoints_buffer


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

        # xy为未归一化的点，xyn为归一化后的位置信息
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
            continue

        keypoints_buffer.append(keypoints)

        if len(keypoints_buffer) == sequence_length:
            break

    cap.release()

    keypoints_buffer = np.array(keypoints_buffer, dtype=np.float32)
    # np.save(output_path, keypoints_buffer)
    # print(f'save to {output_path}')

    return keypoints_buffer
