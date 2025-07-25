import argparse
import time
from pathlib import Path

import numpy as np

from configs.logger import logger
from dataset.dataloader import get_dataloaders


def estimate_height(sequence: np.ndarray) -> float:
    """
    输入：sequence shape = (10, 17, 2)
    返回：估计身高（鼻子与左脚踝之间的最大 y 差）
    """
    return max(abs(frame[0][1] - frame[15][1]) for frame in sequence)


def condition1_met(sequence: np.ndarray, fps: int = 30):
    keypoints = sequence.reshape(-1, 17, 2)
    # print(f"keypoints.shape: {keypoints.shape}")
    frame1 = keypoints[0]
    # print(f"frame1.shape: {frame1.shape}")
    frame10 = keypoints[len(keypoints) - 1]
    # print(f"frame10.shape: {frame10.shape}")

    # 对应关键点
    nose_y1 = frame1[0][1]
    neck_y1 = (frame1[5][1] + frame1[6][1]) / 2
    l_hip_y1 = frame1[11][1]
    r_hip_y1 = frame1[12][1]

    nose_y10 = frame10[0][1]
    neck_y10 = (frame10[5][1] + frame10[6][1]) / 2
    l_hip_y10 = frame10[11][1]
    r_hip_y10 = frame10[12][1]

    # 构建向量
    y1 = np.array([nose_y1, neck_y1, l_hip_y1, r_hip_y1])
    y10 = np.array([nose_y10, neck_y10, l_hip_y10, r_hip_y10])

    v = np.sum(y10 - y1) / (3 * (9 / fps))
    height = estimate_height(keypoints)
    v_th = height * 0.23
    return v >= v_th


def condition2_met(keypoints: np.ndarray):
    frame = keypoints[-1]
    neck_y = (frame[5][1] + frame[6][1]) / 2
    waist_y = (frame[11][1] + frame[12][1]) / 2
    delta_y = abs(neck_y - waist_y)
    height = estimate_height(keypoints)
    h1 = height * 0.1
    return delta_y <= h1


def condition3_met(keypoints: np.ndarray):
    frame = keypoints[-1]
    height = estimate_height(keypoints)
    left_diff = abs(frame[13][1] - frame[15][1])
    right_diff = abs(frame[14][1] - frame[16][1])
    threshold = height * 0.12
    return left_diff <= threshold or right_diff <= threshold


def detect_fall_single_window(sequence: np.ndarray, fps: int = 30) -> bool:
    keypoints = sequence.reshape(-1, 17, 2)
    if not condition1_met(sequence, fps=fps):
        print("Condition 1 not met: centroid velocity too low")
        return False

    if not condition2_met(keypoints):
        print("Condition 2 not met: upper body (neck to waist) distance too large")
        return False

    if not condition3_met(keypoints):
        print("Condition 3 not met: leg to ankle difference too large")
        return False

    print("All conditions met: fall detected")
    return True


def detect_fall_sliding(sequence: np.ndarray, window_size: int = 10, fps: int = 30) -> np.ndarray:
    """
    对完整序列进行滑动窗口检测
    sequence: shape (seq_len=20, 34)
    return: (seq_len - window_size + 1,) array of predictions (0 or 1)
    """
    seq_len = sequence.shape[0]
    print(f"seq_len: {seq_len}")
    outputs = []
    for i in range(seq_len - window_size + 1):
        window = sequence[i:i + window_size]
        label = detect_fall_single_window(window, fps=fps)
        outputs.append(int(label))
    return np.array(outputs)


import matplotlib.pyplot as plt
import numpy as np


def plot_binary_output_bar(outputs, save_path=None):
    """
    绘制一个 1xN 的黑白方格图，其中 1=黑色，0=白色。
    """
    binary_array = np.array(outputs).reshape(1, -1)  # shape: (1, T)

    plt.figure(figsize=(len(outputs), 1))  # 宽度随时间长度自动伸缩
    plt.imshow(binary_array, cmap='gray', aspect='auto', vmin=0, vmax=1)

    plt.xticks(range(len(outputs)))
    plt.yticks([])

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()

def plot_outputs_grid_grouped(output_list, y_labels=None, save_path=None):
    """
    将多个输出序列绘制在一张图中，每行一个 sample，纵轴为视频文件名
    """
    n = len(output_list)
    T = len(output_list[0]) if n > 0 else 0
    grid = np.array(output_list).reshape(n, T)

    fig, ax = plt.subplots(figsize=(T, n * 0.5))
    ax.imshow(grid, cmap='gray_r', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(T))
    ax.set_yticks(range(n))
    ax.set_yticklabels(y_labels if y_labels else [f"Sample {i}" for i in range(n)])

    ax.set_xlabel("Frame Index")
    ax.set_title("Sliding Window Detection Results")

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Handcrafted feature loading test")
    parser.add_argument('--csv_path', type=str, default='dataset/YouTube/processed/extracted_labels_Youtube.csv',
                        help='Path to CSV file containing npy paths and labels')
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    start = time.time()

    logger.info(f"Loading dataset from: {csv_path}")
    X, y, names = get_dataloaders(csv_path=csv_path, mode='handcrafted')

    logger.info(f"Dataset loaded in {time.time() - start:.2f} seconds")
    logger.info(f"X.shape = {X.shape}")
    logger.info(f"y.shape = {y.shape}")
    logger.info(f"Number of classes: {len(set(y))}")

    # 数据合法性检查
    assert X.shape[0] == y.shape[0], "Mismatch between X and y lengths"
    assert len(X.shape) == 3, "Expected shape [samples, seq_len, feat_dim]"

    gt1_outputs = []
    gt0_outputs = []
    gt1_names = []
    gt0_names = []

    for (i, x) in enumerate(X):
        outputs = detect_fall_sliding(x, window_size=10)
        print(f"Sample {i} outputs: {outputs}, Ground truth: {y[i]}")
        if y[i] == 1:
            gt1_outputs.append(outputs)
            gt1_names.append(names[i])
        else:
            gt0_outputs.append(outputs)
            gt0_names.append(names[i])

    # 绘制所有 GT=1 的输出
    plot_outputs_grid_grouped(gt1_outputs, y_labels=gt1_names)

    # 绘制所有 GT=0 的输出
    plot_outputs_grid_grouped(gt0_outputs, y_labels=gt0_names)

