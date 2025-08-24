# -*- coding: utf-8 -*-
import argparse
import ast
import time
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader, Dataset

from configs.logger import logger
from src.LSTM import LSTM


def load_dataset(csv_path: Path):
    df = pd.read_csv(csv_path)

    npy_paths = df['npy_path'].tolist()
    names = df['video'].tolist()
    raw_labels = df['label'].tolist()

    # 自动解析 label 字符串为列表
    labels = []
    for l in raw_labels:
        if isinstance(l, str):
            parsed = ast.literal_eval(l.strip())
            labels.append(parsed)
        else:
            raise ValueError(f"标签格式错误: {l}")

    X = [np.load(path) for path in npy_paths]
    X = np.array(X, dtype=object)
    y = np.array([np.array(lbl) for lbl in labels], dtype=object)
    return X, y, names


class KeypointDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        if self.transform:
            x = self.transform(x)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def evaluate_model(model, test_loader, device, threshold=0.7):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device).float()
            yb = yb.to(device).float()

            clip_out, _ = model(xb)  # clip_out: (B, 1)
            print(clip_out)
            preds = (clip_out.squeeze(1) > threshold).float()

            all_preds.append(preds.cpu())
            all_labels.append(yb.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1_score": f1_score(all_labels, all_preds, zero_division=0),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "frame_preds": all_preds,
        "frame_labels": all_labels,
    }


def build_model_from_ckpt(ckpt_dict: dict, cli_args) -> LSTM:
    hp = ckpt_dict.get("hyperparameters", {}) if isinstance(ckpt_dict, dict) else {}
    input_size = hp.get("input_size", cli_args.input_size)
    hidden_size = hp.get("hidden_size", cli_args.hidden_size)
    num_layers = hp.get("num_layers", cli_args.num_layers)
    dropout = hp.get("dropout", cli_args.dropout)
    use_sigmoid = True
    output_size = 1

    print(f"[✓] Reconstructed LSTM with input={input_size}, hidden={hidden_size}, layers={num_layers}")
    return LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size,
        dropout=dropout,
        use_sigmoid=use_sigmoid
    )


def evaluate_model_frame_level_sliding(
        model,
        X,
        y,
        names,
        device,
        window_size=20,
        stride=1,
        batch_size=32,
        save_path=None
):
    """
    用滑动窗口方式评估模型的帧级别性能。

    Args:
        model: 已加载的 PyTorch 模型
        X: np.ndarray of shape (N,)，每个元素为 (T, D) 的 numpy array
        y: np.ndarray of shape (N,)，每个元素为 (T,) 的标签列表
        names: list[str]，每段序列的视频名
        device: torch.device
        window_size: 滑动窗口长度
        stride: 滑动窗口步长
        batch_size: 批大小
        save_path: 可选，保存预测 CSV 文件路径

    Returns:
        dict: 包含各项评估指标
    """

    # Step 1: 滑窗生成 clip 数据
    X_clips, y_clips, clip_names = [], [], []

    for seq_x, seq_y, name in zip(X, y, names):
        T = len(seq_x)
        for t in range(window_size, T + 1, stride):
            x_clip = seq_x[t - window_size:t]  # (W, D)
            y_clip = seq_y[t - 1]  # 最后一帧标签
            X_clips.append(x_clip)
            y_clips.append(y_clip)
            clip_names.append(f"{name}_frame{t - 1}")

    print(f"[✓] Total clips: {len(X_clips)}")

    # Step 2: 构造 DataLoader
    dataset = KeypointDataset(X_clips, y_clips)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Step 3: 模型评估
    print("[✓] Evaluating...")
    t0 = time.time()
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device).float()
            yb = yb.to(device).float()

            clip_out, _ = model(xb)  # (B, 1)
            # print(clip_out)
            preds = (clip_out.squeeze(1) > 0.5).float()

            all_preds.append(preds.cpu())
            all_labels.append(yb.cpu())

    # Step 4: 聚合结果
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    far = fp / (fp + tn + 1e-8)  # False Alarm Rate（避免除0）

    result_dict = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1_score": f1_score(all_labels, all_preds, zero_division=0),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "far": far,
        "frame_preds": all_preds,
        "frame_labels": all_labels,
        "clip_names": clip_names
    }

    logger.info(f"\n[✓] Done in {time.time() - t0:.2f}s")
    logger.info(f"Accuracy : {result_dict['accuracy'] * 100:.2f}%")
    logger.info(f"Precision: {result_dict['precision']:.4f}")
    logger.info(f"Recall   : {result_dict['recall']:.4f}")
    logger.info(f"F1 Score : {result_dict['f1_score']:.4f}")
    logger.info(f"FAR      : {result_dict['far']:.4f}")
    logger.info(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}")

    # Step 5: 保存预测（可选）
    if save_path:
        df = pd.DataFrame({
            "name": clip_names,
            "label": all_labels,
            "pred": all_preds
        })
        df.to_csv(save_path, index=False)
        print(f"[✓] Saved predictions to: {save_path}")

    return result_dict


import numpy as np
import torch
import os
import matplotlib.pyplot as plt


def compute_ttd_single(gt: np.ndarray, pred: np.ndarray):
    """计算单个序列的 TTD"""
    if np.sum(gt) == 0:
        return None  # 无跌倒事件，跳过

    gt_start = np.argmax(gt > 0)
    pred_pos = np.where(pred > 0)[0]

    if len(pred_pos) == 0:
        return np.inf  # 没检测到
    pred_start = pred_pos[0]
    return gt_start - pred_start  # 正值表示提前检测


import matplotlib.pyplot as plt
import numpy as np
import os


def plot_ttd_distribution(ttd_results, save_path="ttd_hist.png", title="TTD Distribution"):
    """
    绘制 TTD 的分布直方图，并标注 P50/P90/P95/Max。

    Args:
        ttd_results: dict[str, float], 每段视频的 TTD 值（单位：帧）
        save_path: str, 图像保存路径
        title: str, 图像标题
    """
    ttd_values = np.array([v for v in ttd_results.values() if np.isfinite(v)])
    if len(ttd_values) == 0:
        print("[✗] 无有效 TTD 可用于绘图")
        return

    p50 = np.percentile(ttd_values, 50)
    p90 = np.percentile(ttd_values, 90)
    p95 = np.percentile(ttd_values, 95)
    max_v = np.max(ttd_values)

    plt.figure(figsize=(8, 5))
    plt.hist(ttd_values, bins=20, color="skyblue", edgecolor="black", alpha=0.7)
    plt.axvline(p50, color='orange', linestyle='--', label=f'P50: {p50:.1f}')
    plt.axvline(p90, color='red', linestyle='--', label=f'P90: {p90:.1f}')
    plt.axvline(p95, color='purple', linestyle='--', label=f'P95: {p95:.1f}')
    plt.axvline(max_v, color='gray', linestyle='--', label=f'Max: {max_v:.1f}')

    plt.title(title)
    plt.xlabel("TTD (frames)")
    plt.ylabel("Count")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[✓] Saved TTD histogram: {save_path}")


def visualize_frame_level_predictions(
        model,
        X,
        y,
        names,
        device,
        window_size=20,
        stride=1,
        threshold=0.5,
        save_dir="frame_vis"
):
    """
    对每个完整序列生成帧级别预测可视化图像（clip_out 映射到每个窗口末尾帧）并计算 TTD。

    Returns:
        dict: 每段视频的 ttd 值，{name: ttd}
    """

    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    ttd_results = {}

    for seq_x, seq_y, name in zip(X, y, names):
        T = len(seq_x)
        frame_scores = np.zeros(T)
        frame_counts = np.zeros(T)

        with torch.no_grad():
            for t in range(window_size, T + 1, stride):
                x_clip = seq_x[t - window_size:t]
                x_tensor = torch.tensor(x_clip, dtype=torch.float32).unsqueeze(0).to(device)

                clip_out, _ = model(x_tensor)  # (1, 1)
                clip_score = clip_out.item()

                frame_index = t - 1  # 只更新最后一帧
                frame_scores[frame_index] += clip_score
                frame_counts[frame_index] += 1

        avg_scores = np.divide(frame_scores, frame_counts + 1e-8)
        num_valid_frames = np.sum(frame_counts > 0)

        # === 获取帧级预测结果 ===
        binary_preds = (avg_scores > threshold).astype(int)

        # === 计算 TTD ===
        ttd = compute_ttd_single(seq_y, binary_preds)
        ttd_results[name] = ttd
        ttd_display = "未检测" if ttd == np.inf else f"{ttd} 帧"
        print(f"[{name}] 原始帧数: {T}, 有预测帧数: {num_valid_frames}, TTD: {ttd_display}")

        # === 可视化 ===
        plt.figure(figsize=(10, 4))
        plt.plot(seq_y, label="Ground Truth", color='green', alpha=0.6)
        plt.plot(avg_scores, label="Predicted Score (clip_out)", color='blue')
        plt.axhline(y=threshold, color='red', linestyle='--', label='Threshold')

        plt.title(f"{name} - Frame-level Prediction\nFrames: {T}, Predicted: {num_valid_frames}, TTD: {ttd_display}")
        plt.xlabel("Frame Index")
        plt.ylabel("Score / Label")
        plt.ylim(0, 1.05)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"{name}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"[✓] Saved frame visualization: {save_path}")

    return ttd_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate LSTM on sliding window clips (frame-level labels).")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to .pth model checkpoint')
    parser.add_argument('--save_preds', type=str, default='', help='Optional: Save per-clip predictions to CSV')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])

    # model config
    parser.add_argument('--input_size', type=int, default=34)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.6)

    # sliding window
    parser.add_argument('--window_size', type=int, default=20)
    parser.add_argument('--stride', type=int, default=1)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')
    print(f"[✓] Using device: {device}")

    # ====== 固定两个 CSV 文件路径 ======
    csv_paths = [
        # "F:\\IC\\fall_detection_code\\dataset\\UR\\processed\\extracted_labels_adls_URFD.csv",  # 你可替换成绝对路径
        "F:\\IC\\fall_detection_code\\dataset\\UR\\processed\\extracted_labels_falls_URFD.csv"
    ]

    # 合并两个 CSV 文件为一个 DataFrame
    dfs = [pd.read_csv(p) for p in csv_paths]
    df_all = pd.concat(dfs, ignore_index=True)
    temp_csv_path = Path("temp_combined_eval.csv")
    df_all.to_csv(temp_csv_path, index=False)

    # ====== 加载模型 ======
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model = build_model_from_ckpt(ckpt, args)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    # ====== 加载合并后的数据 ======
    X, y, names = load_dataset(temp_csv_path)

    print("✅ 数据加载成功！")
    print(f"合并样本数: {len(X)}")
    print(f"第1个样本 shape: {X[0].shape}")
    print(f"第1个标签长度: {len(y[0])}")
    print(f"第1个视频名: {names[0]}")

    # ====== 滑窗评估 ======
    results = evaluate_model_frame_level_sliding(
        model=model,
        X=X,
        y=y,
        names=names,
        device=device,
        window_size=args.window_size,
        stride=args.stride,
        batch_size=args.batch_size,
        save_path=args.save_preds
    )

    ttd_dict = visualize_frame_level_predictions(
        model=model,
        X=X,
        y=y,
        names=names,
        device=device,
        window_size=args.window_size,
        stride=args.stride,
        threshold=0.5,
        save_dir="frame_vis"
    )
    plot_ttd_distribution(ttd_dict, save_path="ttd_hist.png")

    # 如需统计整体 P50/P90：
    valid_ttds = [v for v in ttd_dict.values() if np.isfinite(v)]
    if len(valid_ttds) > 0:
        logger.info("\n[✓] TTD统计:")
        logger.info(f"  P50: {np.percentile(valid_ttds, 50):.2f} 帧")
        logger.info(f"  P90: {np.percentile(valid_ttds, 90):.2f} 帧")
        logger.info(f"  P95: {np.percentile(valid_ttds, 95):.2f} 帧")
        logger.info(f"  Min: {np.max(valid_ttds):.2f} 帧")
        logger.info(f"  Max: {np.min(valid_ttds):.2f} 帧")
    else:
        logger.info("[!] 没有可用于统计的 TTD 值（全部为无事件或未预测）")


if __name__ == '__main__':
    main()
