# -*- coding: utf-8 -*-
import argparse
import ast
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
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


def evaluate_event_level_from_model(
        model,
        X,
        y,
        names,
        device,
        window_size=20,
        stride=1,
        score_threshold=0.5,
        tiou_threshold=0.5,
        min_event_len=1,
        merge_gap=0,
        save_framewise_csv=None,  # 可选：保存每个视频的帧级分数与预测
):
    """
    基于滑窗推理的事件级评估（tIoU 一对一匹配）。
    关键点：只在“有预测的帧区间”内进行事件抽取和匹配，自动裁剪 GT 与预测的对齐范围。

    Args:
        model, X, y, names, device: 与你现有代码一致
        window_size, stride        : 滑窗参数
        score_threshold            : 帧级分数二值化阈值
        tiou_threshold             : 事件匹配的 tIoU 阈值（如 0.5）
        min_event_len              : 过滤过短事件（帧数），默认 1
        merge_gap                  : 合并近邻事件的最大间隙（帧），默认 0 不合并
        save_framewise_csv         : 可选路径，若提供将保存每个视频逐帧的 score/pred/gt（仅有效区间）

    Returns:
        metrics: dict（全局微平均 Precision/Recall/F1/TP/FP/FN/mean_matched_tIoU 等）
        details: list[dict]（逐视频的事件段、匹配结果等）
    """

    model.eval()

    # ===== 工具函数：二值序列 -> 事件段；tIoU；贪心匹配 =====
    def _segments_from_binary(arr, min_len=1, merge_gap=0):
        arr = np.asarray(arr).astype(int)
        T = len(arr)
        segs = []
        in_seg = False
        start = -1
        for i in range(T):
            if arr[i] == 1 and not in_seg:
                in_seg = True
                start = i
            if arr[i] == 0 and in_seg:
                segs.append((start, i - 1))
                in_seg = False
        if in_seg:
            segs.append((start, T - 1))
        # 过滤短段
        segs = [(s, e) for (s, e) in segs if (e - s + 1) >= min_len]
        # 合并近邻
        if merge_gap > 0 and len(segs) > 1:
            merged = [segs[0]]
            for (s, e) in segs[1:]:
                ps, pe = merged[-1]
                if s - pe - 1 <= merge_gap:
                    merged[-1] = (ps, max(pe, e))
                else:
                    merged.append((s, e))
            segs = merged
        return segs

    def _tiou(a, b):
        (s1, e1), (s2, e2) = a, b
        inter = max(0, min(e1, e2) - max(s1, s2) + 1)
        if inter == 0:
            return 0.0
        union = (e1 - s1 + 1) + (e2 - s2 + 1) - inter
        return inter / union

    def _greedy_match(gt_segs, pr_segs, thr):
        if len(gt_segs) == 0 or len(pr_segs) == 0:
            return [], set(range(len(gt_segs))), set(range(len(pr_segs)))
        tiou_mat = np.zeros((len(gt_segs), len(pr_segs)), dtype=float)
        for i, g in enumerate(gt_segs):
            for j, p in enumerate(pr_segs):
                tiou_mat[i, j] = _tiou(g, p)

        matches = []
        used_gt, used_pr = set(), set()
        candidates = [(tiou_mat[i, j], i, j) for i in range(len(gt_segs)) for j in range(len(pr_segs))]
        candidates.sort(reverse=True, key=lambda x: x[0])
        for t, i, j in candidates:
            if t < thr:
                break
            if i in used_gt or j in used_pr:
                continue
            matches.append((i, j, float(t)))
            used_gt.add(i)
            used_pr.add(j)
        unmatched_gt = set(range(len(gt_segs))) - used_gt
        unmatched_pr = set(range(len(pr_segs))) - used_pr
        return matches, unmatched_gt, unmatched_pr

    # ===== 汇总容器 =====
    all_gt_valid = []
    all_pred_valid = []
    all_names = []
    per_seq_scores = []  # 可选：保存有效区间的 scores
    per_seq_threshold = score_threshold

    # ===== 逐视频滑窗推理，聚合到帧级 =====
    for seq_x, seq_y, name in zip(X, y, names):
        T = len(seq_x)
        frame_scores = np.zeros(T, dtype=float)
        frame_counts = np.zeros(T, dtype=float)

        with torch.no_grad():
            for t in range(window_size, T + 1, stride):
                x_clip = seq_x[t - window_size:t]  # (W, D)
                x_tensor = torch.tensor(x_clip, dtype=torch.float32).unsqueeze(0).to(device)
                clip_out, _ = model(x_tensor)  # (1, 1)
                clip_score = float(clip_out.item())
                frame_idx = t - 1  # 仅累加窗口末帧
                frame_scores[frame_idx] += clip_score
                frame_counts[frame_idx] += 1.0

        # 仅保留被至少1个窗口覆盖的帧
        valid_mask = frame_counts > 0
        if not np.any(valid_mask):
            # 没有有效帧，跳过该视频
            continue

        # 有效帧的分数均值
        avg_scores = np.zeros_like(frame_scores)
        avg_scores[valid_mask] = frame_scores[valid_mask] / np.clip(frame_counts[valid_mask], 1e-8, None)

        # 有效区间的索引范围（理论上是 [window_size-1, T-1]，但加一层保险）
        valid_indices = np.where(valid_mask)[0]
        v_start, v_end = valid_indices[0], valid_indices[-1]

        # 裁剪到有效区间
        gt_valid = np.asarray(seq_y, dtype=int)[v_start: v_end + 1]
        scores_valid = avg_scores[v_start: v_end + 1]
        pred_valid = (scores_valid > score_threshold).astype(int)

        # 保存
        all_gt_valid.append(gt_valid)
        all_pred_valid.append(pred_valid)
        all_names.append(name)
        per_seq_scores.append(scores_valid)

    # ===== 事件级匹配（微平均） =====
    total_TP = total_FP = total_FN = 0
    matched_tious = []
    details = []

    for name, gt_arr, pr_arr in zip(all_names, all_gt_valid, all_pred_valid):
        gt_segs = _segments_from_binary(gt_arr, min_len=min_event_len, merge_gap=merge_gap)
        pr_segs = _segments_from_binary(pr_arr, min_len=min_event_len, merge_gap=merge_gap)
        matches, un_gt, un_pr = _greedy_match(gt_segs, pr_segs, tiou_threshold)

        TP, FP, FN = len(matches), len(un_pr), len(un_gt)
        total_TP += TP
        total_FP += FP
        total_FN += FN
        matched_tious.extend([m[2] for m in matches])

        details.append({
            "name": name,
            "gt_events": gt_segs,
            "pred_events": pr_segs,
            "TP": TP, "FP": FP, "FN": FN,
            "matches": matches,  # (gt_idx, pred_idx, tIoU)
        })

    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0.0
    recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_tiou = float(np.mean(matched_tious)) if len(matched_tious) > 0 else 0.0

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": total_TP,
        "fp": total_FP,
        "fn": total_FN,
        "support_events": total_TP + total_FN,
        "mean_matched_tIoU": mean_tiou,
        "tiou_threshold": tiou_threshold,
        "score_threshold": score_threshold,
        "min_event_len": min_event_len,
        "merge_gap": merge_gap,
        "window_size": window_size,
        "stride": stride,
    }

    # 可选保存逐视频帧级结果（仅有效区间）
    if save_framewise_csv:
        rows = []
        for name, gt_arr, score_arr in zip(all_names, all_gt_valid, per_seq_scores):
            pred_arr = (score_arr > score_threshold).astype(int)
            for i, (g, s, p) in enumerate(zip(gt_arr, score_arr, pred_arr)):
                rows.append({"video": name, "frame_idx_valid": i, "gt": int(g), "score": float(s), "pred": int(p)})
        pd.DataFrame(rows).to_csv(save_framewise_csv, index=False)
        print(f"[✓] Saved framewise valid results to: {save_framewise_csv}")

    return metrics, details


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

    # ====== 事件级（event-level）评估：tIoU 匹配 ======
    event_metrics, event_details = evaluate_event_level_from_model(
        model=model,
        X=X,
        y=y,
        names=names,
        device=device,
        window_size=args.window_size,
        stride=args.stride,
        score_threshold=0.5,  # 与你可视化时用的阈值一致
        tiou_threshold=0.5,  # 常用 0.5，可按需要改 0.3/0.7 等
        min_event_len=3,  # 过滤毛刺，按帧率设定（如 30fps 可设 3~5）
        merge_gap=2,  # 合并近邻段的小间隙，可按需要调
        save_framewise_csv="framewise_valid_scores.csv"  # 可选
    )

    logger.info("\n[✓] Event-level (tIoU) metrics:")
    logger.info(f"  Precision: {event_metrics['precision']:.4f}")
    logger.info(f"  Recall   : {event_metrics['recall']:.4f}")
    logger.info(f"  F1       : {event_metrics['f1']:.4f}")
    logger.info(f"  Mean tIoU (matched): {event_metrics['mean_matched_tIoU']:.4f}")
    logger.info(
        f"  TP={event_metrics['tp']}, FP={event_metrics['fp']}, FN={event_metrics['fn']}, Support={event_metrics['support_events']}")


if __name__ == '__main__':
    main()
