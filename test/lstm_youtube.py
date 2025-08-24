# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import time
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from configs.logger import logger
from dataset.dataloader import load_dataset, KeypointDataset
from src.LSTM import LSTM
from utils.model_evaluate import evaluate_model


def build_model_from_ckpt(ckpt_dict: dict, cli_args) -> LSTM:
    """
    根据 checkpoint 中保存的超参数构建 LSTM 模型；
    若 ckpt 缺少某些超参，则使用命令行参数兜底。
    """
    hp = ckpt_dict.get("hyperparameters", {}) if isinstance(ckpt_dict, dict) else {}

    input_size = hp.get("input_size", cli_args.input_size)
    hidden_size = hp.get("hidden_size", cli_args.hidden_size)
    num_layers = hp.get("num_layers", cli_args.num_layers)
    dropout = hp.get("dropout", cli_args.dropout)
    use_sigmoid = True  # 训练时就是Sigmoid + BCELoss
    output_size = 1

    logger.info("==== Reconstructed LSTM from checkpoint (with fallbacks) ====")
    logger.info(f"input_size = {input_size}, hidden_size = {hidden_size}, "
                f"num_layers = {num_layers}, dropout = {dropout}, output_size = {output_size}")

    model = LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size,
        dropout=dropout,
        use_sigmoid=use_sigmoid
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved LSTM model on a dataset CSV.")
    parser.add_argument('--csv_path', type=str, required=True,
                        help='CSV with columns [npy_path, video, label] etc. (same as training).')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to saved checkpoint .pth from training.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for evaluation (no grad).')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'],
                        help='Evaluation device. "auto" picks cuda if available.')
    parser.add_argument('--mode', type=str, default='handcrafted',
                        help='Pass-through to load_dataset(..., mode=...). Use the same as training.')
    parser.add_argument('--save_preds', type=str, default='',
                        help='Optional path to save per-sample predictions CSV.')
    # 兜底超参（当 ckpt 里没有时用）
    parser.add_argument('--input_size', type=int, default=34)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.6)

    args = parser.parse_args()

    # 设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # 读取 checkpoint
    ckpt_path = Path(args.checkpoint)
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"
    logger.info(f"Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')  # 放到CPU，稍后再to(device)

    # 构建模型并加载权重
    model = build_model_from_ckpt(ckpt, args)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(f"Missing keys when loading state_dict: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys when loading state_dict: {unexpected}")
    model.to(device)
    model.eval()

    # 载入数据（不做增强）
    csv_path = Path(args.csv_path)
    logger.info(f"Loading dataset from: {csv_path}")
    X, y, names = load_dataset(csv_path, mode=args.mode)
    logger.info(f"Dataset loaded: X shape {X.shape}, y shape {y.shape}")

    test_dataset = KeypointDataset(X, y, transform=None)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 评估
    t0 = time.time()
    with torch.no_grad():
        results = evaluate_model(model, test_loader, device)
    dt = time.time() - t0

    # 打印结果
    logger.info("==== Evaluation Results ====")
    logger.info(f"Time: {dt:.2f}s for {len(test_dataset)} samples")
    if isinstance(ckpt, dict):
        logger.info(f"Checkpoint meta: fold={ckpt.get('fold')}, "
                    f"train_best_acc={ckpt.get('accuracy')}")
        logger.info(f"Hyperparameters (ckpt): {json.dumps(ckpt.get('hyperparameters', {}), ensure_ascii=False)}")
    logger.info(f"Confusion Matrix: TP={results['tp']}, TN={results['tn']}, FP={results['fp']}, FN={results['fn']}")
    logger.info(f"Accuracy : {results['accuracy'] * 100:.2f}%")
    logger.info(f"Precision: {results['precision']:.4f}")
    logger.info(f"Recall   : {results['recall']:.4f}")
    logger.info(f"F1 score : {results['f1_score']:.4f}")

    # 可选：保存逐样本预测（若evaluate_model已有返回逐样本，可直接用；若没有，这里演示如何再跑一遍前向计算拿到clip概率）
    if args.save_preds:
        logger.info(f"Saving per-sample predictions to: {args.save_preds}")
        all_probs, all_preds, all_labels = [], [], []
        model.eval()
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device).float()
                yb = yb.cpu().numpy().tolist()
                clip_out, _ = model(xb)  # (B,1)
                probs = clip_out.squeeze(1).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                all_probs.extend(probs.tolist())
                all_preds.extend(preds.tolist())
                all_labels.extend(yb)

        # 若 names 是样本名列表（与 X 对齐），写入方便对照；否则写入索引
        idx_col = list(range(len(all_probs)))
        if names is not None and len(names) == len(all_probs):
            id_or_name = names
        else:
            id_or_name = idx_col

        out_df = pd.DataFrame({
            "id_or_name": id_or_name,
            "label": all_labels,
            "prob": all_probs,
            "pred": all_preds
        })
        out_df.to_csv(args.save_preds, index=False, float_format="%.6f")
        logger.info("Per-sample predictions saved.")

if __name__ == '__main__':
    main()
