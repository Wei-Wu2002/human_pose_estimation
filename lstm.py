import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from configs.logger import logger
from dataset.dataloader import get_dataloaders
from src.LSTM import LSTM
from utils.model_evaluate import evaluate_model

def joint_loss(clip_out, frame_outs, target, alpha=0.7, threshold=0.5):
    """
    计算 clip-level 和 frame-level 联合 BCE 损失
    Args:
        clip_out: (B, 1) 片段级输出
        frame_outs: (B, T, 1) 帧级输出
        target: (B, 1) clip-level 标签（0或1）
        alpha: 权重，越大越重视 clip-level loss
        threshold: 用于 binary 分割（非必需，这里不应用）

    Returns:
        loss: 加权后的总损失
    """
    bce = nn.BCELoss()

    # Clip-level loss
    clip_loss = bce(clip_out, target)

    # Frame-level loss：broadcast target → (B, T, 1)
    target_frame = target.unsqueeze(1).repeat(1, frame_outs.shape[1], 1)
    frame_loss = bce(frame_outs, target_frame)

    # Total loss
    loss = alpha * clip_loss + (1 - alpha) * frame_loss
    return loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Handcrafted feature loading test")
    parser.add_argument('--csv_path', type=str, default='dataset/YouTube/processed/extracted_labels_Youtube.csv',
                        help='Path to CSV file containing npy paths and labels')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Learning rate for the optimizer')
    parser.add_argument('--num_epochs', type=int, default=500,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training and testing')
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    start = time.time()

    logger.info(f"Loading dataset from: {csv_path}")
    train_loader, test_loader = get_dataloaders(csv_path=csv_path, mode='train', batch_size=args.batch_size, shuffle=True,
                                                test_ratio=0.2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # # 从 train_loader 中取出一个 batch 做检查
    # train_batch = next(iter(train_loader))
    # test_batch = next(iter(test_loader))
    #
    # x_train, y_train = train_batch
    # x_test, y_test = test_batch
    #
    # logger.info(f"Train batch - x shape: {x_train.shape}, y shape: {y_train.shape}")
    # logger.info(f"Test batch  - x shape: {x_test.shape}, y shape: {y_test.shape}")
    #
    # # 打印类别分布（可选）
    # train_labels = [label.item() for _, label in train_loader.dataset]
    # test_labels = [label.item() for _, label in test_loader.dataset]
    #
    # logger.info(f"Train label classes: {sorted(set(train_labels))}, total samples: {len(train_labels)}")
    # logger.info(f"Test label classes: {sorted(set(test_labels))}, total samples: {len(test_labels)}")

    input_size = 34
    hidden_size = 128
    num_layers = 3
    output_size = 1
    dropout = 0.6
    use_sigmoid = True
    weight_decay = 1e-6
    alpha = 0.8

    model = LSTM(input_size=input_size,
                 hidden_size=hidden_size,
                 num_layers=num_layers,
                 output_size=output_size,
                 dropout=dropout,
                 use_sigmoid=use_sigmoid).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=weight_decay)

    # ==== 打印模型和超参数配置 ====
    logger.info("==== LSTM Model Configuration ====")
    logger.info(f"Input size   : {input_size}")
    logger.info(f"Hidden size  : {hidden_size}")
    logger.info(f"Num layers   : {num_layers}")
    logger.info(f"Dropout      : {dropout}")
    logger.info(f"Use Sigmoid  : {use_sigmoid}")
    logger.info(f"Output size  : {output_size}")

    logger.info("==== Training Hyperparameters ====")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Optimizer    : AdamW")
    logger.info(f"Weight decay : {weight_decay}")
    logger.info(f"Loss function: BCELoss")
    logger.info(f"Num epochs   : {args.num_epochs}")
    logger.info(f"Batch size   : {args.batch_size}")

    # ==== 训练循环 ====
    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0
        correct = total = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device).float()  # (B, T, 34)
            y_batch = y_batch.to(device).float().unsqueeze(1)  # (B, 1)

            optimizer.zero_grad()
            clip_out, frame_outs = model(x_batch)
            loss = joint_loss(clip_out, frame_outs, y_batch, alpha=alpha)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x_batch.size(0)
            preds = (clip_out > 0.5).float()  # 概率转标签
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

        train_acc = 100 * correct / total
        avg_loss = running_loss / total
        logger.info(f"[Epoch {epoch + 1}] Loss: {avg_loss:.4f} | Accuracy: {train_acc:.4f}%")

    # ==== 测试评估 ====
    results = evaluate_model(model, test_loader, device)

    logger.info("==== Test Evaluation Results ====")
    logger.info(f"Confusion Matrix:")
    logger.info(f"  TP: {results['tp']}, TN: {results['tn']}, FP: {results['fp']}, FN: {results['fn']}")
    logger.info(f"Precision : {results['precision']:.4f}")
    logger.info(f"Recall    : {results['recall']:.4f}")
    logger.info(f"F1 Score  : {results['f1_score']:.4f}")
    logger.info(f"Accuracy  : {results['accuracy'] * 100:.2f}%")
