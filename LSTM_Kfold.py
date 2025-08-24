import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from configs.logger import logger
from dataset.dataloader import load_dataset, KeypointDataset, LightPoseAugmentation
from src.LSTM import LSTM
from utils.model_evaluate import evaluate_model


# def joint_loss(clip_out, frame_outs, target, alpha=0.7, threshold=0.5):
#     """
#     计算 clip-level 和 frame-level 联合 BCE 损失
#     Args:
#         clip_out: (B, 1) 片段级输出
#         frame_outs: (B, T, 1) 帧级输出
#         target: (B, 1) clip-level 标签（0或1）
#         alpha: 权重，越大越重视 clip-level loss
#         threshold: 用于 binary 分割（非必需，这里不应用）
#
#     Returns:
#         loss: 加权后的总损失
#     """
#     bce = nn.BCELoss()
#
#     # Clip-level loss
#     clip_loss = bce(clip_out, target)
#
#     # Frame-level loss：broadcast target → (B, T, 1)
#     target_frame = target.unsqueeze(1).repeat(1, frame_outs.shape[1], 1)
#     frame_loss = bce(frame_outs, target_frame)
#
#     # Total loss
#     loss = alpha * clip_loss + (1 - alpha) * frame_loss
#     return loss

def joint_loss(clip_out, frame_outs, target, alpha=0.7, w_neg=5.0, w_pos=1.0):
    bce = nn.BCELoss(reduction='none')
    # clip
    clip_loss_raw = bce(clip_out, target)  # (B,1)
    clip_w = torch.where(target > 0.5, torch.full_like(target, w_pos), torch.full_like(target, w_neg))
    clip_loss = (clip_loss_raw * clip_w).mean()

    # frame: broadcast target -> (B,T,1)
    target_frame = target.unsqueeze(1).repeat(1, frame_outs.shape[1], 1)
    frame_loss_raw = bce(frame_outs, target_frame)  # (B,T,1)
    frame_w = torch.where(target_frame > 0.5,
                          torch.full_like(target_frame, w_pos),
                          torch.full_like(target_frame, w_neg))
    frame_loss = (frame_loss_raw * frame_w).mean()

    return alpha * clip_loss + (1 - alpha) * frame_loss



def get_dataloaders_for_kfold(X, y, train_idx, test_idx, batch_size=32, shuffle=True):
    """根据索引创建训练和测试数据加载器"""
    # 获取训练和测试数据
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # 数据增强
    pose_aug = LightPoseAugmentation()

    # 创建数据集
    train_dataset = KeypointDataset(X_train, y_train, transform=pose_aug)
    test_dataset = KeypointDataset(X_test, y_test, transform=None)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Handcrafted feature loading test")
    parser.add_argument('--csv_path', type=str, default='dataset/YouTube/processed/extracted_labels_Youtube.csv',
                        help='Path to CSV file containing npy paths and labels')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Learning rate for the optimizer')
    parser.add_argument('--num_epochs', type=int, default=300,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training and testing')
    parser.add_argument('--k_folds', type=int, default=5,
                        help='Number of folds for k-fold cross validation')
    parser.add_argument('--results_file', type=str, default='kfold_results.json',
                        help='File to save k-fold results')
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    start = time.time()

    logger.info(f"Loading dataset from: {csv_path}")

    # 加载完整数据集
    X, y, _ = load_dataset(csv_path, mode='handcrafted')
    logger.info(f"Dataset loaded: X shape {X.shape}, y shape {y.shape}")

    kfold = KFold(n_splits=args.k_folds, shuffle=True, random_state=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    input_size = 34
    hidden_size = 128
    num_layers = 3
    output_size = 1
    dropout = 0.6
    use_sigmoid = True
    weight_decay = 1e-6
    alpha = 0.8

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
    logger.info(f"K-Folds      : {args.k_folds}")

    # 存储每一折的结果
    fold_results = []
    best_fold_accuracy = 0.0
    best_fold_model = None
    best_fold_index = -1

    # K折交叉验证
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X)):
        logger.info(f"==== Starting Fold {fold + 1}/{args.k_folds} ====")
        logger.info(f"Train samples: {len(train_idx)}, Test samples: {len(test_idx)}")

        # 获取当前折的数据加载器
        train_loader, test_loader = get_dataloaders_for_kfold(
            X, y, train_idx, test_idx, batch_size=args.batch_size, shuffle=True
        )

        # 初始化模型和优化器
        model = LSTM(input_size=input_size,
                     hidden_size=hidden_size,
                     num_layers=num_layers,
                     output_size=output_size,
                     dropout=dropout,
                     use_sigmoid=use_sigmoid).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=weight_decay)
        # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)

        # ==== 训练循环 ====
        best_epoch_loss = float('inf')
        best_epoch_acc = 0.0

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

            # 记录最佳epoch
            if avg_loss < best_epoch_loss:
                best_epoch_loss = avg_loss
                best_epoch_acc = train_acc

            if (epoch + 1) % 50 == 0:  # 每50个epoch打印一次
                logger.info(f"[Fold {fold + 1} Epoch {epoch + 1}] Loss: {avg_loss:.4f} | Accuracy: {train_acc:.4f}%")

        # ==== 测试评估 ====
        results = evaluate_model(model, test_loader, device)

        fold_result = {
            'fold': fold + 1,
            'train_loss': best_epoch_loss,
            'train_accuracy': best_epoch_acc,
            'test_tp': results['tp'],
            'test_tn': results['tn'],
            'test_fp': results['fp'],
            'test_fn': results['fn'],
            'test_precision': results['precision'],
            'test_recall': results['recall'],
            'test_f1_score': results['f1_score'],
            'test_accuracy': results['accuracy']
        }

        fold_results.append(fold_result)

        logger.info(f"==== Fold {fold + 1} Test Evaluation Results ====")
        logger.info(f"Confusion Matrix:")
        logger.info(f"  TP: {results['tp']}, TN: {results['tn']}, FP: {results['fp']}, FN: {results['fn']}")
        logger.info(f"Precision : {results['precision']:.4f}")
        logger.info(f"Recall    : {results['recall']:.4f}")
        logger.info(f"F1 Score  : {results['f1_score']:.4f}")
        logger.info(f"Accuracy  : {results['accuracy'] * 100:.2f}%")

        # 保存最佳模型
        if results['accuracy'] > best_fold_accuracy:
            best_fold_accuracy = results['accuracy']
            best_fold_model = model.state_dict().copy()
            best_fold_index = fold + 1
            logger.info(f"New best model found at fold {fold + 1} with accuracy: {best_fold_accuracy * 100:.2f}%")
    # ==== 计算平均结果 ====
    avg_results = {
        'avg_test_accuracy': float(np.mean([r['test_accuracy'] for r in fold_results])),
        'avg_test_precision': float(np.mean([r['test_precision'] for r in fold_results])),
        'avg_test_recall': float(np.mean([r['test_recall'] for r in fold_results])),
        'avg_test_f1_score': float(np.mean([r['test_f1_score'] for r in fold_results])),
        'std_test_accuracy': float(np.std([r['test_accuracy'] for r in fold_results])),
        'std_test_precision': float(np.std([r['test_precision'] for r in fold_results])),
        'std_test_recall': float(np.std([r['test_recall'] for r in fold_results])),
        'std_test_f1_score': float(np.std([r['test_f1_score'] for r in fold_results])),
    }

    # ==== 保存所有结果到CSV文件 ====
    # 创建DataFrame
    fold_df = pd.DataFrame(fold_results)

    # 添加平均结果行
    avg_row = {
        'fold': 'Average',
        'train_loss': np.nan,
        'train_accuracy': np.nan,
        'test_tp': np.nan,
        'test_tn': np.nan,
        'test_fp': np.nan,
        'test_fn': np.nan,
        'test_precision': avg_results['avg_test_precision'],
        'test_recall': avg_results['avg_test_recall'],
        'test_f1_score': avg_results['avg_test_f1_score'],
        'test_accuracy': avg_results['avg_test_accuracy']
    }

    std_row = {
        'fold': 'Std',
        'train_loss': np.nan,
        'train_accuracy': np.nan,
        'test_tp': np.nan,
        'test_tn': np.nan,
        'test_fp': np.nan,
        'test_fn': np.nan,
        'test_precision': avg_results['std_test_precision'],
        'test_recall': avg_results['std_test_recall'],
        'test_f1_score': avg_results['std_test_f1_score'],
        'test_accuracy': avg_results['std_test_accuracy']
    }

    fold_df = pd.concat([fold_df, pd.DataFrame([avg_row, std_row])], ignore_index=True)

    # 保存到CSV
    fold_df.to_csv(args.results_file, index=False, float_format='%.4f')

    # 保存详细结果到另一个CSV文件（包含超参数信息）
    summary_data = {
        'experiment_time': [time.strftime("%Y-%m-%d %H:%M:%S")],
        'best_fold': [best_fold_index],
        'best_accuracy': [best_fold_accuracy],
        'learning_rate': [args.lr],
        'num_epochs': [args.num_epochs],
        'batch_size': [args.batch_size],
        'k_folds': [args.k_folds],
        'input_size': [input_size],
        'hidden_size': [hidden_size],
        'num_layers': [num_layers],
        'dropout': [dropout],
        'weight_decay': [weight_decay],
        'alpha': [alpha]
    }

    summary_df = pd.DataFrame(summary_data)
    summary_file = args.results_file.replace('.csv', '_summary.csv')
    summary_df.to_csv(summary_file, index=False, float_format='%.6f')

    logger.info(f"Fold results saved to {args.results_file}")
    logger.info(f"Experiment summary saved to {summary_file}")

    # ==== 打印平均结果 ====
    logger.info("==== Average Test Results ====")
    logger.info(
        f"Accuracy  : {avg_results['avg_test_accuracy'] * 100:.2f}% ± {avg_results['std_test_accuracy'] * 100:.2f}%")
    logger.info(f"Precision : {avg_results['avg_test_precision']:.4f} ± {avg_results['std_test_precision']:.4f}")
    logger.info(f"Recall    : {avg_results['avg_test_recall']:.4f} ± {avg_results['std_test_recall']:.4f}")
    logger.info(f"F1 Score  : {avg_results['avg_test_f1_score']:.4f} ± {avg_results['std_test_f1_score']:.4f}")
    logger.info(f"Best fold : {best_fold_index} with accuracy: {best_fold_accuracy * 100:.2f}%")

    # ==== 保存最佳模型 ====
    save_dir = Path(r"F:\IC\fall_detection_code\work_space\temp\lstm")
    save_dir.mkdir(parents=True, exist_ok=True)

    model_path = save_dir / f"lstm_best_fold{best_fold_index}_{best_fold_accuracy * 100:.2f}.pth"
    torch.save({
        "fold": best_fold_index,
        "model_state_dict": best_fold_model,
        "accuracy": float(best_fold_accuracy),
        "hyperparameters": {
            'learning_rate': args.lr,
            'num_epochs': args.num_epochs,
            'batch_size': args.batch_size,
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': dropout,
            'weight_decay': weight_decay,
            'alpha': alpha
        }
    }, model_path)

    logger.info(f"Best model (fold {best_fold_index}) saved to {model_path}")
