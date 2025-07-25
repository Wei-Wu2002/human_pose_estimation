import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from configs.logger import logger
from dataset.dataloader import get_dataloaders
from src.GRU import GRU
from utils.model_evaluate import evaluate_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Handcrafted feature loading test")
    parser.add_argument('--csv_path', type=str, default='dataset/YouTube/processed/extracted_labels_Youtube.csv',
                        help='Path to CSV file containing npy paths and labels')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Learning rate for the optimizer')
    parser.add_argument('--num_epochs', type=int, default=300,
                        help='Number of training epochs')
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    start = time.time()

    logger.info(f"Loading dataset from: {csv_path}")
    train_loader, test_loader = get_dataloaders(csv_path=csv_path, mode='train', batch_size=32, shuffle=True,
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

    model = GRU(input_size=34,
                hidden_size=128,
                num_layers=3,
                output_size=1,
                dropout=0.6,
                use_sigmoid=True).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)

    # ==== 训练循环 ====
    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0
        correct = total = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device).float()  # (B, T, 34)
            y_batch = y_batch.to(device).float().unsqueeze(1)  # (B, 1)

            optimizer.zero_grad()
            outputs = model(x_batch)  # (B, 1)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x_batch.size(0)
            preds = (outputs > 0.5).float()  # 概率转标签
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
