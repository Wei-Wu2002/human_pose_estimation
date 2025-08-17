import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import os


def evaluate_model(model, test_loader, device, threshold=0.5, save_dir="eval_vis"):
    """
    对模型在测试集上进行 clip 级评估，并可视化每段的 frame-level 预测置信度。

    Args:
        model: 已训练好的模型
        test_loader: DataLoader
        device: torch.device
        threshold: clip-level 阈值
        save_dir: 可视化图像保存路径（每段一张图）

    Returns:
        dict: clip-level 评估指标 + 混淆矩阵
    """
    model.eval()
    all_preds = []
    all_labels = []

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with torch.no_grad():
        for idx, (x_batch, y_batch) in enumerate(test_loader):
            x_batch = x_batch.to(device).float()
            y_batch = y_batch.to(device).float().unsqueeze(1)

            clip_out, frame_outs = model(x_batch)  # clip_out: (B, 1), frame_outs: (B, T, 1)
            preds = (clip_out > threshold).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

            # === 帧级可视化 ===
            for i in range(x_batch.size(0)):
                frame_conf = frame_outs[i].squeeze().cpu().numpy()  # shape: (T,)
                label = int(y_batch[i].item())
                pred = int(preds[i].item())

                plt.figure(figsize=(8, 3))
                binary_pred = (frame_conf > threshold).astype(int)
                plt.step(np.arange(len(binary_pred)), binary_pred, where='mid', label='Frame Prediction', color='blue')
                plt.axhline(y=threshold, color='red', linestyle='--', label='Threshold')
                plt.title(f"Sample {idx * test_loader.batch_size + i} | GT: {label} | Pred: {pred}")
                plt.xlabel("Frame Index")
                plt.ylabel("Confidence")
                plt.ylim(0, 1.05)
                plt.grid(True)
                plt.legend()
                save_path = os.path.join(save_dir, f"sample_{idx * test_loader.batch_size + i}.png")
                plt.tight_layout()
                plt.savefig(save_path)
                plt.close()

    # ==== 片段级评估 ====
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()

    return {
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'precision': precision_score(all_labels, all_preds, pos_label=1),
        'recall': recall_score(all_labels, all_preds, pos_label=1),
        'f1_score': f1_score(all_labels, all_preds, pos_label=1),
        'accuracy': accuracy_score(all_labels, all_preds)
    }
