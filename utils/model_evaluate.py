import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score


def evaluate_model(model, test_loader, device, threshold=0.5):
    """
    对给定模型在测试集上进行评估，返回混淆矩阵及各项指标（不打印）。

    参数：
        model (nn.Module): 要评估的模型
        test_loader (DataLoader): 测试数据加载器
        device (torch.device): 使用的设备
        threshold (float): 判断为正类（fall）的概率阈值

    返回：
        dict: 含混淆矩阵和指标的字典
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device).float()
            y_batch = y_batch.to(device).float().unsqueeze(1)

            outputs = model(x_batch)
            preds = (outputs > threshold).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

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
