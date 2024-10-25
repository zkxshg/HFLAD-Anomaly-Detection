import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

def calculate_anomaly_scores(matrices, threshold):
    """
    根据签名矩阵计算异常得分。
    
    参数:
    -------
    matrices : list of np.ndarray
        签名矩阵列表，每个矩阵代表一个时间窗口。
    threshold : float
        异常检测的得分阈值。
        
    返回:
    -------
    scores : list of float
        每个时间窗口的异常得分列表。
    labels : list of int
        每个时间窗口的异常标签（0表示正常，1表示异常）。
    """
    scores = []
    labels = []
    for matrix in matrices:
        # 计算矩阵的平均值或其他统计信息，作为异常分数
        score = np.linalg.norm(matrix - np.mean(matrix))
        scores.append(score)
        
        # 根据阈值判断是否为异常
        label = 1 if score > threshold else 0
        labels.append(label)
        
    return scores, labels

def evaluate_performance(true_labels, pred_labels):
    """
    使用常用指标评估模型的性能。
    
    参数:
    -------
    true_labels : list of int
        真实标签列表（0表示正常，1表示异常）。
    pred_labels : list of int
        预测标签列表（0表示正常，1表示异常）。
        
    返回:
    -------
    metrics : dict
        评估指标字典，包括精确率、召回率、F1分数和AUC。
    """
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    auc = roc_auc_score(true_labels, pred_labels)
    
    metrics = {
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'AUC': auc
    }
    return metrics

def load_matrices(path, num_matrices):
    """
    加载签名矩阵文件。
    
    参数:
    -------
    path : str
        矩阵文件的存储目录。
    num_matrices : int
        要加载的矩阵数量。
        
    返回:
    -------
    matrices : list of np.ndarray
        签名矩阵列表。
    """
    matrices = []
    for i in range(num_matrices):
        matrix_path = f"{path}/matrix_{i}.npy"
        matrix = np.load(matrix_path)
        matrices.append(matrix)
        print(f"已加载签名矩阵: {matrix_path}")
    return matrices

if __name__ == "__main__":
    # 配置参数
    matrices_path = "./signature_matrices"  # 矩阵文件目录
    threshold = 1.0  # 异常阈值，根据需要调整
    num_matrices = 50  # 加载的矩阵数量
    
    # 加载签名矩阵
    matrices = load_matrices(matrices_path, num_matrices)
    
    # 计算异常分数和预测标签
    scores, pred_labels = calculate_anomaly_scores(matrices, threshold)
    
    # 真实标签（示例中假设前40个为正常，后10个为异常）
    true_labels = [0] * 40 + [1] * 10
    
    # 评估性能
    metrics = evaluate_performance(true_labels, pred_labels)
    
    # 打印评估结果
    print("评估结果:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
