import numpy as np
import os

def generate_signature_matrices(data, window_size, stride, save_path):
    """
    生成系统签名矩阵用于多元时间序列异常检测。
    
    参数:
    -------
    data : np.ndarray
        多元时间序列数据 (形状: [样本数, 特征数])
    window_size : int
        滑动窗口的大小，用于计算签名矩阵。
    stride : int
        滑动窗口的步长。
    save_path : str
        保存生成签名矩阵的目录。
    """
    # 获取数据的样本数量和特征数
    num_samples, num_features = data.shape
    print(f"开始生成签名矩阵，共 {num_samples} 个样本，每个样本包含 {num_features} 个特征。")
    
    # 检查保存路径是否存在，如果不存在则创建
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 通过滑动窗口生成签名矩阵
    matrix_count = 0
    for i in range(0, num_samples - window_size + 1, stride):
        # 提取窗口内的数据
        window_data = data[i:i + window_size]
        
        # 计算窗口内数据的协方差矩阵或相关矩阵
        # 可以根据需求使用 np.cov 或 np.corrcoef
        signature_matrix = np.corrcoef(window_data, rowvar=False)
        
        # 保存矩阵到指定路径
        matrix_filename = os.path.join(save_path, f"matrix_{i}.npy")
        np.save(matrix_filename, signature_matrix)
        print(f"保存签名矩阵: {matrix_filename}")
        
        matrix_count += 1

    print(f"全部签名矩阵生成完毕，共生成 {matrix_count} 个矩阵。")

def load_data(filepath):
    """
    从文件加载多元时间序列数据。

    参数:
    -------
    filepath : str
        数据文件的路径（假设为CSV文件，每列代表一个特征）。

    返回:
    -------
    np.ndarray
        加载后的多元时间序列数据。
    """
    try:
        data = np.loadtxt(filepath, delimiter=',')
        print(f"数据加载成功，形状为: {data.shape}")
        return data
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None

if __name__ == "__main__":
    # 加载数据
    filepath = "sample_data.csv"  # 数据文件路径
    data = load_data(filepath)
    
    # 参数设置
    if data is not None:
        window_size = 10  # 滑动窗口大小
        stride = 5        # 滑动步长
        save_dir = "./signature_matrices"  # 保存签名矩阵的目录

        # 生成签名矩阵
        generate_signature_matrices(data, window_size, stride, save_dir)
