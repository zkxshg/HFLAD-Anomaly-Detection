import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

# 配置参数
window_size = 10   # 时间窗口大小
feature_dim = 5    # 输入特征维度
latent_dim = 64    # 潜在空间维度
time_scales = [2, 4, 8]  # 时间编码器的层次时间尺度

class TimeEncoder(tf.keras.layers.Layer):
    """
    时间编码器，使用因果卷积、膨胀卷积和时间卷积网络（TCN）来捕获多尺度的时间模式。
    """
    def __init__(self, time_scales, **kwargs):
        super(TimeEncoder, self).__init__(**kwargs)
        self.time_scales = time_scales
        self.conv_layers = [layers.Conv1D(filters=latent_dim, kernel_size=scale, padding='causal', dilation_rate=scale)
                            for scale in time_scales]

    def call(self, inputs):
        outputs = []
        for conv in self.conv_layers:
            outputs.append(conv(inputs))
        return tf.concat(outputs, axis=-1)  # 合并多尺度的时间编码

class FeatureEncoder(tf.keras.layers.Layer):
    """
    特征编码器，压缩特征维度以提取多元时间序列数据中的特征依赖关系。
    """
    def __init__(self, latent_dim, **kwargs):
        super(FeatureEncoder, self).__init__(**kwargs)
        self.gru = layers.GRU(latent_dim, return_sequences=True)
    
    def call(self, inputs):
        return self.gru(inputs)  # 返回编码后的特征信息

class HVAE(tf.keras.layers.Layer):
    """
    层次变分自编码器（HVAE）模块，用于重构输入数据并通过重构误差来检测异常。
    """
    def __init__(self, latent_dim, **kwargs):
        super(HVAE, self).__init__(**kwargs)
        self.encoder = layers.Dense(latent_dim * 2)  # 用于生成潜在变量的均值和标准差
        self.decoder = layers.Dense(feature_dim)  # 用于重构输入

    def sample(self, mean, logvar):
        """基于均值和log方差的采样方法"""
        eps = tf.random.normal(shape=mean.shape)
        return mean + tf.exp(logvar * 0.5) * eps

    def call(self, inputs):
        # 编码阶段
        x = layers.Flatten()(inputs)
        z_params = self.encoder(x)
        mean, logvar = tf.split(z_params, num_or_size_splits=2, axis=-1)
        z = self.sample(mean, logvar)

        # 解码阶段
        reconstructed = self.decoder(z)
        return reconstructed, mean, logvar

class HFLAD(Model):
    """
    HFLAD 模型，包括时间编码器、特征编码器和层次变分自编码器。
    """
    def __init__(self, time_scales, latent_dim, **kwargs):
        super(HFLAD, self).__init__(**kwargs)
        self.time_encoder = TimeEncoder(time_scales)
        self.feature_encoder = FeatureEncoder(latent_dim)
        self.hvae = HVAE(latent_dim)

    def call(self, inputs):
        # 时间编码阶段
        time_encoded = self.time_encoder(inputs)

        # 特征编码阶段
        feature_encoded = self.feature_encoder(time_encoded)

        # 重构与异常检测
        reconstructed, mean, logvar = self.hvae(feature_encoded)

        return reconstructed, mean, logvar

def compute_loss(inputs, reconstructed, mean, logvar):
    """
    损失函数，包括重构误差和KL散度。
    """
    # 计算重构误差（均方误差）
    reconstruction_loss = tf.reduce_mean(tf.square(inputs - reconstructed))
    
    # 计算KL散度
    kl_divergence = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar))
    
    # 总损失
    return reconstruction_loss + kl_divergence

def train_step(model, inputs, optimizer):
    """
    单步训练函数，进行前向传播、计算损失并进行反向传播。
    """
    with tf.GradientTape() as tape:
        reconstructed, mean, logvar = model(inputs)
        loss = compute_loss(inputs, reconstructed, mean, logvar)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def train(model, dataset, epochs, optimizer):
    """
    训练模型
    """
    for epoch in range(epochs):
        epoch_loss = 0
        for step, inputs in enumerate(dataset):
            loss = train_step(model, inputs, optimizer)
            epoch_loss += loss
        print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(dataset):.4f}")

# 加载数据并创建数据集
def load_data(filepath, batch_size=32):
    """
    加载多元时间序列数据并创建 TensorFlow 数据集
    """
    data = np.load(filepath)
    dataset = tf.data.Dataset.from_tensor_slices(data).batch(batch_size)
    return dataset

if __name__ == "__main__":
    # 配置参数
    data_path = "sample_data.npy"  # 输入数据路径
    batch_size = 32
    epochs = 10
    learning_rate = 0.001

    # 加载数据
    dataset = load_data(data_path, batch_size)

    # 创建模型和优化器
    model = HFLAD(time_scales, latent_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    # 训练模型
    train(model, dataset, epochs, optimizer)
