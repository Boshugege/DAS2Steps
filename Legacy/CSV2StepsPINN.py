import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert
from scipy.ndimage import gaussian_filter1d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# 1. 配置参数
# =========================
CSV_PATH = "D:/DAS2Steps/tdms/tdms_output/daiweijia.csv"
FS = 2000.0
BANDPASS_LOWCUT = 5.0
BANDPASS_HIGHCUT = 80.0
DOWNSAMPLE_FACTOR = 5   # 保持较高的时间分辨率

# 训练超参数
LR = 1e-3
MAX_EPOCHS = 300
BATCH_SIZE = 8          # 显存允许的情况下尽量大
PATCH_CH = 64           # 增大视野，必须能看到 V 字形的“翅膀”
PATCH_TIME = 256        # 增大视野

# 物理参数初始化
INIT_VELOCITY = 15.0    # 初始猜测速度 (channels/sec 对应的像素数)
                        # 这里的单位是: pixels_in_channel_dim / pixels_in_time_dim
                        # 需根据下采样后的比例估算

# =========================
# 2. 数据预处理 (关键：瞬时包络)
# =========================
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def get_instantaneous_envelope(data):
    """
    计算瞬时包络：保留了波形的时间精度，但去除了相位震荡
    """
    analytic_signal = hilbert(data, axis=0)
    envelope = np.abs(analytic_signal)
    return envelope

print("[Step 1] Loading & Preprocessing...")
try:
    df = pd.read_csv(CSV_PATH)
    raw_data = df.values.astype(np.float32) # (T, C)
except:
    print("Error: CSV not found.")
    exit()

# 1. 带通滤波
b, a = butter_bandpass(BANDPASS_LOWCUT, BANDPASS_HIGHCUT, FS, order=4)
filtered_data = np.zeros_like(raw_data)
for c in range(raw_data.shape[1]):
    filtered_data[:, c] = filtfilt(b, a, raw_data[:, c])

# 2. 提取瞬时包络 (Instantaneous Envelope)
# 这比滑窗能量好，因为它没有时间延迟，分辨率极高
envelope_data = get_instantaneous_envelope(filtered_data)

# 3. 归一化 (自适应背景抑制)
# 减去背景噪音底
noise_floor = np.percentile(envelope_data, 30, axis=0, keepdims=True)
envelope_data = np.maximum(envelope_data - noise_floor, 0)
# 归一化
scale = np.percentile(envelope_data, 99.5)
envelope_data = envelope_data / (scale + 1e-9)
envelope_data = np.clip(envelope_data, 0, 1)

# 4. 下采样
envelope_data = envelope_data[::DOWNSAMPLE_FACTOR, :] # (T, C)
full_data_tensor = torch.tensor(envelope_data.T, dtype=torch.float32).unsqueeze(0) # (1, C, T)

print(f"Data Shape: {full_data_tensor.shape}")

# =========================
# 3. 物理层：能量扩散模型
# =========================
class EnergyPhysicsLayer(nn.Module):
    def __init__(self, kernel_size_c=31, kernel_size_t=61):
        super().__init__()
        self.pad_c = kernel_size_c // 2
        self.pad_t = kernel_size_t // 2
        
        # 可学习参数：慢度 (Slowness = 1/Velocity)
        # 使用 log 保证非负
        self.log_slowness = nn.Parameter(torch.log(torch.tensor(1.0))) 
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(5.0)))
        
        # 坐标网格
        c_range = torch.arange(-self.pad_c, self.pad_c + 1, dtype=torch.float32)
        t_range = torch.arange(-self.pad_t, self.pad_t + 1, dtype=torch.float32)
        self.grid_c, self.grid_t = torch.meshgrid(c_range, t_range, indexing='ij')
        self.grid_c = self.grid_c.to(device)
        self.grid_t = self.grid_t.to(device)

    def get_kernel(self):
        s = torch.exp(self.log_slowness)
        sigma = torch.exp(self.log_sigma)
        
        # 物理模型：能量沿 V 字形传播
        # t = |x| * s
        expected_t = torch.abs(self.grid_c) * s
        diff = self.grid_t - expected_t
        
        # 使用高斯核模拟能量分布 (而不是 Ricker)
        # 这避免了相位对齐的问题，只关注能量是否重合
        kernel = torch.exp(-0.5 * (diff / sigma)**2)
        
        # 归一化：保证卷积不改变总能量量级
        kernel = kernel / (kernel.sum() + 1e-9)
        return kernel.unsqueeze(0).unsqueeze(0)

    def forward(self, source_map):
        kernel = self.get_kernel()
        return F.conv2d(source_map, kernel, padding=(self.pad_c, self.pad_t))

# =========================
# 4. 网络架构：U-Net 检测器
# =========================
class FootstepDetector(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 简单的 U-Net 结构，利用上下文信息
        self.enc1 = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(16))
        self.enc2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32))
        
        self.bottleneck = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU())
        
        self.dec2 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU())
        self.dec1 = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1), nn.ReLU())
        
        # 输出层：预测震源概率
        self.final = nn.Sequential(nn.Conv2d(16, 1, 1), nn.Sigmoid()) # 输出 0-1 概率
        
        self.physics = EnergyPhysicsLayer()

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        b = self.bottleneck(e2)
        
        # Decoder
        d2 = self.dec2(b) + e2 # Skip connection
        d1 = self.dec1(d2) + e1
        
        # Source Prediction (S)
        source = self.final(d1)
        
        # Physics Reconstruction (S * G)
        recon = self.physics(source)
        
        return source, recon

# =========================
# 5. 损失函数 (关键改进)
# =========================
def correlation_loss(pred, target):
    """
    归一化互相关损失 (Normalized Cross Correlation Loss)
    比 MSE 更鲁棒，只关心形状匹配，不关心绝对数值大小
    """
    # 展平
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    
    # 去均值
    pred_mean = pred_flat - pred_flat.mean(dim=1, keepdim=True)
    target_mean = target_flat - target_flat.mean(dim=1, keepdim=True)
    
    # 计算相关系数
    numerator = (pred_mean * target_mean).sum(dim=1)
    denominator = torch.sqrt((pred_mean**2).sum(dim=1) * (target_mean**2).sum(dim=1) + 1e-9)
    
    correlation = numerator / denominator
    return 1.0 - correlation.mean() # 最大化相关性 = 最小化 (1 - corr)

# =========================
# 6. 训练
# =========================
model = FootstepDetector().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

full_data_gpu = full_data_tensor.to(device)
C_full, T_full = full_data_gpu.shape[1], full_data_gpu.shape[2]

print("Start Training...")

for epoch in range(MAX_EPOCHS):
    optimizer.zero_grad()
    
    # 随机裁剪 Patch
    batch_input_list = []
    for _ in range(BATCH_SIZE):
        c_s = np.random.randint(0, C_full - PATCH_CH)
        t_s = np.random.randint(0, T_full - PATCH_TIME)
        crop = full_data_gpu[:, c_s:c_s+PATCH_CH, t_s:t_s+PATCH_TIME]
        batch_input_list.append(crop)
    batch_input = torch.stack(batch_input_list)
    
    # Forward
    source_pred, recon_pred = model(batch_input)
    
    # Loss
    # 1. 形状匹配损失 (NCC)
    loss_shape = correlation_loss(recon_pred, batch_input)
    
    # 2. 稀疏损失 (L1) - 只有少数地方有脚步
    loss_sparse = torch.mean(source_pred)
    
    # 3. 峰值约束 - 鼓励预测值接近 0 或 1 (二值化)
    loss_binary = torch.mean(source_pred * (1 - source_pred))
    
    total_loss = loss_shape + 0.2 * loss_sparse + 0.1 * loss_binary
    
    total_loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        slowness = torch.exp(model.physics.log_slowness).item()
        print(f"Ep {epoch}: Loss={total_loss.item():.4f} (Shape={loss_shape.item():.4f}) | Slowness={slowness:.2f}")

# =========================
# 7. 结果展示
# =========================
model.eval()
with torch.no_grad():
    # 预测整个时间段 (分块处理以防显存溢出，这里简化直接预测)
    # 实际使用建议沿时间轴切片预测再拼接
    source_full, recon_full = model(full_data_gpu.unsqueeze(0))

input_np = full_data_gpu.cpu().numpy().squeeze()
source_np = source_full.cpu().numpy().squeeze()
recon_np = recon_full.cpu().numpy().squeeze()
kernel_np = model.physics.get_kernel().detach().cpu().numpy().squeeze()

plt.figure(figsize=(18, 10))

# 1. 输入 (瞬时包络)
plt.subplot(3, 1, 1)
plt.imshow(input_np, aspect='auto', cmap='jet', origin='lower', vmin=0, vmax=0.8)
plt.title("Input: Instantaneous Envelope (High Res Energy)")
plt.colorbar()

# 2. 预测的脚步位置 (Source Probability)
plt.subplot(3, 1, 2)
plt.imshow(source_np, aspect='auto', cmap='hot', origin='lower', vmin=0, vmax=1)
plt.title("Output: Detected Footsteps (Source Probability)")
plt.colorbar()

# 3. 物理重构验证
plt.subplot(3, 1, 3)
plt.imshow(recon_np, aspect='auto', cmap='jet', origin='lower', vmin=0, vmax=0.8)
plt.title("Physics Check: Reconstructed Energy Field")
plt.colorbar()

plt.tight_layout()
plt.show()

# 显示学习到的核
plt.figure()
plt.imshow(kernel_np, aspect='auto', cmap='hot', origin='lower')
plt.title(f"Learned Physics Kernel (V-Shape)\nSlowness={torch.exp(model.physics.log_slowness).item():.2f}")
plt.colorbar()
plt.show()