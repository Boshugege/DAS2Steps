import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# 可调超参数（集中管理）
# =========================
# 数据与预处理
CSV_PATH = "D:/DAS2Steps/tdms/tdms_output/daiweijia.csv"  # 数据路径
FS = 2000.0                           # 采样率 (Hz)
BANDPASS_LOWCUT = 10.0                # 带通低切 (Hz)
BANDPASS_HIGHCUT = 60.0               # 带通高切 (Hz)
BANDPASS_ORDER = 4                    # 滤波器阶数
GAUSS_SIGMA = 2.0                     # 时间高斯平滑 sigma
NOISE_FLOOR_PCT = 25                  # 背景百分位（减去）
NORM_PCT = 99                         # 归一化百分位（除以）
DOWNSAMPLE_FACTOR = 10                # 下采样因子

# 训练超参数
LR = 5e-3                             # 学习率
MAX_EPOCHS = 500                      # 训练轮数
CLIP_NORM = 1.0                       # 梯度裁剪

# 损失项权重
W_RECON = 1.0
W_PEAK = 0.8
W_SHARP = 0.5
W_SPARSE = 0.1
W_SMOOTH = 0.05
W_E = 10.0                             # 放宽E：允许无脚步时段

# 停留惩罚（抑制长平台）
DWELL_THRESH = 0.3                    # 概率阈值，视为高激活
W_DWELL = 0.02                        # 惩罚权重

# =========================
# 数据预处理工具函数
# =========================
def butter_bandpass(lowcut, highcut, fs, order=4):
    """设计 Butterworth 带通滤波器"""
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 1e-6)
    high = min(highcut / nyq, 0.999999)
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter_array(X, fs, lowcut=10.0, highcut=60.0, order=4):
    """
    对多通道信号（shape = [T, C]）按列进行带通滤波
    使用 filtfilt 做零相位滤波以减少噪声
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    Xf = np.zeros_like(X)
    for c in range(X.shape[1]):
        col = X[:, c]
        col = col - np.mean(col)
        colf = filtfilt(b, a, col, axis=0, method="pad")
        Xf[:, c] = colf
    return Xf



# =========================
# Load CSV
# =========================
parser = argparse.ArgumentParser(description="DAS footstep detection")
parser.add_argument("--mask-first-n", type=int, default=0, help="屏蔽前 N 个通道（默认 0，不屏蔽）")
args = parser.parse_args()

csv_path = CSV_PATH   # ← 改成你的路径（集中于超参数区）
df = pd.read_csv(csv_path)

data = df.values.astype(np.float64)  # (T, C)
T, C = data.shape

# ===== 数据预处理：降噪 =====
print("[Step 1] 数据标准化与去趋势...")
# 去均值并归一化（每通道）
channel_means = np.mean(data, axis=0)
channel_stds = np.std(data, axis=0, ddof=1) + 1e-9
data = (data - channel_means[np.newaxis, :]) / channel_stds[np.newaxis, :]

print(f"[Step 2] 带通滤波（{BANDPASS_LOWCUT}-{BANDPASS_HIGHCUT} Hz）...")
# 假设采样率为 FS，可根据实际调整
fs = FS
data = bandpass_filter_array(data, fs=fs, lowcut=BANDPASS_LOWCUT, highcut=BANDPASS_HIGHCUT, order=BANDPASS_ORDER)

print("[Step 3] 高斯平滑（时间轴）...")
# 沿时间轴进行高斯平滑以进一步降噪
data = gaussian_filter1d(data, sigma=GAUSS_SIGMA, axis=0)

# 数据检查：检查是否有 NaN/Inf 值
if np.any(np.isnan(data)):
    print("[Warning] 检测到 NaN 值，用 0 替换")
    data = np.nan_to_num(data, nan=0.0, posinf=1e3, neginf=-1e3)
if np.any(np.isinf(data)):
    print("[Warning] 检测到 Inf 值，进行裁剪")
    data = np.clip(data, -1e3, 1e3)

# 转置为 (C, T)，计算能量
data = data.T  # (C, T)

# 可调屏蔽前 N 个通道
if args.mask_first_n and args.mask_first_n > 0:
    n = min(int(args.mask_first_n), data.shape[0])
    data[:n, :] = 0.0
    print(f"[Preprocess] 屏蔽前 {n} 个通道")

# 计算能量（平方）
energy = np.abs(data)  # 使用绝对值而非平方，避免过大的值

# 对能量进行 log 变换，以压缩动态范围并更好地表示信号
energy = np.log10(energy + 1e-9)  # log scale

# 从能量中减去通道背景
noise_floor = np.percentile(energy, NOISE_FLOOR_PCT, axis=1, keepdims=True)
energy = energy - noise_floor
energy = np.maximum(energy, 0.0)

# 归一化到 [0, 1]
energy_max = np.percentile(energy, NORM_PCT)
if energy_max > 1e-6:
    energy = energy / (energy_max + 1e-9)
else:
    energy = np.zeros_like(energy)

energy = np.clip(energy, 0, 1)

# 下采样：减少时间维度，提速（可调）
downsample_factor = DOWNSAMPLE_FACTOR
energy_downsampled = energy[:, ::downsample_factor]

print(f"[Info] 原始数据形状: {(T, C)}")
print(f"[Info] 处理后数据形状: {energy_downsampled.shape}")
print(f"[Info] 能量值范围: [{energy_downsampled.min():.6f}, {energy_downsampled.max():.6f}]")
print(f"[Info] 能量值均值: {energy_downsampled.mean():.6f}")

# 最后检查
if np.any(np.isnan(energy_downsampled)):
    print("[Error] 处理后仍存在 NaN，重置为小值")
    energy_downsampled = np.nan_to_num(energy_downsampled, nan=1e-6, posinf=1.0, neginf=0.0)

# 转 torch
energy_t = torch.tensor(energy_downsampled, dtype=torch.float32).to(device)


class FootstepNet(nn.Module):
    def __init__(self, channels):
        super().__init__()

        # 增强的时空特征提取（更深的网络）
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(16, 1, kernel_size=1)
        
        # 批归一化稳定训练
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(16)

        # 存在性 E(t)
        self.exist_head = nn.Sequential(
            nn.Linear(channels, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: (1, 1, C, T)
        返回 logits（未经 sigmoid），用于 BCEWithLogitsLoss
        """
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))
        h = F.relu(self.bn4(self.conv4(h)))
        P_logits = self.conv5(h)  # (1,1,C,T) - 返回 logits，不经 sigmoid

        # E(t)
        energy_t = x.squeeze(0).squeeze(0)  # (C, T)
        E = self.exist_head(energy_t.T)     # (T,1)

        return P_logits.squeeze(), E.squeeze()

def loss_fn(P_logits, E, energy):
    """
    P_logits: (C, T) - 脚步 logits（未经 sigmoid）
    E: (T,)  - 全局置信度
    energy: (C, T) - 输入能量（已归一化到 0-1）
    
    改进策略：
    1. 峰值吸引：激活应该集中到能量最高的通道
    2. 尖锐化：鼓励形成尖锐的峰值而不是平缓的平台
    3. 中心聚集：相邻通道激活应该相似
    """
    P_logits = P_logits.float()
    E = E.float()
    energy = energy.float()
    
    # 检查数值稳定性
    assert not torch.isnan(P_logits).any(), "P_logits contains NaN"
    assert not torch.isnan(E).any(), "E contains NaN"
    assert not torch.isnan(energy).any(), "energy contains NaN"
    
    P = torch.sigmoid(P_logits)
    
    # ---- 主重构 loss ----
    criterion_bce = nn.BCEWithLogitsLoss()
    L_recon = criterion_bce(P_logits, energy)

    # ---- 峰值吸引约束：激活应该集中到能量最高的通道 ----
    # 对每个时刻，找到能量最高的通道，奖励其高激活
    max_energy_per_time, argmax_ch = torch.max(energy, dim=0)  # (T,)
    # 构造软目标：最高能量通道应该有高激活
    target_peaks = torch.zeros_like(P)
    target_peaks[argmax_ch, torch.arange(P.shape[1])] = 1.0
    
    # 只在有能量的地方计算峰值吸引
    energy_mask = max_energy_per_time > 0.05
    if energy_mask.any():
        L_peak_attraction = F.mse_loss(
            P[:, energy_mask] * max_energy_per_time[energy_mask].unsqueeze(0),
            target_peaks[:, energy_mask] * max_energy_per_time[energy_mask].unsqueeze(0)
        )
    else:
        L_peak_attraction = torch.tensor(0.0, device=P.device)
    
    # ---- 尖锐化约束：激活应该尖锐而不是平缓 ----
    # 计算每个时刻的熵，低熵=尖锐
    P_norm = P / (P.sum(dim=0, keepdim=True) + 1e-7)  # 归一化为概率
    entropy = -torch.sum(P_norm * torch.log(P_norm + 1e-7), dim=0)  # 熵
    L_sharpness = entropy.mean()  # 最小化熵 = 尖锐化
    
    # ---- 弱稀疏约束 ----
    L_sparse = torch.abs(P).mean() * 0.05
    
    # ---- 时间连续性 ----
    if P.shape[1] > 1:
        P_diff_t = torch.abs(P[:, 1:] - P[:, :-1]).mean()
    else:
        P_diff_t = torch.tensor(0.0, device=P.device)

    # ---- E 约束（放宽，允许无脚步时段）：不施加约束 ----
    L_E = torch.tensor(0.0, device=E.device)

    # 停留惩罚：抑制通道上相邻帧同时高激活（长平台）
    if P.shape[1] > 1:
        H = torch.relu(P - DWELL_THRESH)
        L_dwell = (H[:, 1:] * H[:, :-1]).mean()
    else:
        L_dwell = torch.tensor(0.0, device=P.device)

    total_loss = (
        W_RECON * L_recon             # 主损失：预测能量
        + W_PEAK * L_peak_attraction  # 峰值吸引：激活应集中到最高能量通道
        + W_SHARP * L_sharpness       # 尖锐化：形成尖锐峰值
        + W_SPARSE * L_sparse         # 弱稀疏约束
        + W_SMOOTH * P_diff_t         # 时间连续性
        + W_DWELL * L_dwell           # 停留惩罚：抑制长时间维持高激活
        + W_E * L_E                   # E项（默认0）
    )
    
    return total_loss

C, T = energy_t.shape

model = FootstepNet(C).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# 混合精度训练（提速）
scaler = torch.cuda.amp.GradScaler()

x = energy_t.unsqueeze(0).unsqueeze(0).to(device)  # (1,1,C,T)

# 训练轮次
max_epochs = MAX_EPOCHS
best_loss = float('inf')

# 存储最终的 P_logits 和 E
P_logits_final = None
E_final = None

for epoch in range(max_epochs):
    optimizer.zero_grad()

    with torch.cuda.amp.autocast():
        P_logits, E = model(x)
        loss = loss_fn(P_logits, E, energy_t)
    
    # 保存最终结果
    P_logits_final = P_logits
    E_final = E
    
    # 检查 NaN
    if torch.isnan(loss):
        print(f"[Error] Epoch {epoch}: Loss is NaN! Stopping training.")
        print(f"  P_logits range: [{P_logits.min():.6f}, {P_logits.max():.6f}]")
        print(f"  E range: [{E.min():.6f}, {E.max():.6f}]")
        break
    
    # 梯度剪辑防止爆炸
    scaler.scale(loss).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP_NORM)
    scaler.step(optimizer)
    scaler.update()

    if epoch % 10 == 0:
        loss_val = loss.item()
        print(f"Epoch {epoch:3d}, loss = {loss_val:.6f}, P_range=[{P_logits.min():.4f}, {P_logits.max():.4f}], E_range=[{E.min():.4f}, {E.max():.4f}]")
        if loss_val < best_loss:
            best_loss = loss_val
        else:
            if epoch > 50 and loss_val > best_loss * 1.5:
                print("[Info] Loss diverging, stopping early")
                break

P_logits_np = P_logits_final.detach().cpu().numpy()
P_np = torch.sigmoid(P_logits_final).detach().cpu().numpy()  # 转换为概率
E_np = E_final.detach().cpu().numpy()

# 恢复能量用于可视化（下采样前的版本）
energy_viz = energy[:, ::downsample_factor]

plt.figure(figsize=(14, 8))

# 原始能量
plt.subplot(2, 1, 1)
plt.imshow(
    energy_viz,
    aspect="auto",
    origin="lower",
    cmap="gray",
    vmin=0, vmax=1,
    interpolation='bilinear'
)
plt.colorbar(label="Normalized Energy")
plt.ylabel("Channel")
plt.title("Input Energy (Preprocessed)")

# 脚步概率
plt.subplot(2, 1, 2)
plt.imshow(
    P_np,
    aspect="auto",
    origin="lower",
    cmap="hot",
    vmin=0, vmax=1,
    interpolation='bilinear'
)
plt.colorbar(label="Footstep Probability")
plt.xlabel("Time Frame")
plt.ylabel("Channel")
plt.title("Predicted Footstep Probability")

plt.tight_layout()
plt.show()

# 打印诊断信息
print("\n=== 诊断信息 ===")
print(f"P_np 范围: [{P_np.min():.6f}, {P_np.max():.6f}]")
print(f"E_np 范围: [{E_np.min():.6f}, {E_np.max():.6f}]")
print(f"能量范围: [{energy_viz.min():.6f}, {energy_viz.max():.6f}]")
print(f"最强脚步通道（按平均概率）: {np.argmax(P_np.mean(axis=1))}")
print(f"最强能量通道（按平均能量）: {np.argmax(energy_viz.mean(axis=1))}")
