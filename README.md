# DAS2Steps

## 202601更新：从tdms文件流式读取数据并输出

详见 `Usage.md` `TDMS2JSONStream` 。

从多通道分布式光纤读取 TDMS 数据到 CSV 中（每列 ch_*，每行一个时间采样），计算时域上“脚步存在的通道概率分布”（channels × time），并绘图标出每个时刻最可能的通道。

## TDMS2CSV.py

将 DAS 获得的逐分钟 TDMS 数据依赖 Airtag 切分到每个人对应的时间范围，并输出到 CSV 文件，便于通用读取。

CSV 输出示例：

```
ch_1, ch_2, ch_3

0, 0, 0

1, 1, 1

2, 2, 2
```

## CSV2Steps.py

### 依赖：

  ```bash
  pip install numpy pandas scipy matplotlib seaborn
  ```

### 使用方法（示例）：
  ```bash
  python CSV2Steps.py --input your_data.csv
  ```

### 主要输出（保存到 outdir）：
  - energy_matrix.npy  (channels x frames, 短时能量)
  - prob_matrix.npy    (channels x frames, 每列归一化为概率)
  - prob_matrix.csv
  - argmax_channel.csv (每帧最可能的 channel index)
  - figures: raw_heatmap.png, filtered_heatmap.png, prob_heatmap_with_argmax.png

### 输入参数信息

```
usage: CSV2Steps.py [-h] --input INPUT [--outdir OUTDIR] [--fs FS] [--lowcut LOWCUT] [--highcut HIGHCUT]
                    [--filter_order FILTER_ORDER] [--energy_win_ms ENERGY_WIN_MS] [--energy_step_ms ENERGY_STEP_MS]
                    [--savgol_window SAVGOL_WINDOW] [--median_smooth_frames MEDIAN_SMOOTH_FRAMES]
                    [--gauss_smooth_sigma GAUSS_SMOOTH_SIGMA] [--downsample_factor DOWNSAMPLE_FACTOR]
                    [--ignore_channels_start IGNORE_CHANNELS_START] [--hot_channel_k HOT_CHANNEL_K]
                    [--hot_mask_scale HOT_MASK_SCALE] [--hot_channel_post_threshold HOT_CHANNEL_POST_THRESHOLD]
                    [--hot_channel_post_scale HOT_CHANNEL_POST_SCALE] [--alpha_kalman ALPHA_KALMAN]

Compute time-channel footstep probability map from CSV.

options:
  -h, --help            show this help message and exit
  --input INPUT, -i INPUT
                        输入 CSV 文件路径（列名包含 ch_0 ... ch_N）
  --outdir OUTDIR, -o OUTDIR
                        结果输出目录（默认 ./output）
  --fs FS               采样率 Hz（可选，如未提供则用行索引作为时间单位）
  --lowcut LOWCUT       带通滤波低频截止（Hz），默认 10 Hz
  --highcut HIGHCUT     带通滤波高频截止（Hz），默认 60 Hz
  --filter_order FILTER_ORDER
                        Butterworth 滤波器阶数
  --energy_win_ms ENERGY_WIN_MS
                        短时能量窗口（ms），默认 50 ms
  --energy_step_ms ENERGY_STEP_MS
                        短时能量窗口滑动步长（ms），默认 10 ms
  --savgol_window SAVGOL_WINDOW
                        Savitzky-Golay 平滑窗口（奇数）用于信号平滑，默认 7
  --median_smooth_frames MEDIAN_SMOOTH_FRAMES
                        对 argmax 结果的中值滤波窗口（帧），默认 5
  --gauss_smooth_sigma GAUSS_SMOOTH_SIGMA
                        对概率矩阵时间轴做高斯平滑的 sigma（帧），默认 1.0
  --downsample_factor DOWNSAMPLE_FACTOR
                        如数据很长，可按时间轴下采样（整数因子），1 表示不下采样
  --ignore_channels_start IGNORE_CHANNELS_START
                        从通道 0 开始屏蔽到指定通道索引（不含该索引），用于去掉光源附近异常信号，默认 0 表示不屏蔽
  --hot_channel_k HOT_CHANNEL_K
                        检测热通道的阈值参数 k（median + k*MAD），默认 3.0
  --hot_mask_scale HOT_MASK_SCALE
                        热通道幅度抑制比例因子，默认 1.0（即抑制到 median 水平）
  --hot_channel_post_threshold HOT_CHANNEL_POST_THRESHOLD
                        二次热通道检测阈值：若某通道 argmax 占比 > threshold 则视为噪声，默认 0.1
  --hot_channel_post_scale HOT_CHANNEL_POST_SCALE
                        二次热通道屏蔽/抑制比例，0=完全屏蔽, 0<x<1=按比例缩小概率，默认 0.5
  --alpha_kalman ALPHA_KALMAN
                        一阶卡尔曼平滑参数 alpha，默认 0.01，越大越贴近原高斯平滑轨迹
```
