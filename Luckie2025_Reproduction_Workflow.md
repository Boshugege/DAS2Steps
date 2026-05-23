# Luckie 2025 Footstep Reproduction Workflow

这份说明对应 `luckie_footstep_pipeline.py`。目标是复现 Luckie et al. (2025) 的脚步检测与二维定位思路，但不直接照搬论文参数，而是让参数从当前 DAS 数据中自动调出来。

## 论文方法摘要

Luckie et al. 把行走看成一串离散脚步冲击事件。流程可以拆成四步：

1. 预处理 DAS record section，压制坏通道和背景噪声。
2. 用 STA/LTA 拾取候选到时，并把多通道 picks 聚成单个 footstep event。
3. 用事件 picks 估计浅层/地板中的有效脚步波速。
4. 对每个事件做二维 backprojection：候选源点到每个 DAS 通道的距离给出传播延迟，按延迟平移各通道 envelope 后叠加，叠加最大的位置就是脚步位置。

论文中的数值，例如 500 Hz、0.1 Hz 高通、STA/LTA 参数、约 311.5 m/s 的速度，是他们场地、采样率、介质和光纤耦合下的结果。本项目不把这些数值当默认答案。

## 本项目默认约束

- 采样率保持 `2000 Hz`，不做初始降采样。
- 只使用前 `100` 个通道。
- 通道 `0..14` 默认置零，因为这段噪声过大。
- 每个通道间距按 `1 m` 处理。
- 光纤二维形状来自仓库根目录的 `FiberPath`。
- 优先读取已有 `output/name_signals/{name}.csv`；若不存在，可以按 `DATA_DIR/Airtag` 的时间范围从 `DATA_DIR/DAS` 的 TDMS 中抽取。
- 脚本不会删除原有文件；遇到同名抽取结果会生成新的唯一文件名。

## 当前实现策略

### 1. 自动预处理调参

脚本扫描多个候选频带：

```text
(0.5, 8), (1, 12), (2, 18), (5, 10), (5, 20), (10, 35)
```

对每个频带和多个 MAD 阈值组合计算 RMS envelope mask，然后用以下指标给组合打分：

- mask 是否稀疏但不消失；
- 检测事件频率是否接近正常行走 cadence；
- 每个事件是否有足够多通道参与；
- envelope excess 的高分位是否明显。

这样可以让 `wangjiahui`、`wangdihai` 这类比较清楚的数据自动选到可用参数，同时其他名字也能跑出诊断报告。

### 2. 事件检测

当前主检测器不是 STA/LTA，而是：

1. 带通滤波；
2. 计算 RMS envelope；
3. 对每个通道做 log-envelope 的 MAD 阈值；
4. 根据同一时间帧的 active channels 合并成 event windows。

STA/LTA 在这批数据上比较容易被局部尖峰和长尾振动影响，所以暂时不作为主检测器。它可以以后作为辅助 pick refinement 加回来。

### 3. 波速估计

对每个事件：

1. 找 peak channel；
2. 假设 peak channel 附近是最近源点；
3. 对其他通道 envelope threshold crossing 做 arrival picks；
4. 在候选速度范围 `100..600 m/s` 中最小化 L1 travel-time residual；
5. 再用少量高分事件的 backprojection 聚焦度二次选择最终 `v0`。

如果某个数据没有足够可靠 picks，脚本会报告速度估计失败，并使用诊断 fallback 继续生成 backprojection 图，方便判断问题出在检测还是速度。

### 4. 二维 Backprojection

对每个事件，脚本在 `FiberPath` 外扩的二维网格上计算：

```text
B(x, y) = max_t0 mean_i envelope_i(t0 + distance((x,y), channel_i) / v0)
```

默认只用 peak channel 附近 `±25 m` 的通道，避免远处低 SNR 通道稀释叠加结果。热图最大点输出为 `x_hat, y_hat`，并用 `>= 95% max` 的连通区域估计不确定性半径。

当前脚本还加入了一个弱近场先验：候选点离事件 peak channel 过远时会被轻微压低。这不是论文里的固定参数，而是针对单根光纤 2D 离轴歧义的实用约束；如果以后有视频/平面图先验，可以替换成更明确的可行区域 mask。

## 使用方式

快速验证 `wangjiahui`，只跑前 60 秒、最多定位 20 个事件：

```powershell
.\.venv\Scripts\python.exe .\luckie_footstep_pipeline.py --names wangjiahui --max-duration 60 --max-events 20 --grid-dx 2
```

处理两个推荐名字：

```powershell
.\.venv\Scripts\python.exe .\luckie_footstep_pipeline.py --names wangjiahui wangdihai --grid-dx 1 --bp-radius 25
```

处理其他名字时建议先小跑：

```powershell
.\.venv\Scripts\python.exe .\luckie_footstep_pipeline.py --names aipumin --max-duration 45 --max-events 10 --grid-dx 2
```

常用可调参数：

```powershell
--zero-first-channels 15
--env-win-sec 0.35
--env-hop-sec 0.05
--min-active-channels 4
--speed-min 100
--speed-max 600
--bp-radius 25
--bp-source-prior-radius 10
--edge-guard-sec 5
```

输出目录格式：

```text
results/luckie_reproduction/{name}/{YYYYMMDD_HHMMSS}/
```

主要输出：

```text
config.json
channel_to_xy.csv
preprocessing_band_tuning.csv
psd_mean.png
detection_overview.png
event_catalog.csv
velocity_by_event.csv
velocity_picks.csv
velocity_summary.png
velocity_residual_hist.png
trajectory_backprojection.csv
trajectory_backprojection.png
event_000_heatmap.png
reproduction_report.md
```

## 需要人工检查的图

先看 `detection_overview.png`：

- 白色 mask 点是否落在明显脚步能量上；
- 橙色 event window 是否把相邻脚步过度合并；
- peak channel 是否沿合理轨迹移动。

再看 `velocity_summary.png`：

- 速度分布是否集中；
- residual 是否足够小；
- 如果速度贴着搜索边界，说明频带、pick 或源点假设需要重调。

最后看 `trajectory_backprojection.png`：

- 定位点是否沿着合理路径；
- 平滑轨迹是否出现不可能的跳变；
- 不确定性半径是否过大。
