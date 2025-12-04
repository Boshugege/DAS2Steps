import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ============================
# User inputs
# ============================
# data: (nt, nx) DAS array: time × channels
# data = ...

# csv_path = r"D:\DAS2Steps\tdms\tdms_output\wangdihai.csv"
# df = pd.read_csv(csv_path)
# ch_cols = [c for c in df.columns if c.lower().startswith("ch_")]
# if len(ch_cols) == 0:
#     raise RuntimeError("CSV 未找到 ch_ 列")
# data = df[ch_cols].values.astype(np.float64)  # shape (nt, nx)

npy_path = r"D:\DAS2Steps\output\data.npy"   # <- 把此处改为你的 .npy 文件路径
data = np.load(npy_path)                 # data shape 应为 (nt, nx)

if data.ndim != 2:
    raise ValueError("加载的 .npy 必须是 2D 数组 (nt, nx)，但得到 shape=" + str(data.shape))

fs = 2000.0             # sampling frequency
dt = 1.0 / fs          # time step
dx = 1               # channel spacing (m)

fmin_crop = -0.5      # crop low freq for plotting
fmax_crop =  0.5      # crop high freq for plotting
db_min = -50
db_max = 0

v_lines = [1]
v_colors = ["w", "r", "c"]


# ============================
# f–k computation with symmetric frequency & wavenumber
# ============================
def fk_spectrum_sym_freq(data, dt, dx):
    """
    Compute symmetric frequency (−fmax … +fmax)
    and symmetric wavenumber (−kmax … +kmax)
    f–k spectrum using 2D FFT and fftshift.
    """
    nt, nx = data.shape

    # -------- Temporal FFT (with shift for symmetric frequency) --------
    nfft_t = 1 << int(np.ceil(np.log2(nt)))
    freq_full = np.fft.fftfreq(nfft_t, d=dt)   # includes negative freqs
    freq_full = np.fft.fftshift(freq_full)     # reorder to [-fmax..fmax]

    # FFT along time → shape (nt_fft, nx)
    F_t = np.fft.fft(data, n=nfft_t, axis=0)
    F_t = np.fft.fftshift(F_t, axes=0)         # shift frequency dimension

    # -------- Spatial FFT (shifted for symmetric k) --------
    nfft_x = 1 << int(np.ceil(np.log2(nx)))
    k_full = np.fft.fftfreq(nfft_x, d=dx) * 2 * np.pi
    k_full = np.fft.fftshift(k_full)           # reorder to [-kmax..kmax]

    F_k = np.fft.fft(F_t, n=nfft_x, axis=1)
    F_k = np.fft.fftshift(F_k, axes=1)

    fk_amp = np.abs(F_k)   # amplitude

    return freq_full, k_full, fk_amp


# ============================
# Main f–k processing
# ============================
# Detrend data (recommended)
data_detr = data - np.mean(data, axis=0, keepdims=True)

# Compute full symmetric f–k spectrum
freqs, k_sym, fk_amp = fk_spectrum_sym_freq(data_detr, dt, dx)

# Convert to dB scale
fk_amp_norm = fk_amp / (np.max(fk_amp) + 1e-12)
fk_dB = 20 * np.log10(fk_amp_norm + 1e-12)

# Frequency crop (−15 to +15 Hz)
fmask = (freqs >= fmin_crop) & (freqs <= fmax_crop)
freqs_crop = freqs[fmask]
fk_crop_dB = fk_dB[fmask, :]    # shape: (nf_crop, nk)

# ============================
# Plot: f on x-axis (−fmax..fmax),
#       k on y-axis (−kmax..kmax)
# ============================
plt.figure(figsize=(11, 6), dpi=120)

plt.imshow(
    fk_crop_dB.T,                                   # (nk, nf)
    aspect='auto',
    origin='lower',
    extent=[freqs_crop[0], freqs_crop[-1],
            k_sym[0], k_sym[-1]],
    cmap="viridis",
    vmin=db_min,
    vmax=db_max
)

plt.colorbar(label="Amplitude (dB)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Wavenumber k (rad/m)")
plt.title("Symmetric f–k Spectrum (frequency from -fmax to +fmax)")


# ============================
# Overlay phase velocity lines
# k(f) = 2π f / v   → both +k and -k
# ============================
for v, col in zip(v_lines, v_colors):
    # positive k branch
    k_pos = 2 * np.pi * freqs_crop / v
    # negative branch
    k_neg = -k_pos

    # mask inside visible k-range
    kmin, kmax = k_sym[0], k_sym[-1]

    mask_pos = (k_pos >= kmin) & (k_pos <= kmax)
    mask_neg = (k_neg >= kmin) & (k_neg <= kmax)

    if np.any(mask_pos):
        plt.plot(freqs_crop[mask_pos], k_pos[mask_pos],
                 ls="--", color=col, lw=1, alpha=0.5)
    if np.any(mask_neg):
        plt.plot(freqs_crop[mask_neg], k_neg[mask_neg],
                 ls="--", color=col, lw=1, alpha=0.5,
                 label=f"{v:.0f} m/s")

plt.legend(loc="upper right")
plt.tight_layout()
plt.show()