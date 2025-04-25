#!/usr/bin/env python3

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def main():
    parser = argparse.ArgumentParser(
        description="Generate a single micro-Doppler spectrogram over the entire multi-frame recording."
    )
    parser.add_argument('-i','--infile', type=str, default="radar_data.pkl",
                        help="Input .pkl file with recorded data (multiple frames).")
    parser.add_argument('-o','--outfile', type=str, default="md_spectrogram_entire.png",
                        help="Output image file.")
    parser.add_argument('--range_bin', type=int, default=-1,
                        help="If >=0, pick that bin +/- bin_window. If <0, sum all bins.")
    parser.add_argument('--bin_window', type=int, default=3,
                        help="If range_bin>=0, also sum +/- bin_window around that bin.")
    parser.add_argument('--dc_remove', action='store_true',
                        help="If set, remove DC offset in slow-time domain by subtracting the mean.")
    parser.add_argument('--hp_cutoff', type=float, default=0.0,
                        help="Optional high-pass cutoff in Hz (slow-time domain). 0 => no HP filter.")
    parser.add_argument('--window_size', type=int, default=128,
                        help="STFT window size in chirps.")
    parser.add_argument('--overlap', type=int, default=64,
                        help="STFT overlap in chirps.")
    parser.add_argument('--onesided', action='store_true',
                        help="If set, only plot 0..+Fmax in STFT (return_onesided=True).")
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # 1) Load the entire dataset
    # -----------------------------------------------------------------------
    with open(args.infile, 'rb') as f:
        data = pickle.load(f)

    session_name = data["session_name"]
    frames       = data["frames"]   # list of length F
    radar_params = data["radar_params"]
    frame_rate   = data["frame_rate"]
    F            = len(frames)      # number of frames

    print(f"Loaded {F} frames from session: {session_name}")
    print(f"Frame rate = {frame_rate} fps")

    # Inspect the shape of the first frame
    ex_iq = frames[0]["iq_data"]   # shape => (num_rx, M, N)
    num_rx, M, N = ex_iq.shape
    print(f"Each frame shape: (num_rx={num_rx}, M={M}, N={N})")

    # => total chirps = F*M
    # slow-time sampling rate => M * frame_rate (chirps per second)
    slow_time_fs = M * frame_rate
    total_chirps = F * M

    print(f"Total chirps across all frames = {total_chirps}")
    print(f"Inferred slow-time sample rate = {slow_time_fs} Hz")

    # -----------------------------------------------------------------------
    # 2) Combine all frames into a single slow-time array
    #    We'll build all_chirps shape => (F*M, N)
    # -----------------------------------------------------------------------
    all_chirps = np.zeros((total_chirps, N), dtype=np.complex64)

    idx = 0
    for f_idx in range(F):
        # shape => (num_rx, M, N)
        frame_iq = frames[f_idx]["iq_data"]
        # sum across RX
        sum_rx = np.sum(frame_iq, axis=0)  # shape => (M, N)

        # store into all_chirps
        all_chirps[idx : idx+M, :] = sum_rx
        idx += M

    # -----------------------------------------------------------------------
    # 3) Range FFT for each chirp
    # -----------------------------------------------------------------------
    # We'll do a Blackman-Harris window in range dimension
    range_window = signal.blackmanharris(N)

    # We'll store the range FFT in range_fft_mat => shape (total_chirps, N)
    range_fft_mat = np.zeros((total_chirps, N), dtype=np.complex64)

    for c_idx in range(total_chirps):
        chirp_samples = all_chirps[c_idx,:]
        chirp_win = chirp_samples * range_window
        fft_chirp = np.fft.fft(chirp_win, n=N)
        range_fft_mat[c_idx, :] = fft_chirp

    # Now range_fft_mat => shape (total_chirps, N)
    # "rows" = slow-time index (0..F*M-1), "columns" = range bin

    # -----------------------------------------------------------------------
    # 4) Select or sum range bins
    # -----------------------------------------------------------------------
    if args.range_bin >= 0:
        center_bin = args.range_bin
        low_bin    = max(0, center_bin - args.bin_window)
        high_bin   = min(N, center_bin + args.bin_window + 1)
        rd_slice   = np.sum(range_fft_mat[:, low_bin:high_bin], axis=1)  # shape => (F*M,)
        desc_str   = f"Range bin {center_bin} +/- {args.bin_window}"
    else:
        # sum across ALL bins
        rd_slice = np.sum(range_fft_mat, axis=1)  # shape => (F*M,)
        desc_str = "Sum across all bins"

    # -----------------------------------------------------------------------
    # 5) Optional DC removal / HP filter in slow-time domain
    # -----------------------------------------------------------------------
    if args.dc_remove:
        print("DC Remove")
        mean_val = np.mean(rd_slice)
        rd_slice = rd_slice - mean_val
        desc_str += ", DC removed"

    if args.hp_cutoff > 0.0:
        sos = signal.butter(4, args.hp_cutoff/(slow_time_fs/2.0), btype='high', output='sos')
        rd_slice = signal.sosfilt(sos, rd_slice)
        desc_str += f", HP>={args.hp_cutoff}Hz"

    # Now we have a single slow-time signal of length total_chirps => rd_slice

    # -----------------------------------------------------------------------
    # 6) STFT on the entire slow-time
    # -----------------------------------------------------------------------
    stft_window = signal.windows.hanning(args.window_size, sym=False)
    return_onesided = args.onesided

    f_stft, t_stft, Zxx = signal.stft(
        rd_slice,
        fs=slow_time_fs,
        window=stft_window,
        nperseg=args.window_size,
        noverlap=args.overlap,
        nfft=args.window_size*2,  # zero-pad
        return_onesided=return_onesided
    )
    Zxx_dB = 20.0 * np.log10(np.abs(Zxx) + 1e-6)

    # If we did two-sided, reorder freq so negative is at bottom
    if not return_onesided:
        idx_sort = np.argsort(f_stft)
        f_stft = f_stft[idx_sort]
        Zxx_dB = Zxx_dB[idx_sort,:]

    # Convert time from STFT frames to actual seconds
    # t_stft is in seconds from the STFT call. So 0..(total_chirps/slow_time_fs).

    # -----------------------------------------------------------------------
    # 7) Plot
    # -----------------------------------------------------------------------
    plt.figure(figsize=(8,6))
    plt.title(f"mD Frequency-Time Spectrogram over entire {F} frames\n"
              f"{desc_str}, {session_name}")
    time_extent = [t_stft[0], t_stft[-1]]
    freq_extent = [f_stft[0], f_stft[-1]]

    plt.imshow(
        Zxx_dB,
        extent=[time_extent[0], time_extent[1], freq_extent[0], freq_extent[1]],
        aspect='auto',
        origin='lower',
        cmap='jet'
    )
    plt.colorbar(label="Amplitude (dB)")
    plt.xlabel("Time (s) [slow-time across entire measurement]")
    plt.ylabel("mD Frequency (Hz)")
    plt.tight_layout()
    plt.savefig(args.outfile, dpi=150)
    plt.show()

if __name__ == "__main__":
    main()
