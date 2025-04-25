#!/usr/bin/env python3
"""
md_spectrogram.py

Generates an mD (micro-Doppler) frequency-time spectrogram from one
frame of FMCW data. This follows a 2-stage approach:

1) Range FFT across fast time (N samples per chirp).
2) Short-Time Fourier Transform (STFT) along slow time (the M chirps),
   typically focusing on the range bin(s) where the target is strongest.

We then plot the resulting time-frequency spectrogram of the micro-Doppler.

Usage:
    python md_spectrogram.py -i radar_data.pkl -f 0 -o md_spectrogram.png
"""

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def main():
    parser = argparse.ArgumentParser(
        description="Generate micro-Doppler (mD) frequency-time spectrogram from one radar frame."
    )
    parser.add_argument('-i','--infile', type=str, default="radar_data.pkl",
                        help="Input .pkl file with recorded radar data.")
    parser.add_argument('-f','--frame_index', type=int, default=0,
                        help="Which frame to process (default=0).")
    parser.add_argument('-o','--outfile', type=str, default="md_spectrogram.png",
                        help="Output image file for the mD spectrogram.")
    parser.add_argument('--range_bin', type=int, default=-1,
                        help="If >= 0, pick this range bin for STFT. If <0, sum across all range bins.")
    parser.add_argument('--window_size', type=int, default=32,
                        help="STFT window size (in chirps).")
    parser.add_argument('--overlap', type=int, default=16,
                        help="STFT overlap (in chirps).")
    parser.add_argument('--sample_rate_slow', type=float, default=1000.0,
                        help="Slow-time sampling rate in Hz. (i.e. PRF or chirp rate). "
                             "Used to label the time axis in the STFT.")
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # 1) Load data
    # -----------------------------------------------------------------------
    with open(args.infile, 'rb') as f:
        data = pickle.load(f)

    session_name = data["session_name"]
    frames       = data["frames"]
    n_frames     = len(frames)
    print(f"Loaded {n_frames} frames from session: {session_name}")

    if args.frame_index < 0 or args.frame_index >= n_frames:
        raise ValueError(f"frame_index={args.frame_index} out of range [0..{n_frames-1}]")

    # Some radar params
    radar_params = data["radar_params"]
    max_speed    = radar_params.get("max_speed_m_s", 3.0)  # used for display if we want

    # -----------------------------------------------------------------------
    # 2) Extract one frame
    # -----------------------------------------------------------------------
    frame_data = frames[args.frame_index]
    iq_data    = frame_data["iq_data"]  # shape: (num_rx, M, N)
    # Typically: num_rx = 3, M = # chirps, N = # samples per chirp

    num_rx, M, N = iq_data.shape
    print(f"Processing frame {args.frame_index}, shape=({num_rx},{M},{N})")

    # We'll just combine all RX by summing, or pick one RX if you like
    # For micro-Doppler, often we just pick the channel with strongest SNR or sum them
    combined_iq = np.sum(iq_data, axis=0)  # shape => (M, N)

    # -----------------------------------------------------------------------
    # 3) Range FFT across fast time
    # -----------------------------------------------------------------------
    # For each chirp i, we have N samples. We'll do a 1D FFT (size=N or zero-pad).
    # Let's do a Blackman window for range dimension.
    range_window = signal.blackmanharris(N)
    # or you can do np.pad if you want bigger FFT
    # We'll store the result in range_fft_mat, shape => (M, N)
    range_fft_mat = np.zeros((M, N), dtype=np.complex64)

    for i_chirp in range(M):
        chirp_samples = combined_iq[i_chirp,:]

        # apply window
        chirp_win = chirp_samples * range_window

        # range FFT
        fft_chirp = np.fft.fft(chirp_win, n=N)
        # keep it in shape N for now
        range_fft_mat[i_chirp, :] = fft_chirp

    # Now range_fft_mat => shape (M, N)
    # "rows" = chirps (slow time), "columns" = range bins (fast freq)
    # Often we keep the first N/2 as positive range. But let's keep all for now.
    # If you want only half, do range_fft_mat[i_chirp, :N//2], etc.

    # -----------------------------------------------------------------------
    # 4) Select or sum range bins
    # -----------------------------------------------------------------------
    # If args.range_bin >= 0, we pick that bin from each chirp => shape (M,)
    # If <0, sum across all bins => shape (M,)

    if args.range_bin >= 0:
        # pick that bin
        if args.range_bin >= N:
            raise ValueError(f"range_bin={args.range_bin} out of [0..{N-1}]")
        rd_slice = range_fft_mat[:, args.range_bin]
        desc_str = f"Range bin {args.range_bin}"
    else:
        # sum across all range bins
        rd_slice = np.sum(range_fft_mat, axis=1)  # shape (M,)
        desc_str = "Sum across range bins"

    # We now have a slow-time signal of length M => rd_slice

    # -----------------------------------------------------------------------
    # 5) STFT (Short-Time Fourier Transform) along slow time
    # -----------------------------------------------------------------------
    # We'll do a standard approach: define a window_size and overlap in chirps.
    # We'll use e.g. Hanning window. Then compute the spectrogram with e.g. scipy.signal.spectrogram
    # or manually with stft. We'll do the manual approach for clarity.

    slow_window = signal.hanning(args.window_size, sym=False)

    f_slow, t_slow, Zxx = signal.stft(
        rd_slice, 
        fs=args.sample_rate_slow,   # the slow-time sampling rate in Hz, i.e. PRF
        window=slow_window,
        nperseg=args.window_size,
        noverlap=args.overlap,
        nfft=args.window_size*2,    # zero-pad in slow-time if you want finer freq resolution
        return_onesided=False       # so we get negative freq as well
    )
    # Zxx => shape (nfft, n_segments)
    # f_slow => freq bins in Hz (range: -fs/2..+fs/2 if return_onesided=False is False)
    # t_slow => time in seconds (center of each STFT window)

    # If you want the "mD frequency" in rad/s or something else, you can scale f_slow accordingly.
    # Typically micro-Doppler freq in Hz => you can convert to velocity if you know the wavelength.

    # Let's get magnitude in dB
    Zxx_dB = 20.0 * np.log10(np.abs(Zxx) + 1e-6)

    # If you want negative frequencies at bottom, reorder the frequency axis
    # Because we used return_onesided=False, f_slow is from -fs/2..+fs/2
    # Let's reorder so that negative freq is at the bottom
    idx_sort = np.argsort(f_slow)
    f_slow_sorted = f_slow[idx_sort]
    Zxx_dB_sorted = Zxx_dB[idx_sort, :]

    # -----------------------------------------------------------------------
    # 6) Plot the mD frequency-time spectrogram
    # -----------------------------------------------------------------------
    plt.figure(figsize=(8,6))
    plt.title(f"mD Frequency-Time Spectrogram\n{desc_str}, Frame {args.frame_index}, {session_name}")
    # extent => time from t_slow[0]..t_slow[-1], freq from f_slow_sorted[0]..f_slow_sorted[-1]
    plt.imshow(
        Zxx_dB_sorted,
        extent=[t_slow[0], t_slow[-1], f_slow_sorted[0], f_slow_sorted[-1]],
        aspect='auto',
        origin='lower',
        cmap='jet'
    )
    plt.colorbar(label="Amplitude (dB)")
    plt.xlabel("Time (slow-time seconds)")
    plt.ylabel("mD Frequency (Hz)")
    plt.tight_layout()
    plt.savefig(args.outfile, dpi=150)
    plt.show()

if __name__ == "__main__":
    main()
