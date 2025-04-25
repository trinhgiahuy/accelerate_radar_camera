#!/usr/bin/env python3

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import median_filter

def main():
    parser = argparse.ArgumentParser(
        description="Advanced publication-style velocity-time spectrogram with 3-RX summation, STFT overlap, and median filtering."
    )
    parser.add_argument('-i','--infile', type=str, default="radar_data.pkl",
                        help="Input pickle file with recorded data.")
    parser.add_argument('-o','--outfile', type=str, default="spectrogram.png",
                        help="Output image file.")
    parser.add_argument('--mti_alpha', type=float, default=1.0,
                        help="Alpha for internal exponential average (clutter removal).")
    parser.add_argument('--final_alpha', type=float, default=0.9,
                        help="Alpha for final velocity domain MTI filter.")
    parser.add_argument('--overlap_frames', type=int, default=5,
                        help="Number of frames in each short-time window for overlap.")
    parser.add_argument('--median_size', type=int, default=3,
                        help="Kernel size for 2D median filter (e.g. 3 => 3x3).")
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # 1) Load Data
    # -----------------------------------------------------------------------
    with open(args.infile, 'rb') as f:
        data = pickle.load(f)

    session_name = data["session_name"]
    frames       = data["frames"]         
    radar_params = data["radar_params"]   
    frame_rate   = data["frame_rate"]     
    n_frames     = len(frames)

    print(f"Loaded {n_frames} frames from session: {session_name}")
    print(f"Frame rate = {frame_rate} fps")

    # Check shape
    example_iq   = frames[0]["iq_data"]
    num_rx, n_chirps_frame, n_samples_chirp = example_iq.shape
    print(f"Data shape: (num_rx={num_rx}, n_chirps={n_chirps_frame}, samples={n_samples_chirp})")

    # If you want finer velocity resolution, ensure n_chirps_frame=256 in your new recordings
    # (or at least 128). More chirps => less 'blocky' velocity axis.

    dt_frame = 1.0 / frame_rate
    time_axis_frames = np.linspace(0, (n_frames-1)*dt_frame, n_frames)

    # -----------------------------------------------------------------------
    # 2) Range-Doppler Processing Setup
    # -----------------------------------------------------------------------
    range_window   = signal.blackmanharris(n_samples_chirp).reshape(1, n_samples_chirp)
    doppler_window = signal.blackmanharris(n_chirps_frame).reshape(1, n_chirps_frame)

    n_doppler_bins = 2 * n_chirps_frame
    # We'll keep half of the range dimension => n_samples_chirp//2
    n_range_bins = n_samples_chirp // 2

    # For internal exponential average
    dopp_avg = np.zeros((n_range_bins, n_doppler_bins, num_rx), dtype=np.complex64)

    # We'll store a single velocity profile (summed across range) per frame => shape (n_frames, n_doppler_bins)
    velocity_profiles = np.zeros((n_frames, n_doppler_bins), dtype=np.float32)

    # final alpha-based filter
    t_prev = np.zeros(n_doppler_bins, dtype=np.float32)

    # -----------------------------------------------------------------------
    # 3) Process Each Frame
    # -----------------------------------------------------------------------
    for i_frame, fr in enumerate(frames):
        iq_data = fr["iq_data"]  # shape: (num_rx, n_chirps_frame, n_samples_chirp)

        # Accumulate range-doppler across 3 RX
        fft_2_sum = np.zeros((n_range_bins, n_doppler_bins), dtype=np.complex64)

        for rx_idx in range(num_rx):
            mat = iq_data[rx_idx,:,:]

            # 1) Debias
            chirp_means = np.mean(mat, axis=1, keepdims=True)
            mat_debiased = mat - chirp_means

            # 2) Range window
            mat_win_range = mat_debiased * range_window

            # 3) Zero-pad in range dimension
            zp_range = np.pad(mat_win_range, ((0,0),(0,n_samples_chirp)), 'constant')

            # 4) Range FFT
            range_fft = np.fft.fft(zp_range, axis=1) / n_samples_chirp
            # keep only first half
            range_fft = 2.0 * range_fft[:, :n_range_bins]

            # 5) Transpose => shape (n_range_bins, n_chirps_frame)
            fft1d = range_fft.T

            # 6) Doppler window
            fft1d_win = fft1d * doppler_window

            # 7) Zero-pad in doppler dimension
            zp_doppler = np.pad(fft1d_win, ((0,0),(0,n_chirps_frame)), 'constant')

            # 8) Doppler FFT
            fft2d = np.fft.fft(zp_doppler, axis=1) / n_chirps_frame

            # 9) Internal exponential average (clutter removal)
            dopp_avg[:,:,rx_idx] = args.mti_alpha*dopp_avg[:,:,rx_idx] + (1.0 - args.mti_alpha)*fft2d
            fft2d_mti = fft2d - dopp_avg[:,:,rx_idx]

            # 10) Shift doppler axis
            dopplerfft = np.fft.fftshift(fft2d_mti, axes=(1,))

            fft_2_sum += dopplerfft

        # Sum of 3 RX => shape (n_range_bins, n_doppler_bins)
        rd_abs = np.abs(fft_2_sum)

        # collapse range => e.g. max across range
        rd_profile = np.max(rd_abs, axis=0)

        # final alpha filter in velocity domain
        t_cur = args.final_alpha * rd_profile + (1.0 - args.final_alpha)*t_prev
        r_filt = np.abs(rd_profile - t_prev)

        velocity_profiles[i_frame,:] = r_filt
        t_prev = t_cur

        print(f"Processed frame {i_frame+1}/{n_frames}")

    # -----------------------------------------------------------------------
    # 4) Short-Time Overlap in Time + 2D Median Filter
    # -----------------------------------------------------------------------
    # We'll do the STFT approach with partial overlap, then a median filter on the 2D matrix.

    window_size = args.overlap_frames
    hop_size    = window_size // 2 if window_size>1 else 1

    time_stft   = []
    vel_stft    = []

    # a time window for weighting frames in each segment
    time_window = signal.hanning(window_size, sym=False)

    i_start = 0
    while i_start + window_size <= n_frames:
        segment = velocity_profiles[i_start : i_start+window_size, :].copy()  # shape (window_size, n_doppler_bins)

        # multiply by time window
        for i_seg in range(window_size):
            segment[i_seg,:] *= time_window[i_seg]

        seg_mean = np.mean(segment, axis=0)  # shape (n_doppler_bins,)
        vel_stft.append(seg_mean)

        mid_frame = i_start + (window_size//2)
        time_stft.append(time_axis_frames[mid_frame])

        i_start += hop_size

    vel_stft = np.array(vel_stft)   # shape (n_windows, n_doppler_bins)
    time_stft = np.array(time_stft) # shape (n_windows,)

    # convert amplitude to dB
    vel_stft_db = 20.0 * np.log10(vel_stft + 1e-6)

    # 2D median filter to reduce random speckle => filter shape = (args.median_size, args.median_size)
    # But note our matrix is shape (time, velocity). We'll do:
    vel_stft_db_filt = median_filter(vel_stft_db, size=(args.median_size, args.median_size))

    # velocity axis
    max_speed = radar_params.get("max_speed_m_s", 3)
    vel_axis = np.linspace(-max_speed, max_speed, vel_stft.shape[1], endpoint=False)

    # -----------------------------------------------------------------------
    # 5) Plot
    # -----------------------------------------------------------------------
    plt.figure(figsize=(8,5))
    plt.title(f"Velocity–Time Spectrogram (Session: {session_name})\n"
              f"STFT overlap={window_size}, median filter={args.median_size}x{args.median_size}")
    plt.imshow(
        vel_stft_db_filt.T,
        extent=[time_stft[0], time_stft[-1], vel_axis[0], vel_axis[-1]],
        aspect='auto',
        origin='lower',
        cmap='jet'
    )
    plt.colorbar(label="Amplitude (dB)")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.tight_layout()
    plt.savefig(args.outfile, dpi=150)
    plt.show()

if __name__ == "__main__":
    main()
