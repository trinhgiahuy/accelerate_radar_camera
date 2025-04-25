#!/usr/bin/env python3

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def main():
    parser = argparse.ArgumentParser(
        description="Process recorded radar data with STFT overlap in time, using 3 RX channels, 128 chirps, 64 samples, etc.")
    parser.add_argument('-i','--infile', type=str, default="radar_data.pkl",
                        help="Input pickle file with recorded data.")
    parser.add_argument('-o','--outfile', type=str, default="spectrogram.png",
                        help="Output image file for the spectrogram.")
    parser.add_argument('--mti_alpha', type=float, default=1.0,
                        help="Internal alpha for exponential average clutter removal.")
    parser.add_argument('--final_alpha', type=float, default=0.9,
                        help="Final literature-based MTI alpha for velocity-time filtering.")
    parser.add_argument('--overlap_frames', type=int, default=5,
                        help="Number of frames in each short-time window for overlap. E.g. 5 means ~0.5s if fps=10.")
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # 1) Load Data
    # -----------------------------------------------------------------------
    with open(args.infile, 'rb') as f:
        data = pickle.load(f)

    session_name = data["session_name"]
    frames       = data["frames"]         # list of dicts
    radar_params = data["radar_params"]   # from your record script
    frame_rate   = data["frame_rate"]     # e.g. 10 fps
    n_frames     = len(frames)

    print(f"Loaded {n_frames} frames from session: {session_name}")
    print(f"Frame rate = {frame_rate} fps")

    # Check shape of first frame
    # frames[0]["iq_data"] => shape (num_rx, n_chirps, n_samples_per_chirp)
    example_iq   = frames[0]["iq_data"]
    num_rx, n_chirps_frame, n_samples_chirp = example_iq.shape
    print(f"Data shape for first frame: {example_iq.shape} (num_rx, chirps, samples)")

    # We assume 3 RX, 128 chirps, 64 samples
    # Time between frames
    dt_frame = 1.0 / frame_rate
    time_axis_frames = np.linspace(0, (n_frames-1)*dt_frame, n_frames)

    # -----------------------------------------------------------------------
    # 2) Prepare for Range-Doppler processing
    # -----------------------------------------------------------------------
    # Window in range dimension (Blackman-Harris)
    range_window = signal.blackmanharris(n_samples_chirp).reshape(1, n_samples_chirp)

    # Window in Doppler dimension
    doppler_window = signal.blackmanharris(n_chirps_frame).reshape(1, n_chirps_frame)

    # We'll have n_doppler_bins = 2 * n_chirps_frame (after zero-padding + shift)
    n_doppler_bins = 2 * n_chirps_frame

    # For internal exponential average (static clutter removal)
    dopp_avg = np.zeros((n_samples_chirp//2, n_doppler_bins, num_rx), dtype=np.complex64)

    # We'll store one velocity profile per frame => shape (n_frames, n_doppler_bins)
    velocity_profiles = np.zeros((n_frames, n_doppler_bins), dtype=np.float32)

    # Final alpha filter state across velocity bins
    t_prev = np.zeros(n_doppler_bins, dtype=np.float32)

    # -----------------------------------------------------------------------
    # 3) Process Each Frame -> get velocity profile
    # -----------------------------------------------------------------------
    for i_frame, frame_dict in enumerate(frames):
        frame_iq = frame_dict["iq_data"]  # shape: (num_rx, n_chirps_frame, n_samples_chirp)

        # Accumulate range-Doppler over 3 RX
        fft_2_sum = np.zeros((n_samples_chirp//2, n_doppler_bins), dtype=np.complex64)

        for rx_idx in range(num_rx):
            mat = frame_iq[rx_idx,:,:]  # shape: (n_chirps_frame, n_samples_chirp)

            # 1) Debias
            mat_means = np.mean(mat, axis=1, keepdims=True)
            mat_debiased = mat - mat_means

            # 2) Window in range
            mat_win_range = mat_debiased * range_window

            # 3) Zero-pad in range dimension
            zp_range = np.pad(mat_win_range, ((0,0),(0,n_samples_chirp)), 'constant')
            # shape => (n_chirps_frame, 2*n_samples_chirp)

            # 4) Range FFT
            range_fft = np.fft.fft(zp_range, axis=1) / n_samples_chirp
            # keep only first half => shape (n_chirps_frame, n_samples_chirp//2)
            range_fft = 2.0 * range_fft[:, :n_samples_chirp//2]

            # 5) transpose => shape (n_samples_chirp//2, n_chirps_frame)
            fft1d = range_fft.T

            # 6) Window in Doppler dimension
            fft1d_win = fft1d * doppler_window  # broadcast

            # 7) Zero-pad in doppler dimension
            zp_doppler = np.pad(fft1d_win, ((0,0),(0,n_chirps_frame)), 'constant')
            # => shape (n_samples_chirp//2, 2*n_chirps_frame)

            # 8) Doppler FFT
            fft2d = np.fft.fft(zp_doppler, axis=1) / n_chirps_frame

            # 9) Internal exponential average (static clutter removal)
            dopp_avg[:,:,rx_idx] = (args.mti_alpha * dopp_avg[:,:,rx_idx]) \
                                   + ((1.0 - args.mti_alpha) * fft2d)
            fft2d_mti = fft2d - dopp_avg[:,:,rx_idx]

            # 10) Shift Doppler axis
            dopplerfft = np.fft.fftshift(fft2d_mti, axes=(1,))

            fft_2_sum += dopplerfft

        # Summation over RX => shape (n_samples_chirp//2, n_doppler_bins)
        rd_abs = np.abs(fft_2_sum)

        # Collapse range dimension by max or sum
        rd_profile = np.max(rd_abs, axis=0)  # shape => (n_doppler_bins,)

        # final alpha-based MTI in velocity domain
        t_cur  = args.final_alpha * rd_profile + (1.0 - args.final_alpha)*t_prev
        r_filt = np.abs(rd_profile - t_prev)

        velocity_profiles[i_frame,:] = r_filt
        t_prev = t_cur

        print(f"Processed frame {i_frame+1}/{n_frames}")

    # -----------------------------------------------------------------------
    # 4) Short-Time Overlap in Time (STFT approach)
    # -----------------------------------------------------------------------
    # Instead of plotting velocity_profiles[i,:] for i in [0..n_frames-1],
    # we do a sliding window over frames. For example, each window = args.overlap_frames,
    # step by half that window => 50% overlap. Then we average (or do a small 1D transform) 
    # across those frames. This yields a smoother time axis.

    window_size = args.overlap_frames
    hop_size    = window_size // 2 if window_size>1 else 1  # 50% overlap
    time_stft   = []
    vel_stft    = []

    # define a window function across the "window_size" frames
    # e.g., Hanning or Blackman
    time_window = signal.hanning(window_size, sym=False)

    i_start = 0
    while i_start + window_size <= n_frames:
        # frames in [i_start : i_start+window_size]
        segment = velocity_profiles[i_start : i_start+window_size, :]  # shape (window_size, n_doppler_bins)

        # multiply each frame row by time_window => shape (window_size,) broadcast
        # do axis=0 for doppler bins
        for i_seg in range(window_size):
            segment[i_seg,:] *= time_window[i_seg]

        # average across the window dimension => shape (n_doppler_bins,)
        # or you could do a small FFT across time for another dimension
        seg_mean = np.mean(segment, axis=0)

        # store it
        vel_stft.append(seg_mean)

        # midpoint time
        mid_frame = i_start + (window_size//2)
        time_stft.append(time_axis_frames[mid_frame])

        i_start += hop_size

    vel_stft = np.array(vel_stft)          # shape (n_windows, n_doppler_bins)
    time_stft = np.array(time_stft)        # shape (n_windows,)

    # Convert amplitude to dB
    vel_stft_db = 20.0 * np.log10(vel_stft + 1e-6)

    # velocity axis from -max_speed.. +max_speed 
    # you might retrieve from radar_params or simply define:
    max_speed = radar_params.get("max_speed_m_s", 3)
    vel_axis = np.linspace(-max_speed, max_speed, n_doppler_bins, endpoint=False)

    # -----------------------------------------------------------------------
    # 5) Plot the Overlapped STFT Velocity-Time
    # -----------------------------------------------------------------------
    plt.figure(figsize=(8,5))
    plt.title(f"Velocity–Time Spectrogram (Session: {session_name})\nShort-Time Overlap = {window_size}")
    plt.imshow(
        vel_stft_db.T,
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
