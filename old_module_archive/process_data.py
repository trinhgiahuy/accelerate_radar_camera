#!/usr/bin/env python3
"""
process_radar.py

Script to load raw radar data (recorded by record_radar.py) and
perform range-Doppler processing with an MTI filter. Produces a
velocity–time spectrogram by collapsing the range dimension
(e.g. taking max across range) and applying an additional
MTI filter as described in the literature.

Usage:
    python process_radar.py -i radar_data.pkl -o spectrogram.png

This will load 'radar_data.pkl', process all frames, and save a
velocity-time plot to 'spectrogram.png'. It will also display
the plot if a display is available.
"""

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def main():
    parser = argparse.ArgumentParser(description="Process recorded radar data and generate velocity–time spectrogram.")
    parser.add_argument('-i', '--infile', type=str, default="radar_data.pkl",
                        help="Input pickle file with recorded radar data.")
    parser.add_argument('-o', '--outfile', type=str, default="spectrogram.png",
                        help="Output image file for the spectrogram.")
    parser.add_argument('--mti_alpha', type=float, default=1.0,
                        help="Alpha for static clutter removal (internal exponential average).")
    parser.add_argument('--final_alpha', type=float, default=0.9,
                        help="Alpha for final MTI filter in velocity-time domain.")
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # 1) Load the recorded data
    # -----------------------------------------------------------------------
    with open(args.infile, 'rb') as f:
        data = pickle.load(f)

    session_name = data["session_name"]
    params_dict  = data["radar_params"]
    frames       = data["frames"]
    frame_rate   = data["frame_rate"]
    n_frames     = data["num_frames"]

    # Extract radar parameters
    num_chirps    = params_dict["num_samples_per_chirp"]  # confusing naming, but from your code
    chirp_samples = params_dict["num_samples_per_chirp"]
    # Actually check your stored dictionary carefully: you might need to rename keys if they differ.
    # For example, you might have 'max_speed_m_s' etc.

    max_speed_m_s = params_dict["max_speed_m_s"]  # ±3 m/s from your example
    # We'll also do half the range bins for positive frequencies after range FFT
    # but let's confirm with the actual data shape.

    # Let's examine the shape of the first frame
    # frames[0]["iq_data"] => shape: (num_rx, num_chirps, chirp_samples)
    example_shape = frames[0]["iq_data"].shape
    num_rx, actual_num_chirps, actual_chirp_samples = example_shape
    # The 'num_chirps' from your 'metrics' might differ from 'num_samples_per_chirp'; be sure to keep them distinct.
    # In your prior code, 'metrics.num_chirps_per_frame' is the # of chirps, and 'metrics.num_samples_per_chirp' is the # of samples each chirp.

    # For clarity, rename:
    n_chirps_frame = actual_num_chirps
    n_samples_chirp = actual_chirp_samples

    print("===========================================")
    print(f"Session name:         {session_name}")
    print(f"Number of frames:     {n_frames}")
    print(f"Frame rate:           {frame_rate} Hz")
    print(f"Data shape [0]:       (num_rx, num_chirps, chirp_samples) = {example_shape}")
    print(f"max_speed_m_s:        {max_speed_m_s}")
    print(f"MTI alpha (internal): {args.mti_alpha}")
    print(f"Final alpha:          {args.final_alpha}")
    print("===========================================")

    # -----------------------------------------------------------------------
    # 2) Prepare for Range-Doppler Processing
    # -----------------------------------------------------------------------
    # We'll define:
    #   - A window in the range dimension (Blackman-Harris)
    #   - A window in the Doppler dimension (Blackman-Harris)
    #   - Zero-padding
    #   - Summation across antennas
    #   - A running exponential average for static clutter removal (like your code)
    #   - Then we take max across range to get a velocity profile
    #   - We apply the final literature-based MTI filter to produce r_filt

    # 2D shape after range & doppler transforms: (range_bins, doppler_bins)
    #   range_bins   ~ n_samples_chirp//2
    #   doppler_bins = n_chirps_frame * 2   (because we zero-pad chirps dimension and FFT)

    range_bins   = n_samples_chirp // 2
    doppler_bins = n_chirps_frame * 2

    # Windows
    range_window   = signal.blackmanharris(n_samples_chirp).reshape(1, n_samples_chirp)
    doppler_window = signal.blackmanharris(n_chirps_frame).reshape(1, n_chirps_frame)

    # We'll keep an exponential average for each antenna:
    # shape => (range_bins, doppler_bins, num_rx)
    dopp_avg = np.zeros((range_bins, doppler_bins, num_rx), dtype=np.complex64)

    # We also keep an array to store the final velocity profile for each frame
    vel_time_map = np.zeros((n_frames, doppler_bins), dtype=np.float32)

    # We'll keep a 1D array for the final "MTI" filter state across doppler_bins
    t_prev = np.zeros(doppler_bins, dtype=np.float32)

    # Time axis
    time_axis = np.linspace(0, (n_frames-1)/frame_rate, n_frames)

    # -----------------------------------------------------------------------
    # 3) Process each frame
    # -----------------------------------------------------------------------
    for i_frame, frame_dict in enumerate(frames):
        # shape: (num_rx, n_chirps_frame, n_samples_chirp)
        frame_iq = frame_dict["iq_data"]

        # We'll accumulate the range-Doppler across antennas
        fft_2_sum = np.zeros((range_bins, doppler_bins), dtype=np.complex64)

        for rx_idx in range(num_rx):
            mat = frame_iq[rx_idx, :, :]  # shape: (n_chirps_frame, n_samples_chirp)

            # (A) Debias each chirp
            chirp_means = np.mean(mat, axis=1, keepdims=True)
            mat_debiased = mat - chirp_means

            # (B) Window in range dimension
            mat_win_range = mat_debiased * range_window  # shape still (n_chirps_frame, n_samples_chirp)

            # (C) Zero-pad in range dimension
            # We'll double the samples to do a size=2*n_samples_chirp FFT if you like.
            # But your code shows pad((0,0),(0, n_samples_chirp)).
            zp1 = np.pad(mat_win_range, ((0,0),(0, n_samples_chirp)), mode='constant')
            # shape => (n_chirps_frame, 2*n_samples_chirp)

            # (D) Range FFT along axis=1
            range_fft = np.fft.fft(zp1, axis=1) / n_samples_chirp
            # Keep only the first half
            range_fft = 2.0 * range_fft[:, :range_bins]

            # (E) Transpose so range is along axis=0, chirps along axis=1
            # shape => (range_bins, n_chirps_frame)
            fft1d = range_fft.T

            # (F) Window in Doppler dimension
            fft1d = fft1d * doppler_window  # broadcast over range_bins

            # (G) Zero-pad in doppler dimension
            zp2 = np.pad(fft1d, ((0,0),(0, n_chirps_frame)), mode='constant')
            # shape => (range_bins, 2*n_chirps_frame)

            # (H) Doppler FFT along axis=1
            fft2d = np.fft.fft(zp2, axis=1) / n_chirps_frame

            # (I) Exponential average for static clutter
            dopp_avg[:,:,rx_idx] = (args.mti_alpha * dopp_avg[:,:,rx_idx]) + \
                                   ((1.0 - args.mti_alpha) * fft2d)

            # (J) Subtract the average from the current to remove static
            fft2d_mti = fft2d - dopp_avg[:,:,rx_idx]

            # (K) Shift so zero Doppler is in the center
            dopplerfft = np.fft.fftshift(fft2d_mti, axes=(1,))

            # Accumulate across antennas
            fft_2_sum += dopplerfft

        # Now fft_2_sum is the combined range-Doppler (range_bins x doppler_bins)

        # (L) Take magnitude
        rd_abs = np.abs(fft_2_sum)

        # (M) Collapse across range by max or sum
        # For velocity-time, we often do max across range:
        rd_profile = np.max(rd_abs, axis=0)  # shape => (doppler_bins,)

        # (N) Final "literature" MTI filter
        #   t_i = alpha * r_i + (1-alpha)* t_{i-1}
        #   r_filt = | r_i - t_{i-1} |
        alpha = args.final_alpha
        t_cur = alpha * rd_profile + (1.0 - alpha)*t_prev
        r_filt = np.abs(rd_profile - t_prev)

        # (O) Store the result
        vel_time_map[i_frame, :] = r_filt
        t_prev = t_cur

        print(f"Processed frame {i_frame+1}/{n_frames}")

    # -----------------------------------------------------------------------
    # 4) Plot velocity–time spectrogram
    # -----------------------------------------------------------------------
    # Velocity axis: from -max_speed_m_s .. +max_speed_m_s
    vel_axis = np.linspace(-max_speed_m_s, max_speed_m_s, doppler_bins, endpoint=False)

    # Convert amplitude to dB
    vel_time_db = 20.0 * np.log10(vel_time_map + 1e-6)

    plt.figure(figsize=(8,5))
    plt.title(f"Velocity–Time Spectrogram (Session: {session_name})")
    plt.imshow(
        vel_time_db.T,
        extent=[time_axis[0], time_axis[-1], vel_axis[0], vel_axis[-1]],
        aspect='auto',
        origin='lower',
        cmap='jet'
    )
    plt.colorbar(label="Amplitude (dB)")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")

    plt.tight_layout()
    plt.savefig(args.outfile, dpi=150)
    print(f"Saved spectrogram to {args.outfile}")
    plt.show()

if __name__ == "__main__":
    main()
