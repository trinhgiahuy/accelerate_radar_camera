#!/usr/bin/env python3
"""
range_doppler.py

Loads radar data (from the same .pkl file used by record_radar.py),
performs Range-Doppler processing on a selected frame, and plots a
Range-Doppler heatmap (magnitude in dB). Optionally sums across 3 RX
antennas. 

Usage example:
    python range_doppler.py -i radar_data.pkl -f 0 -o rd_map.png
"""

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def main():
    parser = argparse.ArgumentParser(description="Generate Range-Doppler heatmap from radar data.")
    parser.add_argument('-i','--infile', type=str, default='radar_data.pkl',
                        help="Input pickle file with recorded data.")
    parser.add_argument('-f','--frame_index', type=int, default=0,
                        help="Which frame to process for Range-Doppler map (default=0).")
    parser.add_argument('-o','--outfile', type=str, default='rd_map.png',
                        help="Output image file for the Range-Doppler map.")
    parser.add_argument('--mti_alpha', type=float, default=1.0,
                        help="Alpha for exponential average clutter removal. (1.0 => strong removal)")
    parser.add_argument('--sum_rx', action='store_true',
                        help="If set, sum across all RX antennas; otherwise just use RX0.")
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # 1) Load the .pkl data
    # -----------------------------------------------------------------------
    with open(args.infile, 'rb') as f:
        data = pickle.load(f)

    session_name = data["session_name"]
    frames       = data["frames"]         # list of dict
    radar_params = data["radar_params"]   # from record_radar
    frame_rate   = data["frame_rate"]
    n_frames     = len(frames)

    print(f"Loaded {n_frames} frames from session: {session_name}")

    # Check requested frame index
    if args.frame_index < 0 or args.frame_index >= n_frames:
        raise ValueError(f"Invalid frame index {args.frame_index}. Must be in [0..{n_frames-1}]")

    # Some relevant parameters from the first frame
    example_iq   = frames[0]["iq_data"]
    num_rx, n_chirps_frame, n_samples_chirp = example_iq.shape

    # We also want to define or retrieve max_speed and max_range
    max_speed  = radar_params.get("max_speed_m_s", 3)  # ±3 m/s
    # For range, many Infineon demos use metrics.max_range_m. If not stored, we approximate:
    # Rmax ~ c * Tchirp / 2. But let's see if "range_resolution_m" is in radar_params.
    range_res_m = radar_params.get("range_resolution_m", 0.15)
    # Then max_range might be ~ (n_samples_chirp * range_res_m). 
    # Or you can just do half that if you're ignoring negative freq. 
    # For a simpler approach:
    max_range   = n_samples_chirp * range_res_m

    print(f"Frame rate: {frame_rate} Hz, shape=({num_rx},{n_chirps_frame},{n_samples_chirp})")
    print(f"max_speed={max_speed}, range_res={range_res_m}, approximate max_range={max_range:.2f} m")

    # -----------------------------------------------------------------------
    # 2) Prepare windows & arrays
    # -----------------------------------------------------------------------
    # We'll do a Range FFT with zero-padding, keep half, then a Doppler FFT (also zero-padded).
    # The result is shape: (range_bins, doppler_bins). We'll plot magnitude in dB.

    # Window in range dimension
    range_window = signal.blackmanharris(n_samples_chirp).reshape(1, n_samples_chirp)

    # Window in Doppler dimension
    doppler_window = signal.blackmanharris(n_chirps_frame).reshape(1, n_chirps_frame)

    # We'll do a single static clutter removal with an exponential average
    # shape => (range_bins, doppler_bins, num_rx)
    # but if we're only doing one frame, we won't see much effect from an average across frames
    # We'll keep it here for demonstration.
    n_doppler_bins = 2 * n_chirps_frame
    n_range_bins   = n_samples_chirp // 2
    static_avg = np.zeros((n_range_bins, n_doppler_bins, num_rx), dtype=np.complex64)

    # -----------------------------------------------------------------------
    # 3) Range-Doppler transform for the selected frame
    # -----------------------------------------------------------------------
    frame_dict = frames[args.frame_index]
    frame_iq   = frame_dict["iq_data"]  # shape (num_rx, n_chirps_frame, n_samples_chirp)

    # We'll accumulate the range-doppler across antennas if --sum_rx is set
    rd_sum = np.zeros((n_range_bins, n_doppler_bins), dtype=np.complex64)

    for rx_idx in range(num_rx):
        if (not args.sum_rx) and (rx_idx > 0):
            # If we only want RX0, skip others
            continue

        mat = frame_iq[rx_idx,:,:]  # shape: (n_chirps_frame, n_samples_chirp)

        # 1) Debias each chirp
        chirp_means = np.mean(mat, axis=1, keepdims=True)
        mat_debiased = mat - chirp_means

        # 2) Range window
        mat_win = mat_debiased * range_window

        # 3) Zero-pad in range dimension
        zp_range = np.pad(mat_win, ((0,0),(0,n_samples_chirp)), 'constant')
        # => shape (n_chirps_frame, 2*n_samples_chirp)

        # 4) Range FFT
        range_fft = np.fft.fft(zp_range, axis=1) / n_samples_chirp
        # keep only first half => shape => (n_chirps_frame, n_range_bins)
        range_fft = 2.0 * range_fft[:, :n_range_bins]

        # 5) Transpose => shape => (n_range_bins, n_chirps_frame)
        fft1d = range_fft.T

        # 6) Doppler window
        fft1d_win = fft1d * doppler_window

        # 7) Zero-pad in doppler dimension
        zp_doppler = np.pad(fft1d_win, ((0,0),(0,n_chirps_frame)), 'constant')
        # => shape (n_range_bins, 2*n_chirps_frame)

        # 8) Doppler FFT
        fft2d = np.fft.fft(zp_doppler, axis=1) / n_chirps_frame

        # 9) Exponential average for static clutter removal
        static_avg[:,:,rx_idx] = args.mti_alpha*static_avg[:,:,rx_idx] + (1.0-args.mti_alpha)*fft2d
        fft2d_mti = fft2d - static_avg[:,:,rx_idx]

        # 10) Shift doppler axis => zero velocity at center
        dopplerfft = np.fft.fftshift(fft2d_mti, axes=(1,))

        rd_sum += dopplerfft

    # Now rd_sum is the combined Range-Doppler for either just RX0 or sum of all 3 antennas
    rd_abs = np.abs(rd_sum)

    # -----------------------------------------------------------------------
    # 4) Plot Range-Doppler heatmap
    # -----------------------------------------------------------------------
    # We'll put velocity on the horizontal axis from -max_speed..+max_speed
    # and range on the vertical axis from 0..(n_range_bins * range_res)
    # or from 0..some fraction.  We'll do:
    # x axis => velocity (m/s)
    # y axis => range (m)

    # velocity axis
    vel_axis = np.linspace(-max_speed, max_speed, n_doppler_bins, endpoint=False)
    # range axis (we only keep first half of freq => ~ n_range_bins bins)
    # each bin => ~ range_res_m, so from 0..(n_range_bins-1)*range_res_m
    rng_axis = np.linspace(0, (n_range_bins-1)*range_res_m, n_range_bins)

    rd_dB = 20.0 * np.log10(rd_abs + 1e-6)

    plt.figure(figsize=(8,6))
    plt.title(f"Range-Doppler Map (Frame {args.frame_index}) - {session_name}")
    # extent => [xmin, xmax, ymin, ymax]
    # velocity on horizontal => [vel_axis[0], vel_axis[-1]]
    # range on vertical => [rng_axis[0], rng_axis[-1]]
    # we want 'origin=lower' so 0 range is at bottom
    plt.imshow(
        rd_dB,
        extent=[vel_axis[0], vel_axis[-1], rng_axis[0], rng_axis[-1]],
        aspect='auto',
        origin='lower',
        cmap='jet'
    )
    plt.colorbar(label="Amplitude (dB)")
    plt.xlabel("Velocity (m/s)")
    plt.ylabel("Range (m)")
    plt.tight_layout()
    plt.savefig(args.outfile, dpi=150)
    plt.show()

if __name__ == "__main__":
    main()
