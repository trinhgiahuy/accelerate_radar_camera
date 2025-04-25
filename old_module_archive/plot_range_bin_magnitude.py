#!/usr/bin/env python3

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def main():
    parser = argparse.ArgumentParser(
        description="Plot average Range FFT magnitude vs. bin index to find the target's range bin."
    )
    parser.add_argument('-i','--infile', type=str, default='radar_data.pkl',
                        help="Input .pkl file from record_radar.py")
    parser.add_argument('--frame_index', type=int, default=0,
                        help="Which frame to analyze (default=0).")
    parser.add_argument('--num_chirps_avg', type=int, default=10,
                        help="How many chirps to average for stable range profile (default=10).")
    parser.add_argument('--range_resolution_m', type=float, default=0.15,
                        help="Range resolution in meters per bin (e.g. 0.15 m).")
    args = parser.parse_args()

    # 1) Load data
    with open(args.infile, 'rb') as f:
        data = pickle.load(f)

    frames = data["frames"]
    if args.frame_index < 0 or args.frame_index >= len(frames):
        raise ValueError("frame_index out of range.")

    # Radar params might store range_resolution, but we allow overriding
    # or specifying via --range_resolution_m
    range_res = args.range_resolution_m

    # 2) Extract the chosen frame
    frame_data = frames[args.frame_index]  # dict with "iq_data"
    iq_data = frame_data["iq_data"]        # shape => (num_rx, M, N)
    num_rx, M, N = iq_data.shape
    print(f"Frame {args.frame_index}: shape=({num_rx},{M},{N}), range_res={range_res} m/bin")

    # 3) Sum across RX
    sum_rx = np.sum(iq_data, axis=0)  # shape => (M, N)

    # 4) Decide how many chirps to average
    n_chirps_avg = min(args.num_chirps_avg, M)
    print(f"Averaging the first {n_chirps_avg} chirps out of {M} total.")

    # 5) Range FFT for those chirps
    #    We'll do a Blackman window in the range dimension
    range_window = signal.blackmanharris(N)

    # Accumulate magnitudes in an array
    accum_magnitude = np.zeros(N, dtype=np.float32)

    for c in range(n_chirps_avg):
        chirp_samples = sum_rx[c,:]
        chirp_win = chirp_samples * range_window
        fft_chirp = np.fft.fft(chirp_win, n=N)
        mag_chirp = np.abs(fft_chirp)
        accum_magnitude += mag_chirp

    # average
    avg_magnitude = accum_magnitude / n_chirps_avg

    # 6) Plot magnitude vs. bin index
    bin_axis = np.arange(N)  # 0..N-1
    dist_axis = bin_axis * range_res  # approximate distance in meters

    plt.figure(figsize=(8,5))
    plt.subplot(2,1,1)
    plt.plot(bin_axis, avg_magnitude, '-b')
    plt.title("Average Range FFT Magnitude (first {} chirps)".format(n_chirps_avg))
    plt.xlabel("Range Bin Index")
    plt.ylabel("Magnitude (linear)")

    # 7) Also plot vs. distance
    plt.subplot(2,1,2)
    plt.plot(dist_axis, avg_magnitude, '-r')
    plt.xlabel("Distance (m) ~ (bin_index * range_res)")
    plt.ylabel("Magnitude (linear)")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
