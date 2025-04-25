#!/usr/bin/env python3

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats

################################################################################
# Algorithm 1: Find optimal range-bin interval
################################################################################

def compute_range_fft_all_chirps(iq_data):
    """
    Compute the range FFT for each chirp (summing across RX),
    returning an array shape (M, N).
    - iq_data: shape (num_rx, M, N)
    """
    num_rx, M, N = iq_data.shape
    # sum across RX
    sum_rx = np.sum(iq_data, axis=0)  # shape => (M, N)

    # Window for range dimension
    window = signal.blackmanharris(N)

    range_fft_mat = np.zeros((M, N), dtype=np.complex64)
    for i in range(M):
        chirp_samples = sum_rx[i,:]
        chirp_win     = chirp_samples * window
        fft_chirp     = np.fft.fft(chirp_win, n=N)
        range_fft_mat[i,:] = fft_chirp
    return range_fft_mat


def find_pmax_indices(range_fft_mat):
    """
    For each chirp i, find the index of maximum amplitude in range FFT.
    Return an array pmax of length M with those indices.
    """
    M, N = range_fft_mat.shape
    pmax = np.zeros(M, dtype=int)
    for i in range(M):
        mag_i = np.abs(range_fft_mat[i,:])
        pmax[i] = np.argmax(mag_i)
    return pmax


def find_idx_max_repeated(pmax):
    """
    The paper suggests using the 'most repeated index' among pmax[i].
    This is effectively the mode. Return that index.
    """
    # we can do a simple stats.mode or a histogram
    mode_val = stats.mode(pmax, keepdims=True)[0][0]
    return int(mode_val)


def define_intervals_around_idx(idx_max, Q_list, N):
    """
    Create a library of intervals around idx_max. For example, Q_list might be [2,3,4,5].
    Each interval is [idx_max - q : idx_max + q], clipped within [0..N-1].
    Return a list of (start_bin, end_bin) tuples.
    """
    intervals = []
    for q in Q_list:
        start_bin = max(0, idx_max - q)
        end_bin   = min(N, idx_max + q + 1)
        intervals.append((start_bin, end_bin))
    return intervals


def stft_for_range_interval(range_fft_mat, interval, slow_fs, nperseg=64, noverlap=32):
    """
    Sum the range FFT across [interval[0]..interval[1]] bins, then do STFT along chirp axis.
    Return the STFT (complex) array shape => (freq_bins, time_segments).
    Also return the time/freq axes from signal.stft.
    """
    (start_bin, end_bin) = interval
    # sum across those bins
    M, N = range_fft_mat.shape
    selected_slice = range_fft_mat[:, start_bin:end_bin]  # shape => (M, width)
    sum_across_bins = np.sum(selected_slice, axis=1)      # shape => (M,)

    # Do STFT in slow-time dimension
    f_stft, t_stft, Zxx = signal.stft(
        sum_across_bins,
        fs=slow_fs,
        window='hanning',
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nperseg*2,   # zero-pad
        return_onesided=False
    )
    return f_stft, t_stft, Zxx


def compute_average_entropy(Zxx):
    """
    Compute the average information entropy across time for the STFT.
    - Zxx: shape (freq_bins, time_segments), complex
    Steps:
      1) For each time segment, convert the magnitude squared to a probability distribution p(omega).
      2) Compute H_t = - sum_{omega} p(omega)*ln(p(omega)).
      3) Average H_t over all time segments => H_avg.
    """
    mag_sq = np.abs(Zxx)**2
    freq_bins, time_segments = mag_sq.shape

    H_values = np.zeros(time_segments, dtype=float)
    for t in range(time_segments):
        # get magnitude squared at time t
        col = mag_sq[:, t]
        denom = np.sum(col)
        if denom < 1e-12:
            # no signal => entropy = 0
            H_values[t] = 0
        else:
            p = col / denom
            # compute - sum p_i ln p_i
            # add a tiny offset to avoid log(0)
            p[p<1e-12] = 1e-12
            H_values[t] = -np.sum(p * np.log(p))

    H_avg = np.mean(H_values)
    return H_avg


def algorithm1_optimal_range(iq_data, slow_fs, Q_list, nperseg=64, noverlap=32):
    """
    Implement Algorithm 1 from the paper.
    1) Range FFT all chirps
    2) pmax[i] = index of max amplitude
    3) idx_max = mode of pmax
    4) define intervals r_q in Q_list
    5) for each interval, do STFT -> compute average entropy -> pick min

    Return:
      - best_interval (start_bin, end_bin)
      - range_fft_mat
      - idx_max
    """
    range_fft_mat = compute_range_fft_all_chirps(iq_data)
    M, N = range_fft_mat.shape

    # step 2 & 3
    pmax = find_pmax_indices(range_fft_mat)
    idx_max = find_idx_max_repeated(pmax)

    # define intervals
    intervals = define_intervals_around_idx(idx_max, Q_list, N)

    best_interval = None
    best_entropy = 1e9
    all_entropies = []

    for interval in intervals:
        f_stft, t_stft, Zxx = stft_for_range_interval(range_fft_mat, interval,
                                                      slow_fs, nperseg, noverlap)
        H_avg = compute_average_entropy(Zxx)
        all_entropies.append(H_avg)
        if H_avg < best_entropy:
            best_entropy = H_avg
            best_interval = interval

    return best_interval, range_fft_mat, idx_max, all_entropies, intervals


################################################################################
# Algorithm 2: Denoising with cut-threshold
################################################################################

def algorithm2_denoise(Zxx, Th_factor=3.0):
    """
    Implement the 'cut-threshold' approach from the paper.
    For each time slice, compute avg(F_j(l,omega)), define ETh = Th_factor * avg(...).
    Then create mask T_j(l,omega)=1 if |F_j(l,omega)|>ETh, else 0.
    Multiply STFT by that mask => denoised STFT.

    Return the denoised Zxx.
    """
    # Zxx: shape (freq_bins, time_segments)
    freq_bins, time_segments = Zxx.shape
    Zxx_denoise = np.zeros_like(Zxx, dtype=complex)

    for t in range(time_segments):
        column = Zxx[:,t]
        mag_col = np.abs(column)
        avg_col = np.mean(mag_col)
        ETh = Th_factor * avg_col

        mask = (mag_col > ETh).astype(float)
        # multiply
        Zxx_denoise[:,t] = column * mask

    return Zxx_denoise


################################################################################
# Main Script
################################################################################

def main():
    parser = argparse.ArgumentParser(description="Replicate paper's Algorithm 1 (optimal range) + Algorithm 2 (denoising).")
    parser.add_argument('-i','--infile', type=str, default='radar_data.pkl',
                        help="Input .pkl file from record_radar.py with frames, etc.")
    parser.add_argument('-o','--outfile', type=str, default='final_spectrogram.png',
                        help="Output image for final denoised spectrogram.")
    parser.add_argument('--frame_index', type=int, default=0,
                        help="Which frame to process. (Or adapt code to do entire multi-frame approach).")
    parser.add_argument('--frame_rate', type=float, default=10.0,
                        help="Radar frame rate in Hz if we do single-frame slow-time.")
    parser.add_argument('--Q_list', type=str, default='2,3,4',
                        help="Comma-separated half-widths around idx_max to test, e.g. '2,3,4'")
    parser.add_argument('--nperseg', type=int, default=64,
                        help="STFT window size in slow-time dimension.")
    parser.add_argument('--noverlap', type=int, default=32,
                        help="STFT overlap.")
    parser.add_argument('--cut_threshold_factor', type=float, default=3.0,
                        help="Factor Th for ETh = Th*avg(...) in Algorithm 2.")
    args = parser.parse_args()

    # 1) Load data
    with open(args.infile, 'rb') as f:
        data = pickle.load(f)

    frames = data["frames"]
    if args.frame_index<0 or args.frame_index>=len(frames):
        raise ValueError("frame_index out of range.")
    frame_data = frames[args.frame_index]
    iq_data = frame_data["iq_data"]  # shape => (num_rx, M, N)
    num_rx, M, N = iq_data.shape

    # 2) We'll treat this single frame as if we have M chirps => slow-time Fs = M * frame_rate?
    #   But in many setups, each frame is the entire CPI. For a single frame, let's approximate slow_fs = M / frame_duration?
    #   If each frame is 0.1 s, then slow_fs = M/0.1. We'll do the simpler approach:
    slow_fs = M * args.frame_rate

    print(f"Frame shape=({num_rx},{M},{N}), slow_fs={slow_fs:.2f} Hz")

    # parse Q_list
    Q_list = [int(x) for x in args.Q_list.split(',')]
    print(f"Algorithm1: testing intervals half-width = {Q_list}")

    # 3) Algorithm 1: find optimal range interval
    best_interval, range_fft_mat, idx_max, all_entropies, intervals = \
        algorithm1_optimal_range(iq_data, slow_fs, Q_list,
                                 nperseg=args.nperseg, noverlap=args.noverlap)

    print(f"Most repeated bin idx_max = {idx_max}")
    print("Intervals tested:", intervals)
    print("Entropies:", all_entropies)
    print(f"Chosen interval: {best_interval}")

    # For illustration, let's plot the amplitude of range FFT for each chirp's max
    pmax = find_pmax_indices(range_fft_mat)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.title("pmax index per chirp")
    plt.plot(pmax, 'b.')
    plt.axhline(idx_max, color='r', linestyle='--', label=f"idx_max={idx_max}")
    plt.xlabel("Chirp index")
    plt.ylabel("Range-bin of max amplitude")
    plt.legend()

    # Also show the intervals & entropies
    plt.subplot(1,2,2)
    halfwidths = [intervals[i][1]-intervals[i][0] for i in range(len(intervals))]
    plt.plot(halfwidths, all_entropies, 'o-')
    plt.title("Entropy vs. Interval Width")
    plt.xlabel("Interval width (# bins)")
    plt.ylabel("Avg Entropy")
    plt.tight_layout()
    plt.show()

    # 4) Now do final STFT with the chosen interval
    f_stft, t_stft, Zxx_final = stft_for_range_interval(range_fft_mat, best_interval,
                                                        slow_fs, nperseg=args.nperseg,
                                                        noverlap=args.noverlap)
    # 5) Algorithm 2: Denoise
    Zxx_denoise = algorithm2_denoise(Zxx_final, Th_factor=args.cut_threshold_factor)

    # 6) Plot final results: raw STFT vs. denoised
    # reorder freq so negative is at bottom
    freq_order = np.argsort(f_stft)
    f_stft_sorted = f_stft[freq_order]
    Zxx_raw_mag_db = 20*np.log10(np.abs(Zxx_final[freq_order,:])+1e-6)
    Zxx_dn_mag_db  = 20*np.log10(np.abs(Zxx_denoise[freq_order,:])+1e-6)

    time_extent = [t_stft[0], t_stft[-1]]
    freq_extent = [f_stft_sorted[0], f_stft_sorted[-1]]

    plt.figure(figsize=(12,5))
    plt.suptitle(f"Algorithm1+2 final result (Frame={args.frame_index}, interval={best_interval})")

    plt.subplot(1,2,1)
    plt.title("Raw STFT (chosen range interval)")
    plt.imshow(Zxx_raw_mag_db, extent=[time_extent[0], time_extent[1],
                                       freq_extent[0], freq_extent[1]],
               aspect='auto', origin='lower', cmap='jet')
    plt.colorbar(label="Amplitude (dB)")
    plt.xlabel("Slow-time (s)")
    plt.ylabel("Frequency (Hz)")

    plt.subplot(1,2,2)
    plt.title(f"Denoised STFT (Th_factor={args.cut_threshold_factor})")
    plt.imshow(Zxx_dn_mag_db, extent=[time_extent[0], time_extent[1],
                                      freq_extent[0], freq_extent[1]],
               aspect='auto', origin='lower', cmap='jet')
    plt.colorbar(label="Amplitude (dB)")
    plt.xlabel("Slow-time (s)")

    plt.tight_layout()
    plt.savefig(args.outfile, dpi=150)
    plt.show()

if __name__ == "__main__":
    main()
