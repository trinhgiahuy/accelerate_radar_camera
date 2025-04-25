#!/usr/bin/env python3
"""
record_radar.py

Script to record raw radar frames from the Infineon BGT60TR13C device
using the Radar SDK. Stores all frames to a single pickle file for later
offline processing.

Usage:
    python record_radar.py -n 100 -f 10 -o data.pkl

This will capture 100 frames at 10 frames/second and store them in "data.pkl".
"""

import argparse
import pickle
import os
from datetime import datetime
from collections import namedtuple

import numpy as np
from ifxRadarSDK import *
from radar_cfg import common_radar_device_config  # Your helper function

def main():
    parser = argparse.ArgumentParser(description="Record radar frames to a file.")
    parser.add_argument('-n', '--nframes', type=int, default=100,
                        help="Number of frames to capture (default=100).")
    parser.add_argument('-f', '--frate', type=int, default=10,
                        help="Frame rate in Hz (default=10).")
    parser.add_argument('-o', '--outfile', type=str, default="radar_data.pkl",
                        help="Output pickle filename (default=radar_data.pkl).")
    parser.add_argument('-r', '--radarname', type=str, default="MyRadarSession",
                        help="Session/radar name to embed in metadata.")
    args = parser.parse_args()

    # -----------------------------------------------------------
    # Define radar parameters
    # -----------------------------------------------------------
    ParamType = namedtuple('ParamType', [
        'range_resolution_m',
        'num_samples_per_chirp',
        'max_speed_m_s',
        'sample_rate_Hz',
        'frame_repetition_time_s'
    ])

    # Example configuration
    # (You can tweak range_resolution_m, num_samples_per_chirp, etc.)
    params = ParamType(
        range_resolution_m=0.075,
        num_samples_per_chirp=64,
        max_speed_m_s=3,          # ±3 m/s
        sample_rate_Hz=1_000_000, # 1 MHz
        frame_repetition_time_s=1.0 / args.frate
    )

    # -----------------------------------------------------------
    # Open the radar device and create a frame object
    # -----------------------------------------------------------
    device, metrics = common_radar_device_config(params)
    frame = device.create_frame_from_device_handle()

    num_rx         = frame.get_num_rx()
    num_chirps     = metrics.num_chirps_per_frame
    chirp_samples  = metrics.num_samples_per_chirp
    max_range_m    = metrics.max_range_m

    print("==============================================")
    print(f"Radar session name:   {args.radarname}")
    print(f"Output file:          {args.outfile}")
    print("Radar configuration:")
    print(f"  # RX antennas:      {num_rx}")
    print(f"  # chirps/frame:     {num_chirps}")
    print(f"  # samples/chirp:    {chirp_samples}")
    print(f"  max_range_m:        {max_range_m:.2f} m")
    print(f"  max_speed_m_s:      {params.max_speed_m_s:.2f} m/s")
    print("==============================================")

    # Prepare list to store frames
    all_frames = []

    print(f"Capturing {args.nframes} frames at {args.frate} FPS ...")
    for i in range(args.nframes):
        try:
            device.get_next_frame(frame)
        except RadarSDKFifoOverflowError:
            print("FIFO Overflow! Stopping acquisition.")
            break

        # For each frame, collect data from each RX
        # We will store them in shape [num_rx, num_chirps, chirp_samples]
        frame_data = []
        for rx_idx in range(num_rx):
            mat = frame.get_mat_from_antenna(rx_idx)  # shape: (num_chirps, chirp_samples)
            # Convert to np.array if not already
            mat = np.array(mat, dtype=np.complex64)
            frame_data.append(mat)

        # Stack into one array: shape (num_rx, num_chirps, chirp_samples)
        frame_data = np.stack(frame_data, axis=0)

        # Build a dictionary for this frame
        frame_dict = {
            "timestamp": datetime.now().isoformat(),
            "frame_index": i,
            "iq_data": frame_data  # shape: (num_rx, num_chirps, chirp_samples)
        }

        all_frames.append(frame_dict)
        print(f"Captured frame {i+1}/{args.nframes}")

    # Stop device
    device.stop_acquisition()
    device.close()
    print("Acquisition stopped.")

    # -----------------------------------------------------------
    # Store all frames into one pickle file
    # -----------------------------------------------------------
    session_info = {
        "session_name": args.radarname,
        "radar_params": params._asdict(),
        "frame_rate": args.frate,
        "num_frames": len(all_frames),
        "frames": all_frames
    }

    with open(args.outfile, 'wb') as f:
        pickle.dump(session_info, f)

    print(f"Saved {len(all_frames)} frames to {args.outfile}")

if __name__ == "__main__":
    main()
