#!/usr/bin/env python3
"""
record_radar_with_beep.py

Script to record raw radar frames from the Infineon BGT60TR13C device.
This version does not require external GPIO or buzzer hardware.
Instead, it uses a WAV file ("beep.wav") played via Pygame for audible signals.
Procedure:
  1. Wait for user input (press Enter) to start.
  2. Wait for a 5-second countdown.
  3. Play a beep sound to indicate start.
  4. Capture radar frames for a specified duration.
  5. Play a beep sound to indicate the end of capture.
  6. Save the frames into a pickle file.

Usage:
    python3 record_radar_with_beep.py -n 100 -f 10 -o radar_data.pkl
"""

import argparse
import pickle
import os
import time
from datetime import datetime
from collections import namedtuple

import numpy as np
from ifxRadarSDK import *
from radar_cfg import common_radar_device_config  # Your helper function

import pygame  # For playing beep sound

def beep():
    """
    Play a beep sound using pygame.
    Ensure "beep.wav" is present in the same directory as this script.
    """
    try:
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        pygame.mixer.music.load("beep-01a.wav")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
    except Exception as e:
        print("Error playing beep:", e)

def main():
    parser = argparse.ArgumentParser(description="Record radar frames with countdown and beep signals.")
    parser.add_argument('-n', '--nframes', type=int, default=100,
                        help="Number of frames to capture (default=100).")
    parser.add_argument('-f', '--frate', type=int, default=10,
                        help="Frame rate in Hz (default=10).")
    parser.add_argument('-o', '--outfile', type=str, default="radar_data.pkl",
                        help="Output pickle filename (default=radar_data.pkl).")
    parser.add_argument('-r', '--radarname', type=str, default="MyRadarSession",
                        help="Session/radar name to embed in metadata.")
    parser.add_argument('--countdown', type=int, default=5,
                        help="Countdown in seconds before capture starts (default=5).")
    args = parser.parse_args()

    # Wait for user input (simulate button press)
    input("Press Enter to start the recording process...")

    # Countdown before starting
    print("Starting countdown:")
    for sec in range(args.countdown, 0, -1):
        print(f"  {sec}...")
        time.sleep(1)

    # Beep to signal start
    print("Beep! Start recording.")
    beep()

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
    device.start_acquisition()
    for i in range(args.nframes):
        try:
            device.get_next_frame(frame)
        except RadarSDKFifoOverflowError:
            print("FIFO Overflow! Stopping acquisition.")
            break

        # Collect data for each RX (each returns a matrix: (num_chirps, chirp_samples))
        frame_data = []
        for rx_idx in range(num_rx):
            mat = frame.get_mat_from_antenna(rx_idx)
            mat = np.array(mat, dtype=np.complex64)
            frame_data.append(mat)
        frame_data = np.stack(frame_data, axis=0)  # shape: (num_rx, num_chirps, chirp_samples)

        # Build dictionary for the frame
        frame_dict = {
            "timestamp": datetime.now().isoformat(),
            "frame_index": i,
            "iq_data": frame_data
        }
        all_frames.append(frame_dict)
        print(f"Captured frame {i+1}/{args.nframes}")

    # Stop acquisition and close device
    device.stop_acquisition()
    device.close()
    print("Recording completed.")

    # Beep to signal end
    print("Beep! End of recording.")
    beep()

    # -----------------------------------------------------------
    # Store all frames into a pickle file
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
