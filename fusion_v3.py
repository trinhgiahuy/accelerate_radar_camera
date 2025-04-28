#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# fusion_v3.py - Simplified Radar-Camera Data Recorder

import argparse
import sys
import time
import threading
import queue
import os
import json
import logging
from datetime import datetime
from functools import lru_cache
import errno

# --- Matplotlib (Optional Display, needed only if ENABLE_LIVE_VIEW=True) ---
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# --- Radar SDK Imports ---
try:
    from ifxradarsdk import get_version_full
    from ifxradarsdk.fmcw import DeviceFmcw
    from ifxradarsdk.fmcw.types import FmcwSimpleSequenceConfig, FmcwMetrics#, ChirpConfig, FrameConfig, LoopConfig # Import necessary types
    # from ifxradarsdk.fmcw.errors import ??? # Import specific errors if needed
except ImportError:
    print("ERROR: Failed to import ifxradarsdk. Please ensure it is installed.")
    sys.exit(1)

# --- Doppler Algo Helper (Optional Display) ---
try:
    from helpers.DopplerAlgo import DopplerAlgo
except ImportError:
    DopplerAlgo = None

# --- Picamera2 & System Imports ---
try:
    import cv2
    import psutil # Keep for optional resource monitoring
    from picamera2 import Picamera2
    from picamera2.controls import Controls
    from picamera2.devices import IMX500
    from picamera2.devices.imx500 import (NetworkIntrinsics,
                                           postprocess_nanodet_detection)
    from picamera2.devices.imx500.postprocess import scale_boxes
except ImportError:
    print("ERROR: Failed to import Picamera2/OpenCV/psutil. Please ensure they are installed.")
    sys.exit(1)
import numpy as np

# =================================================
# === Configuration Constants ===
# =================================================

# --- General Settings ---
LOG_LEVEL = "INFO" # DEBUG, INFO, WARNING, ERROR, CRITICAL
RECORDING_BASE_DIR = "./recordings_v3" # Parent directory for session folders
SESSION_NAME_PREFIX = "multi_person_scenario" # Prefix for session folder name

# --- Recording Settings ---
RECORDING_QUEUE_SIZE = 50 # Max items buffered before dropping
METADATA_FLUSH_INTERVAL = 50 # Write metadata to disk every N items

# --- Radar Configuration (Based on image_3eb754.png) ---
#RADAR_FRAME_RATE_HZ = 10.0 # From 100 ms frame time
RADAR_FRAME_RATE_HZ = 10.0 # From 100 ms frame time
RADAR_SAMPLE_RATE_HZ = 1_000_000 # 2 MHz
RADAR_NUM_SAMPLES_PER_CHIRP = 128
RADAR_NUM_CHIRPS_PER_FRAME = 64
# RADAR_CHIRP_TO_CHIRP_TIME_US = 391 # Not directly configured, result of other params
RADAR_BANDWIDTH_HZ = 460_000_000 # Not directly configured, result of other params
RADAR_CENTER_FREQ_HZ = 60_750_000_000 # Assumed common value, adjust if known
RADAR_TX_POWER = 31 # Max power level
RADAR_IF_GAIN_DB = 33
RADAR_RX_MASK = "auto" # 'auto' or integer mask (e.g., 7 for antennas 0,1,2)
RADAR_TX_MASK = 1
LIGHT_SPEED = 299_792_458

RADAR_RANGE_RESOLUTION_M = LIGHT_SPEED / (2 * RADAR_BANDWIDTH_HZ)
RADAR_MAX_RANGE_M = (RADAR_NUM_SAMPLES_PER_CHIRP * LIGHT_SPEED) / (4 * RADAR_BANDWIDTH_HZ)
RADAR_MAX_SPEED_M_S = 3.0

# Calculate derived radar parameters (for info, might not be needed for direct config)
RADAR_CHIRP_TIME_S = RADAR_NUM_SAMPLES_PER_CHIRP / RADAR_SAMPLE_RATE_HZ
# Theoretical Range Res = c / (2 * BW). BW depends on ramp time & freq range.
# If we assume the 460MHz BW is correct for the 64us chirp time:
# Ramp speed = 460e6 Hz / 64e-6 s = 7.1875e12 Hz/s
# If we assume 1GHz bandwidth (common for 60GHz):
# Max Range Res ~= c / (2 * 1e9) = 0.15m
# Max Speed = wavelength / (4 * chirp_time_to_chirp_time)
# wavelength = c / center_freq ~= 3e8 / 60.75e9 ~= 0.0049m
# Speed Res = wavelength / (2 * frame_time * num_chirps) - This seems wrong, check SDK calc.
# The SDK's `sequence_from_metrics` is usually easier, but we'll try direct config here.

# --- Camera Configuration ---
CAMERA_RESOLUTION = (640, 480) # [Width, Height]
CAMERA_FORMAT = "RGB888" # Capture format
CAMERA_BUFFER_COUNT = 6
CAMERA_CONTROLS = {
    "FrameRate": 15.0, # Target FPS (best effort)
    # "AeEnable": True,
    # "AwbEnable": True,
}

# --- Camera AI Settings ---
AI_MODEL_PATH = "./imx500-models-backup/imx500_network_yolov8n_pp.rpk"
AI_THRESHOLD = 0.55
AI_IOU = 0.65
AI_MAX_DETECTIONS = 10 # Max persons expected + buffer (e.g., up to 6 people)

# =================================================
# Logging Setup
# =================================================
log_level_map = {
    "DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING,
    "ERROR": logging.ERROR, "CRITICAL": logging.CRITICAL
}
log_level = log_level_map.get(LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(level=log_level,
                    format='%(asctime)s - %(threadName)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# =================================================
# Shared Resources & Globals
# =================================================
# Queues (initialized later based on config)
radar_display_queue = queue.Queue(maxsize=5) #if ENABLE_LIVE_VIEW else None
camera_display_queue = queue.Queue(maxsize=5) #if ENABLE_LIVE_VIEW else None
recording_queue = queue.Queue(maxsize=RECORDING_QUEUE_SIZE)

# Control Events
stop_event = threading.Event()
recording_active = threading.Event() # Set when recording is running

# Global objects/state (initialized in main)
imx500 = None
intrinsics = None
picam2 = None
camera_intrinsics_calib = None
camera_sensor_format = None
radar_config_used = {} # Store the actual config applied
camera_config_used = {}
last_camera_results = []
# =================================================
# RadarDraw Class (Optional Display)
# =================================================
class RadarDraw:
    """Draws Radar Range-Doppler Map for Live View. (Conditional)"""
    # --- Use the implementation from fusion_v4.py ---
    # --- Ensure it checks ENABLE_LIVE_VIEW and uses logging ---
    # --- Omitted for brevity, assume it's the v4 version ---
    # <<< PASTE RadarDraw class code from fusion_v4.py HERE >>>
    # (Ensure it handles `plt=None` if matplotlib is not installed)
    # --- Matplotlib (Radar) Imports ---
    # (Code is identical to fusion_v3.py - omitted for brevity, assume it's here)
    # Key methods: __init__, setup_plot, _draw_first_time, _draw_next_time, draw, handle_close, close
    # Ensure logging is used instead of print if desired within the class.
    """Modified Draw class for radar visualization using Matplotlib."""
    def __init__(self, max_speed_m_s, max_range_m, num_ant, stop_signal_event, display_is_enabled):
        self._h = []
        self._img_h = [] # Separate list for image handles if needed
        self._cbar_h = None # Handle for colorbar
        self._max_speed_m_s = max_speed_m_s # Note: Need to calculate this based on config
        self._max_range_m = max_range_m # Note: Need to calculate this based on config
        self._num_ant = num_ant
        self._fig = None
        self._ax = None
        self._is_window_open = False
        self._stop_event = stop_signal_event # Reference to the main stop event
        self._vmin = -60 # Default min dB for plotting
        self._vmax = 0 # Default max dB for plotting
        self.display_is_enabled = display_is_enabled
    def setup_plot(self):
        """Sets up the Matplotlib figure and axes. Call this from the main thread."""
        if not plt or not self.display_is_enabled:
             # logging.info("Matplotlib display disabled or library not available.")
             return # Skip if display disabled or matplotlib not installed
        try:
            plt.ion()
            # Adjust figsize based on antenna count
            figsize_w = max(5, self._num_ant * 2.5 + 1.5)
            figsize_h = 3.5
            self._fig, self._ax = plt.subplots(nrows=1, ncols=self._num_ant, figsize=(figsize_w, figsize_h), squeeze=False) # Always returns 2D array
            self._ax = self._ax.flatten() # Flatten to 1D array for easier indexing

            self._fig.canvas.manager.set_window_title("Range-Doppler Map (Live)")
            # Connect close event to the main stop signal
            self._fig.canvas.mpl_connect('close_event', self.handle_close)
            self._is_window_open = True
            logging.info("Radar plot initialized.")
        except Exception as e:
             logging.error(f"Failed to initialize radar plot: {e}")
             self._is_window_open = False

    def _update_plot_data(self, data_all_antennas):
        """Helper to update plot data and color limits."""
        if not self._is_window_open or not plt or not self._fig or not plt.fignum_exists(self._fig.number): return False

        valid_data = [d for d in data_all_antennas if d is not None and d.size > 0]
        if not valid_data:
            return False # No data to plot

        min_val = min(np.min(d) for d in valid_data)
        max_val = max(np.max(d) for d in valid_data)
        self._vmin = max(-80, min_val) # Example floor
        self._vmax = max_val

        if len(self._img_h) != self._num_ant: return False

        for i_ant in range(self._num_ant):
             if i_ant < len(self._img_h) and self._img_h[i_ant] is not None: # Check handle exists
                data = data_all_antennas[i_ant]
                if data is not None:
                    self._img_h[i_ant].set_data(data)
                    self._img_h[i_ant].set_clim(vmin=self._vmin, vmax=self._vmax)

        if self._cbar_h:
             self._cbar_h.mappable.set_clim(vmin=self._vmin, vmax=self._vmax)

        return True

    def draw(self, data_all_antennas):
        """Draws the plot. Called from the main thread."""
        if not self._is_window_open or not plt or not self._fig or not plt.fignum_exists(self._fig.number) or not self.display_is_enabled:
            return

        try:
            # Initialize plot elements on first valid draw
            if not self._img_h and any(d is not None for d in data_all_antennas):
                logging.debug("First draw initialization for radar plot.")
                self._img_h = [None] * self._num_ant
                valid_data = [d for d in data_all_antennas if d is not None and d.size > 0]
                if valid_data:
                     self._vmin = max(-80, min(np.min(d) for d in valid_data))
                     self._vmax = max(np.max(d) for d in valid_data)

                for i_ant in range(self._num_ant):
                     if self._ax[i_ant]: # Check if axis exists
                         data = data_all_antennas[i_ant]
                         if data is not None:
                             # *** Calculate max_range and max_speed if not pre-calculated ***
                             # These depend on final radar config, ideally pass them in
                             # Placeholder values for now
                             temp_max_range = 5.0
                             temp_max_speed = 3.0
                             im = self._ax[i_ant].imshow(
                                 data, vmin=self._vmin, vmax=self._vmax, cmap='hot',
                                 extent=(-temp_max_speed, temp_max_speed, 0, temp_max_range),
                                 origin='lower', interpolation='nearest', aspect='auto'
                             )
                             self._img_h[i_ant] = im
                             self._ax[i_ant].set_xlabel("Velocity (m/s)")
                             self._ax[i_ant].set_ylabel("Distance (m)")
                             self._ax[i_ant].set_title(f"Antenna #{i_ant}")
                         else:
                              self._ax[i_ant].set_title(f"Antenna #{i_ant} (No data)")
                              self._ax[i_ant].set_xticks([])
                              self._ax[i_ant].set_yticks([])

                valid_img_handles = [h for h in self._img_h if h is not None]
                if valid_img_handles and not self._cbar_h and self._fig:
                    try:
                        self._fig.subplots_adjust(right=0.85)
                        cbar_ax = self._fig.add_axes([0.88, 0.15, 0.03, 0.7])
                        self._cbar_h = self._fig.colorbar(valid_img_handles[0], cax=cbar_ax)
                        self._cbar_h.ax.set_ylabel("Magnitude (dB)")
                        self._fig.tight_layout(rect=[0, 0, 0.85, 1])
                    except Exception as layout_err:
                         logging.warning(f"Could not apply tight_layout after colorbar: {layout_err}")

            # Update existing plot data
            elif self._img_h:
                 self._update_plot_data(data_all_antennas)

            # Redraw canvas using plt.pause
            plt.pause(0.001)

        except Exception as e:
            logging.error(f"Error during radar plot drawing: {e}", exc_info=True)

    def handle_close(self, event=None):
        """Handles the close event from Matplotlib."""
        if self._is_window_open:
             logging.info("Radar window close requested.")
             self._is_window_open = False
             self._stop_event.set()

    def close(self):
        """Closes the plot resources."""
        if self._is_window_open:
            self._is_window_open = False
            logging.info("Closing radar plot window.")
            if plt and self._fig:
                 plt.close(self._fig)
        if plt:
             plt.close('all')

# =================================================
# Detection Class & AI Helpers
# =================================================
class Detection:
    """Represents a single detection with JSON serialization."""
    def __init__(self, coords, category, conf, metadata, cam_instance, net_intrinsics):
        self.category = category
        self.conf = conf
        #try:
        #    # Ensure net_intrinsics is valid before calling
        #    if net_intrinsics and hasattr(net_intrinsics_obj, 'convert_inference_coords'):
        #         #self.box_xywh = net_intrinsics_obj.convert_inference_coords(coords, metadata, cam_instance)
        self.box_xywh = imx500.convert_inference_coords(coords, metadata, cam_instance)
        #    else:
        #         logging.error("NetworkIntrinsics object invalid in Detection init.")
        #         self.box_xywh = [0,0,0,0] # Default invalid box
        #except Exception as e:
        #    logging.error(f"Error converting inference coords: {e}")
        #self.box_xywh = [0,0,0,0]

    def to_dict(self):
         """Converts detection to a JSON-serializable dictionary."""
         return {
             "category_id": int(self.category),
             "confidence": float(self.conf),
             "box_xywh": [int(c) for c in self.box_xywh]
         }

@lru_cache
def get_labels(net_intrinsics):
    """Gets labels from network intrinsics, handling caching and optional filtering."""
    labels = net_intrinsics.labels
    if net_intrinsics.ignore_dash_labels:
        labels = [label for label in labels if label and label != "-"]
    return labels

def parse_detections(metadata: dict, cam_instance, net_intrinsics):
    """Parse the output tensor into Detection objects."""
    global last_camera_results
    bbox_normalization = net_intrinsics.bbox_normalization

    # Access get_outputs and get_input_size via the imx500 instance passed implicitly via net_intrinsics or globally
    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    input_w, input_h = imx500.get_input_size()

    if np_outputs is None:
        # print("No NN output in metadata")
        return last_camera_results # Return previous if no new output

    detections = []
    try:
        if net_intrinsics.postprocess == "nanodet":
            # Assume postprocess_nanodet_detection is accessible
            results = postprocess_nanodet_detection(
                outputs=np_outputs[0], conf=AI_THRESHOLD, iou_thres=AI_IOU, max_out_dets=AI_MAX_DETECTIONS
            )
            if not results: # Check if results list is empty
                 boxes, scores, classes = [], [], []
            else:
                 boxes, scores, classes = results[0] # Unpack the first element (batch)

            # Assume scale_boxes is accessible
            if len(boxes) > 0: # Only scale if boxes exist
                boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
            else:
                boxes = np.empty((0, 4)) # Ensure boxes is an empty array if no detections

        else: # Default/other postprocessing
            # Check array shapes before accessing indices
            boxes = np_outputs[0][0] if np_outputs[0].shape[0] > 0 else np.empty((0, 4))
            scores = np_outputs[1][0] if np_outputs[1].shape[0] > 0 else np.empty((0,))
            classes = np_outputs[2][0] if np_outputs[2].shape[0] > 0 else np.empty((0,))

            if bbox_normalization and boxes.shape[0] > 0:
                 boxes = boxes / input_h # Assuming square input, otherwise needs adjustment

        # Ensure boxes is correctly shaped for zipping even if empty
        if boxes.shape[0] > 0 and boxes.shape[1] == 4:
             boxes_iter = np.array_split(boxes, 4, axis=1)
             boxes_iter = zip(*boxes_iter)
        else:
             boxes_iter = [] # Empty iterator if no valid boxes


        detections = [
            Detection(box, category, score, metadata, cam_instance, net_intrinsics)
            for box, score, category in zip(boxes_iter, scores, classes)
            if score > AI_THRESHOLD and int(category) < len(get_labels(net_intrinsics)) # Check category index validity
        ]

    except IndexError as e:
         print(f"Error parsing detections (IndexError): {e}. np_outputs shapes: {[o.shape for o in np_outputs]}")
         detections = [] # Return empty on error
    except Exception as e:
         print(f"Error parsing detections: {e}")
         detections = [] # Return empty on error

    last_camera_results = detections # Update global cache
    return detections

def parse_detections_1(metadata: dict, cam_instance, net_intrinsics): # Use 'net_intrinsics' argument name
    """Parse the output tensor into Detection objects, assuming Nanodet postprocessing."""
    # Use the passed 'net_intrinsics' argument, not an undefined variable
    if not net_intrinsics or not hasattr(net_intrinsics, 'labels'): 
        logging.error("parse_detections called with invalid net_intrinsics object.")
        return []

    detections = []
    # Initialize as empty numpy arrays
    boxes = np.empty((0, 4), dtype=np.float32)
    scores = np.empty((0,), dtype=np.float32)
    classes = np.empty((0,), dtype=np.int64)
    process_as_nanodet = False
    
    try:
        
        try:
            if hasattr(net_intrinsics, 'postprocess') and net_intrinsics.postprocess == "nanodet":
                process_as_nanodet = True
                logging.debug("Using 'nanodet'")
            elif hasattr(net_intrinsics, 'postprocess'):
                logging.warning(f"NetworkIntrinsic has postprocess {net_intrinsics.postprocess}, not nanodet")
                process_as_nanodet = True
            else:
                logging.warning("No NetworkIntrinsics postprocessing")
                process_as_nanodet = True
        except Exception as e:
            logging.error(f"Error exception {e}")
            process_as_nanodet = True
        
        np_outputs = imx500.get_outputs(metadata, add_batch=True)
        if np_outputs is None:
            logging.warning("No NN output found in metadata for this frame.")
            return [] # Return empty list

        input_w, input_h = imx500.get_input_size()

        # Use the 'net_intrinsics' argument passed to the function
        labels_cache_key = (tuple(net_intrinsics.labels), net_intrinsics.ignore_dash_labels)
        labels_list = get_labels(labels_cache_key)
        
        if process_as_nanodet:
        # --- Assume Nanodet Postprocessing ---
        # *** The problematic 'if/else' block based on .postprocess is removed ***
            try:
                results = postprocess_nanodet_detection(
                    outputs=np_outputs[0], conf=AI_THRESHOLD, # Use constants
                    iou_thres=AI_IOU, max_out_dets=AI_MAX_DETECTIONS
                )
                if not results: temp_boxes, temp_scores, temp_classes = [], [], []
                else: temp_boxes, temp_scores, temp_classes = results[0]

                # Convert results to NumPy arrays
                boxes = np.array(temp_boxes, dtype=np.float32)
                scores = np.array(temp_scores, dtype=np.float32)
                classes = np.array(temp_classes, dtype=np.int64)

                # Scale boxes if any exist
                if boxes.shape[0] > 0:
                    # Ensure 'scale_boxes' is imported or accessible
                    boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
                # Ensure correct empty shape if conversion results in 1D empty array
                if boxes.ndim == 1 and boxes.shape[0] == 0:
                     boxes = np.empty((0, 4), dtype=np.float32)

            except ValueError as ve:
                # Log the specific ValueError from postprocessing
                logging.error(f"ValueError during Nanodet postprocessing: {ve}. Skipping detections.", exc_info=False)
                # Reset to empty numpy arrays (already initialized, but good practice)
                boxes = np.empty((0, 4), dtype=np.float32)
                scores = np.empty((0,), dtype=np.float32)
                classes = np.empty((0,), dtype=np.int64)
            except Exception as post_err:
                # Catch any other unexpected errors during postprocessing
                logging.error(f"Unexpected error during Nanodet postprocessing: {post_err}", exc_info=True)
                boxes = np.empty((0, 4), dtype=np.float32)
                scores = np.empty((0,), dtype=np.float32)
                classes = np.empty((0,), dtype=np.int64)
        else:
            logging.warning("Skipping postprocessing as Nannodet flag was not set")

        # --- Proceed with creating Detection objects using the resulting arrays ---
        # Check if 'boxes' is a valid NumPy array for iteration
        if isinstance(boxes, np.ndarray) and boxes.shape[0] > 0 and boxes.shape[1] == 4:
             boxes_iter = np.array_split(boxes.astype(float), 4, axis=1)
             boxes_iter = zip(*boxes_iter)
        else:
             boxes_iter = [] # Empty iterator if no valid boxes

        detections = [
            # Use the 'net_intrinsics' argument here too
            Detection(box, category, score, metadata, cam_instance, net_intrinsics)
            for box, score, category in zip(boxes_iter, scores, classes)
            # Combine threshold and index check (safe even if scores/classes are empty)
            if score > AI_THRESHOLD and 0 <= int(category) < len(labels_list)
        ]

    except Exception as e:
         # Catch errors during get_outputs, get_input_size etc.
         logging.error(f"Error in main parse_detections logic: {e}", exc_info=True)
         detections = [] # Ensure empty list on error

    # No need to update global last_camera_results if just returning detections
    # global last_camera_results
    # last_camera_results = detections
    return detections


def parse_detections_old_2(metadata: dict, cam_instance, net_intrinsics): # Use 'net_intrinsics' argument name
    """Parse the output tensor into Detection objects, assuming Nanodet postprocessing."""
    # Use the passed 'net_intrinsics' argument, not an undefined variable
    if not net_intrinsics:
        logging.error("parse_detections called with invalid net_intrinsics object.")
        return []

    detections = []
    # Initialize as empty numpy arrays
    boxes = np.empty((0, 4), dtype=np.float32)
    scores = np.empty((0,), dtype=np.float32)
    classes = np.empty((0,), dtype=np.int64)

    try:
        np_outputs = imx500.get_outputs(metadata, add_batch=True)
        if np_outputs is None:
            logging.warning("No NN output found in metadata for this frame.")
            return [] # Return empty list

        input_w, input_h = imx500.get_input_size()

        # Use the 'net_intrinsics' argument passed to the function
        labels_cache_key = (tuple(net_intrinsics.labels), net_intrinsics.ignore_dash_labels)
        labels_list = get_labels(labels_cache_key)

        # --- Assume Nanodet Postprocessing ---
        # *** The problematic 'if/else' block based on .postprocess is removed ***
        try:
            results = postprocess_nanodet_detection(
                outputs=np_outputs[0], conf=AI_THRESHOLD, # Use constants
                iou_thres=AI_IOU, max_out_dets=AI_MAX_DETECTIONS
            )
            if not results: temp_boxes, temp_scores, temp_classes = [], [], []
            else: temp_boxes, temp_scores, temp_classes = results[0]

            # Convert results to NumPy arrays
            boxes = np.array(temp_boxes, dtype=np.float32)
            scores = np.array(temp_scores, dtype=np.float32)
            classes = np.array(temp_classes, dtype=np.int64)

            # Scale boxes if any exist
            if boxes.shape[0] > 0:
                # Ensure 'scale_boxes' is imported or accessible
                boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
            # Ensure correct empty shape if conversion results in 1D empty array
            if boxes.ndim == 1 and boxes.shape[0] == 0:
                 boxes = np.empty((0, 4), dtype=np.float32)

        except ValueError as ve:
            # Log the specific ValueError from postprocessing
            logging.error(f"ValueError during Nanodet postprocessing: {ve}. Skipping detections.", exc_info=False)
            # Reset to empty numpy arrays (already initialized, but good practice)
            boxes = np.empty((0, 4), dtype=np.float32)
            scores = np.empty((0,), dtype=np.float32)
            classes = np.empty((0,), dtype=np.int64)
        except Exception as post_err:
            # Catch any other unexpected errors during postprocessing
            logging.error(f"Unexpected error during Nanodet postprocessing: {post_err}", exc_info=True)
            boxes = np.empty((0, 4), dtype=np.float32)
            scores = np.empty((0,), dtype=np.float32)
            classes = np.empty((0,), dtype=np.int64)


        # --- Proceed with creating Detection objects using the resulting arrays ---
        # Check if 'boxes' is a valid NumPy array for iteration
        if isinstance(boxes, np.ndarray) and boxes.shape[0] > 0 and boxes.shape[1] == 4:
             boxes_iter = np.array_split(boxes.astype(float), 4, axis=1)
             boxes_iter = zip(*boxes_iter)
        else:
             boxes_iter = [] # Empty iterator if no valid boxes

        detections = [
            # Use the 'net_intrinsics' argument here too
            Detection(box, category, score, metadata, cam_instance, net_intrinsics)
            for box, score, category in zip(boxes_iter, scores, classes)
            # Combine threshold and index check (safe even if scores/classes are empty)
            if score > AI_THRESHOLD and 0 <= int(category) < len(labels_list)
        ]

    except Exception as e:
         # Catch errors during get_outputs, get_input_size etc.
         logging.error(f"Error in main parse_detections logic: {e}", exc_info=True)
         detections = [] # Ensure empty list on error

    # No need to update global last_camera_results if just returning detections
    global last_camera_results
    last_camera_results = detections
    return detections

def parse_detections_old(metadata: dict, cam_instance, net_intrinsics):
    """Parse the output tensor into Detection objects."""
    if not net_intrinsics_obj: return []
    detections = []
    
    boxes = np.empty((0,4),dtype=np.float32)
    scores = np.empty((0,), dtype=np.float32)
    classes = np.empty((0,), dtype=np.int64)
    
    try:
        np_outputs = imx500.get_outputs(metadata, add_batch=True)
        if np_outputs is None: 
            logging.warning("No NN output found in metada for this frame")
            return []

        input_w, input_h = imx500.get_input_size()
        labels_cache_key = (tuple(net_intrinsics_obj.labels), net_intrinsics_obj.ignore_dash_labels)
        labels_list = get_labels(labels_cache_key)
        # Using nanodet postprocessing defined by the constants
        if net_intrinsics_obj.postprocess == "nanodet":
            try:
                results = postprocess_nanodet_detection(
                    outputs=np_outputs[0], conf=AI_THRESHOLD,
                    iou_thres=AI_IOU, max_out_dets=AI_MAX_DETECTIONS
                )
                if not results: # Check if results list is empty
                     boxes, scores, classes = [], [], []
                else:
                     boxes, scores, classes = results[0] # Unpack the first element (batch)
                boxes = np.array(boxes, dtype=np.float32)
                scores = np.array(scores, dtype=np.float32)
                classes = np.array(classes, dtype=np.int64)
                
                if boxes.shape[0] > 0:
                    boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
                if boxes.ndim == 1 and boxes.shape[0] == 0:
                     boxes = np.empty((0, 4), dtype=np.float32)
            except ValueError as ve:
                logging.error(f"ValueError")
            except Exception as post_err:
                logging.error(f"Unexpected error during Nanodet postprocessing {post_err}")
        else:
            logging.warning(f"Unsupported postprocessing type {net_intrinsics_obj.postprocessing}")
        
        if boxes.shape[0] > 0 and boxes.shape[1] == 4:
             boxes_iter = np.array_split(boxes.astype(float), 4, axis=1); boxes_iter = zip(*boxes_iter)
        else: boxes_iter = []

        detections = [
            Detection(box, category, score, metadata, cam_instance, net_intrinsics_obj)
            for box, score, category in zip(boxes_iter, scores, classes)
            if score > AI_THRESHOLD and 0 <= int(category) < len(labels_list)
        ]
    except Exception as e:
         logging.error(f"Error parsing detections: {e}", exc_info=True)
         detections = []
    return detections

def draw_detections(frame, detection_results, net_intrinsics):
    """Draw detections onto frame copy. Returns drawn RGB frame and person count."""
    #if not net_intrinsics_obj: return frame_rgb, 0
    #labels_cache_key = (tuple(net_intrinsics_obj.labels), net_intrinsics_obj.ignore_dash_labels)
    #labels = get_labels(labels_cache_key)
    person_count = 0
    #display_frame = frame_rgb.copy()
    
    # COPY FROM draw_detections in fusion_v2.py
    labels = get_labels(net_intrinsics)
    person_detected = False
    for detection in detection_results:
        try:
            x, y, w, h = map(int, detection.box_xywh)
            label_index = int(detection.category)
            if 0 <= label_index < len(labels):
                 label_name = labels[label_index]
                 label_text = f"{label_name} ({detection.conf:.2f})"
                 person_detected = label_name.lower() == 'person'
                 if person_detected: person_count += 1
                 # Colors are BGR format for OpenCV drawing
                 color = (0, 0, 255) if person_detected else (0, 255, 0) # Red for person

                 cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                 (lw, lh), base = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                 label_y = y - base // 2
                 bg_y1 = max(0, y - lh - base); bg_y2 = max(lh + base, y)
                 cv2.rectangle(frame, (x, bg_y1), (x + lw, bg_y2), color, cv2.FILLED)
                 cv2.putText(frame, label_text, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else: print(f"Invalid category index {label_index} detected.") # Can be noisy
        except Exception as e: logging.error(f"Error drawing detection: {e}", exc_info=True)
    return frame, person_count

# =================================================
# Helper Functions
# =================================================
def linear_to_dB(x):
    """Convert linear scale to dB safely."""
    if x is None: return None
    x_abs = np.abs(x)
    floor = 1e-9
    x_db = 20 * np.log10(np.maximum(x_abs, floor))
    return x_db

# --- get_camera_intrinsics remains a placeholder ---
def get_camera_intrinsics(picam2_instance):
     logging.warning("Fetching precise camera intrinsics not implemented. Use offline calibration.")
     return None

# =================================================
# Worker Threads
# =================================================

def radar_worker(display_q, record_q, stop_event, display_is_enabled):
    """Acquires radar IQ data, queues for recording, optionally processes for display."""
    global radar_config_used # Allow updating global dict
    logging.info("Radar thread starting.")
    print(f"display_q {display_q}, record_q {record_q}, stop_event {stop_event}")
    print(f"[radar_worker] display_is_enabled: {display_is_enabled}")
    
    radar_frame_count = 0
    frames_dropped_queue = 0
    doppler_algo = None

    try:
        with DeviceFmcw() as device:
            sdk_ver = get_version_full()
            sensor_info = device.get_sensor_information()
            num_rx_antennas = sensor_info["num_rx_antennas"]
            logging.info(f"Radar SDK Version: {sdk_ver}")
            logging.info(f"Rx Antennas: {num_rx_antennas}")
            # Calculate required bandwidth for target range res (approx)
            #bandwidth_hz = LIGHT_SPEED / (2 * 0.075) # Target 7.5cm res -> 2GHz BW
            sequence = device.create_simple_sequence(FmcwSimpleSequenceConfig())
            sequence.loop.repetition_time_s = 1.0 / RADAR_FRAME_RATE_HZ
             
            chirp_loop = sequence.loop.sub_sequence.contents
            # Let SDK calculate samples/chirps if possible, or force them?
            # Forcing them might conflict with metrics goals. Let's try forcing.
            chirp_loop.loop.num_repetitions = RADAR_NUM_CHIRPS_PER_FRAME # Force chirps
            chirp = chirp_loop.loop.sub_sequence.contents.chirp # Get the calculated chirp config
            chirp.sample_rate_Hz = RADAR_SAMPLE_RATE_HZ
            chirp.num_samples = RADAR_NUM_SAMPLES_PER_CHIRP # Force samples
            chirp.start_frequency_Hz = RADAR_CENTER_FREQ_HZ - (RADAR_BANDWIDTH_HZ / 2)
            chirp.end_frequency_Hz = RADAR_CENTER_FREQ_HZ + (RADAR_BANDWIDTH_HZ / 2)
            chirp.rx_mask = (1 << num_rx_antennas) - 1 if RADAR_RX_MASK == "auto" else int(RADAR_RX_MASK)
            chirp.tx_mask = int(RADAR_TX_MASK)
            chirp.tx_power_level = int(RADAR_TX_POWER)
            chirp.if_gain_dB = int(RADAR_IF_GAIN_DB)
            
            print(f"Applying sequence derived from metrics, overridden with constants: "
                          f"Samples={chirp.num_samples}, Chirps={chirp_loop.loop.num_repetitions}, "
                          f"Fs={chirp.sample_rate_Hz}, FrameTime={sequence.loop.repetition_time_s:.4f}s")
            device.set_acquisition_sequence(sequence)


            #except Exception as config_err:
            #     logging.error(f"Failed to configure radar sequence: {config_err}", exc_info=True)
            #     raise # Re-raise to stop the thread


            # --- Store final config ---
            final_chirp_config = {
                'num_samples': chirp.num_samples, 'sample_rate_Hz': chirp.sample_rate_Hz,
                'rx_mask': chirp.rx_mask, 'tx_mask': chirp.tx_mask,
                'tx_power_level': chirp.tx_power_level, 'if_gain_dB': chirp.if_gain_dB,
                'start_frequency_Hz': chirp.start_frequency_Hz, 'end_frequency_Hz': chirp.end_frequency_Hz,
                # Add lp/hp cutoffs if set
            }
            final_chirp_loop_config = {'num_repetitions': chirp_loop.loop.num_repetitions}
            final_sequence_config = {'frame_repetition_time_s': sequence.loop.repetition_time_s}
            radar_config_used.update({
                'sdk_version': sdk_ver, 'num_rx_antennas': num_rx_antennas,
                'final_chirp_config': final_chirp_config,
                'final_chirp_loop_config': final_chirp_loop_config,
                'final_sequence_config': final_sequence_config,
#                'metrics_used_for_init': vars(metrics) # Include the metrics used
            })
            print(f"Stored Final Radar Config: {radar_config_used}")
            print(f"display_is_enabled : {display_is_enabled}, DopplerAlgo {DopplerAlgo}")
            
            # Initialize Doppler algo ONLY if display is enabled AND library is available
            if display_is_enabled and DopplerAlgo:
                try:
                    # Need to recalculate max_speed based on final config for display
                    # max_speed = wavelength / (4 * chirp_to_chirp_time)
                    # chirp_to_chirp_time might be frame_config.frame_repetition_time_s / frame_config.num_chirps_per_frame
                    # For now, use a placeholder if calc is complex
                    display_max_speed = 3.0 # Placeholder
                    display_max_range = 5.0 # Placeholder - use metrics?
                    doppler_algo = DopplerAlgo(chirp.num_samples, chirp_loop.loop.num_repetitions, num_rx_antennas)
                    print("Doppler algorithm initialized for live view.")
                except Exception as e:
                     print(f"Failed to initialize DopplerAlgo: {e}")
                     doppler_algo = None


            logging.info("Starting radar acquisition loop...")
            while not stop_event.is_set():
                try:
                    acq_timestamp = time.time()
                    timeout_ms = max(1000, int(2 * sequence.loop.repetition_time_s * 1000))
                    frame_contents = device.get_next_frame(timeout_ms=timeout_ms)

                    if frame_contents is None:
                        logging.warning("Radar frame acquisition timed out.")
                        continue

                    radar_frame_count += 1
                    raw_iq_data = frame_contents[0]

                    # --- Queue Raw Data for Recording ---
                    if raw_iq_data.dtype != np.complex64:
                         raw_iq_data = raw_iq_data.astype(np.complex64)
                    record_data = {'type': 'radar', 'timestamp': acq_timestamp,
                                   'frame_index': radar_frame_count, 'iq_data': raw_iq_data}
                    try: record_q.put(record_data, block=False)
                    except queue.Full:
                        frames_dropped_queue += 1
                        logging.warning(f"RecQ Full! Drop R#{radar_frame_count}. Total: {frames_dropped_queue}")
                        try: record_q.get_nowait()
                        except queue.Empty: pass
                        try: record_q.put(record_data, block=False)
                        except queue.Full: logging.error("RecQ still full after drop!")


                    # --- Process and Queue Data for Live Display (Optional) ---
                    if doppler_algo: # Implicitly checks if display is enabled
                        data_all_antennas_db = []
                        for i_ant in range(num_rx_antennas):
                            mat = raw_iq_data[i_ant, :, :]
                            doppler_map = doppler_algo.compute_doppler_map(mat, i_ant)
                            dfft_dbfs = linear_to_dB(doppler_map)
                            data_all_antennas_db.append(dfft_dbfs)
                        display_data = {'type': 'data', 'data': data_all_antennas_db, 'num_ant': num_rx_antennas}
                        try: display_q.put(display_data, block=False)
                        except queue.Full:
                            try: display_q.get_nowait()
                            except queue.Empty: pass
                            try: display_q.put(display_data, block=False)
                            except queue.Full: pass # Display drop ok

                except Exception as e:
                    logging.error(f"Error in radar loop: {e}", exc_info=True)
                    time.sleep(0.1)

    except Exception as e:
        #logging.critical(f"Radar thread failed: {e}", exc_info=True)
        print(f"Radar thread failed: {e}", exc_info=True)
        stop_event.set()
    finally:
        #logging.info(f"Radar thread finished. Acquired: {radar_frame_count}. Dropped: {frames_dropped_queue}")
        print("Radar thread finished. Acquired: {radar_frame_count}. Dropped: {frames_dropped_queue}")


def camera_worker(display_q, record_q, stop_event, display_is_enabled):
    """Captures camera frames, runs AI, queues data for recording and optional display."""
    #global camera_config_used, camera_sensor_format # Allow updating globals
    global picam2, imx500, intrinsics # Access global instances initialized in main
    logging.info("Camera thread starting.")
    
    print(picam2)
    print(imx500)
    print(intrinsics)
    
    if not picam2 or not imx500 or not intrinsics:
        print("Camera components not initialized!")
        stop_event.set()
        return
        
    #picam2 = cam_objects['picam2']
    #imx500 = cam_objects['imx500']
    #intrinsics = cam_objects['intrinsics']

    #if not all([picam2, imx500, intrinsics]):
    #    logging.critical("Camera components not initialized properly!")
    #    stop_event.set(); return

    camera_frame_count = 0
    frames_dropped_queue = 0
    # Store final config
    camera_config_used.update(picam2.camera_configuration())
    camera_sensor_format = camera_config_used.get('main', {}).get('format', 'Unknown')

    logging.info("Starting camera acquisition loop...")
    while not stop_event.is_set():
        try:
            acq_timestamp = time.time()
            metadata = picam2.capture_metadata()
            frame_array_rgb = picam2.capture_array("main") # Expecting RGB , 
            #frame_array_rgb here correspond to frame_array in fusion_v2.py

            if frame_array_rgb is None:
                 logging.warning("Camera capture failed.")
                 time.sleep(0.1); continue

            camera_frame_count += 1

            # --- Run AI Inference ---
            detections = parse_detections(metadata, picam2, intrinsics)
            detections_list_dict = [det.to_dict() for det in detections]

            # --- Queue Data for Recording ---
            record_data = {'type': 'camera', 'timestamp': acq_timestamp,
                           'frame_index': camera_frame_count,
                           'raw_frame': frame_array_rgb, # Keep as RGB
                           'detections': detections_list_dict}
            try: record_q.put(record_data, block=False)
            except queue.Full:
                frames_dropped_queue += 1
                logging.warning(f"RecQ Full! Drop C#{camera_frame_count}. Total: {frames_dropped_queue}")
                try: record_q.get_nowait()
                except queue.Empty: pass
                try: record_q.put(record_data, block=False)
                except queue.Full: logging.error("RecQ still full after drop!")


            # --- Process and Queue Data for Live Display (Optional) ---
            if display_is_enabled:
                frame_array_rgb_copy = frame_array_rgb
                display_frame_rgb, person_count = draw_detections(frame_array_rgb_copy, detections, intrinsics)
                # Add minimal stats
                cv2.putText(display_frame_rgb, f"Frame: {camera_frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.putText(display_frame_rgb, f"Persons: {person_count}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                # Convert final display frame to BGR for cv2.imshow
                display_frame_bgr = cv2.cvtColor(display_frame_rgb, cv2.COLOR_RGB2BGR)
                try: display_q.put(display_frame_bgr, block=False)
                except queue.Full:
                    try: display_q.get_nowait()
                    except queue.Empty: pass
                    try: display_q.put(display_frame_bgr, block=False)
                    except queue.Full: pass


        except Exception as e:
            logging.error(f"Error in camera loop: {e}", exc_info=True)
            time.sleep(0.1)

    logging.info(f"Camera thread finished. Acquired: {camera_frame_count}. Dropped: {frames_dropped_queue}")


def recording_worker(record_q, session_path, stop_flag):
    """Handles saving data from the recording queue to disk."""
    global camera_sensor_format # Need format for BGR conversion check
    logging.info(f"Recording worker starting. Saving to: {session_path}")
    items_saved = 0
    metadata_entries_buffer = []
    flush_interval = METADATA_FLUSH_INTERVAL

    try:
        radar_dir = os.path.join(session_path, "radar_raw")
        camera_dir = os.path.join(session_path, "camera_frames")
        os.makedirs(radar_dir, exist_ok=True)
        os.makedirs(camera_dir, exist_ok=True)
        metadata_file_path = os.path.join(session_path, "metadata.jsonl")

        with open(metadata_file_path, 'w') as meta_f:
            while not stop_event.is_set() or not record_q.empty():
                try:
                    record_item = record_q.get(block=True, timeout=0.5) # Wait longer
                    if record_item is None: break # Sentinel

                    item_type = record_item['type']
                    timestamp = record_item['timestamp']
                    frame_idx = record_item['frame_index']
                    filepath = None; rel_path = None

                    metadata_entry = {'timestamp': timestamp, 'type': item_type, 'frame_index': frame_idx}

                    if item_type == 'radar':
                        filename = f"radar_{frame_idx:06d}_{timestamp:.6f}.npy"
                        rel_path = os.path.join("radar_raw", filename)
                        filepath = os.path.join(session_path, rel_path)
                        try: np.save(filepath, record_item['iq_data'])
                        except OSError as e: handle_disk_full(e); return # Exit on disk full
                        except Exception as e: logging.error(f"Failed save R#{frame_idx}: {e}"); filepath = None

                    elif item_type == 'camera':
                        filename = f"cam_{frame_idx:06d}_{timestamp:.6f}.png"
                        rel_path = os.path.join("camera_frames", filename)
                        filepath = os.path.join(session_path, rel_path)
                        frame_rgb = record_item['raw_frame']
                        frame_bgr = frame_rgb
                        # Convert RGB to BGR before saving PNG
                        if camera_sensor_format == "RGB888":
                            try: frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                            except Exception as e: logging.error(f"Failed cvt C#{frame_idx}: {e}")
                        try:
                            cv2.imwrite(filepath, frame_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                            metadata_entry['detections'] = record_item['detections']
                        except OSError as e: handle_disk_full(e); return # Exit on disk full
                        except Exception as e: logging.error(f"Failed save C#{frame_idx}: {e}"); filepath = None

                    if rel_path: metadata_entry['filename'] = rel_path

                    # Buffer metadata entry
                    metadata_entries_buffer.append(metadata_entry)
                    items_saved += 1

                    # Flush buffer periodically or if stopping
                    if len(metadata_entries_buffer) >= flush_interval or \
                       (stop_event.is_set() and record_q.empty()):
                        flush_metadata(meta_f, metadata_entries_buffer, items_saved) # Use helper

                except queue.Empty:
                    if stop_event.is_set(): break
                    continue
                except Exception as e:
                    logging.error(f"Error in recording worker loop: {e}", exc_info=True)
                    time.sleep(0.1)

            # Final flush
            flush_metadata(meta_f, metadata_entries_buffer, items_saved, is_final=True)

    except Exception as e:
         logging.critical(f"Recording worker failed critically: {e}", exc_info=True)
    finally:
         logging.info(f"Recording worker finished. Total items processed: {items_saved}.")


def flush_metadata(file_handle, buffer_list, total_saved_count, is_final=False):
    """Helper to write buffered metadata to file."""
    if not buffer_list: return
    action = "Final flush" if is_final else "Flushing"
    logging.debug(f"{action} {len(buffer_list)} metadata entries. Total saved: {total_saved_count}")
    try:
        for entry in buffer_list:
            file_handle.write(json.dumps(entry) + '\n')
        file_handle.flush() # Ensure written to OS buffer
        os.fsync(file_handle.fileno()) # Ensure written to disk (important!)
        buffer_list.clear()
    except Exception as e:
        logging.error(f"Failed during metadata flush: {e}")
        # Consider how to handle buffer on error (e.g., retry? dump to separate file?)

def handle_disk_full(os_error):
    """Checks for disk full error and signals stop."""
    if os_error.errno == errno.ENOSPC:
        logging.critical("Disk full! Stopping recording.")
        stop_event.set()
    else:
        raise os_error # Re-raise other OS errors

# =================================================
# Main Execution Logic
# =================================================
def main():
    #global picam2, imx500, intrinsics, radar_config_used, camera_config_used, camera_intrinsics_calib, camera_sensor_format
    #DO NOT REMOVE THIS LINE, OTHERWISE ABOVE FUNCTIONS WILL BE ERROR
    global picam2, imx500, intrinsics
    
    parser = argparse.ArgumentParser(description="Run and Record Radar-Camera Fusion Data (v3 - Simple).",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Keep args simple as requested
    parser.add_argument('--record', action='store_true', default=True, # Record by default
                        help="Enable recording of sensor data.")
    parser.add_argument('--no-record', action='store_false', dest='record',
                        help="Disable recording.")
    parser.add_argument('--display', action='store_true', default=False,
                        help="Enable live display windows (uses more resources).")
    parser.add_argument('--duration', type=float, default=0,
                        help="Max recording duration in seconds (0 for indefinite until Ctrl+C/quit).")
    parser.add_argument('--session', type=str, default=SESSION_NAME_PREFIX,
                        help="Name prefix for the recording session directory.")

    args = parser.parse_args()

    # Update global based on args if needed (overrides constants)
    display_is_enabled = args.display # Set global based on arg
    print(f"display_is_enabled: {display_is_enabled}")
    
    logging.info("Starting Radar-Camera Fusion Recorder (v3 - Simple)...")
    logging.info(f"Recording: {args.record}, Live Display: {display_is_enabled}")

    # --- Initialize Recording ---
    recording_session_path = None
    recording_thread = None
    if args.record:
        recording_active.set()
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_name = f"{timestamp_str}_{args.session}"
        recording_session_path = os.path.join(RECORDING_BASE_DIR, session_name)
        try:
            os.makedirs(recording_session_path, exist_ok=True)
            logging.info(f"RECORDING ENABLED: Saving to {recording_session_path}")
        except Exception as e:
             logging.critical(f"Cannot create recording dir {recording_session_path}: {e}")
             logging.warning("Disabling recording.")
             args.record = False
             recording_active.clear()
    else:
        logging.info("Recording disabled.")


    # --- Initialize Camera ---
    try:
        print(f"Loading AI model: {AI_MODEL_PATH}")
        imx500 = IMX500(AI_MODEL_PATH)
        intrinsics = imx500.network_intrinsics
        print("IMX500 Initialized.")

        picam2 = Picamera2(imx500.camera_num)
        #controls_dict = CAMERA_CONTROLS.copy()
        #if 'FrameRate' in controls_dict: controls_dict['FrameRate'] = float(controls_dict['FrameRate'])
        print(f"tuple(CAMERA_RESOLUTION) : {tuple(CAMERA_RESOLUTION)}")
        preview_config = picam2.create_preview_configuration(
            main={"size": tuple(CAMERA_RESOLUTION), "format": CAMERA_FORMAT},
            controls={}, queue=False, buffer_count=CAMERA_BUFFER_COUNT
        )
        picam2.configure(preview_config)
        time.sleep(0.5) # Allow config to apply
        
        #camera_config_used = picam2.camera_configuration() # Store final config
        #camera_sensor_format = camera_config_used.get('main', {}).get('format', 'Unknown')
        #logging.info(f"Picamera2 configured: {camera_config_used.get('main', {})}")

        #camera_intrinsics_calib = get_camera_intrinsics(picam2)

        imx500.show_network_fw_progress_bar()
        picam2.start()
        print("Picamera2 Started.")
        time.sleep(2.0) # Allow camera to settle

    except Exception as e:
        logging.critical(f"FATAL: Camera Init Failed: {e}", exc_info=True)
        sys.exit(1)


    # --- Initialize Radar Plot (Optional) ---
    radar_plot = None
    if display_is_enabled and plt and DopplerAlgo:
        num_rx_antennas_guess = 3 # Default guess
        print(f"RADAR_MAX_SPEED_M_S: {RADAR_MAX_SPEED_M_S}, RADAR_MAX_RANGE_M: {RADAR_MAX_RANGE_M}")
        radar_plot = RadarDraw(RADAR_MAX_SPEED_M_S, RADAR_MAX_RANGE_M, num_rx_antennas_guess, stop_event, display_is_enabled) # Placeholder max speed/range
        radar_plot.setup_plot()


    # --- Start Worker Threads ---
    logging.info("Starting worker threads...")
    #cam_objects = {'picam2': picam2, 'imx500': imx500, 'intrinsics': intrinsics}
    radar_thread = threading.Thread(target=radar_worker, name="RadarWorker",
                                    args=(radar_display_queue, recording_queue, stop_event, display_is_enabled), daemon=True) # Pass empty dict for config initially
    camera_thread = threading.Thread(target=camera_worker, name="CameraWorker",
                                     args=(camera_display_queue, recording_queue, stop_event, display_is_enabled), daemon=True)
    #radar_worker(display_q, record_q, stop_event, record_event, display_is_enabled)
    print(f"args.record: {args.record}")
    if args.record:
        recording_thread = threading.Thread(target=recording_worker, name="RecordingWorker",
                                            args=(recording_queue, recording_session_path, stop_event), daemon=True)

    radar_thread.start()
    camera_thread.start()
    if recording_thread: recording_thread.start()


    # --- Save Session Info (if recording) ---
    if args.record and recording_session_path:
         time.sleep(2.0) # Allow threads short time to populate details
         session_info = {
             "session_id": os.path.basename(recording_session_path),
             "start_time_iso": datetime.now().isoformat(),
             "notes": "Multi-person scenario recording.",
             "radar_config_applied": radar_config_used, # Populated by radar thread
             "camera_config_applied": camera_config_used, # Populated by camera thread
             "camera_intrinsics_calib": camera_intrinsics_calib,
             "network_intrinsics": vars(intrinsics) if intrinsics else None,
             "ai_model_path": AI_MODEL_PATH,
             "ai_threshold": AI_THRESHOLD,
             "extrinsic_calibration_notes": "REQUIRED: Perform offline extrinsic calibration."
         }
         info_file_path = os.path.join(recording_session_path, "session_info.json")
         try:
             with open(info_file_path, 'w') as f: json.dump(session_info, f, indent=4, default=str)
             logging.info(f"Saved session info to {info_file_path}")
         except Exception as e: logging.error(f"Error saving session info: {e}", exc_info=True)


    # --- Main Loop ---
    
    logging.info("Starting main loop. Press Ctrl+C to stop.")
    main_start_time = time.time()
    last_radar_antennas = num_rx_antennas_guess
    radar_data = None
    camera_frame = None
    print(f"[main]:display_is_enabled {display_is_enabled}")
    while not stop_event.is_set():
        try:
            current_time = time.time()
            # Check recording duration limit
            if args.record and args.duration > 0 and (current_time - main_start_time) > args.duration:
                logging.info(f"Recording duration ({args.duration}s) reached. Stopping...")
                stop_event.set(); break

            # --- Live Display Logic ---
            #print(f"Here: {display_is_enabled}")
            if display_is_enabled:
                # Update Radar Plot
                try:
                    radar_update = radar_display_queue.get(block=False) # Non-blocking get
                    if radar_update['type'] == 'data':
                        radar_data = radar_update['data']
                        # Check if number of antennas changed (first successful frame)
                        if radar_update['num_ant'] != last_radar_antennas:
                             print(f"Updating radar plot for {radar_update['num_ant']} antennas.")
                             # Need to recreate plot or adjust layout - simpler to close/reopen
                             radar_plot.close()
                             plt.close('all') # Make sure it's gone
                             radar_plot = RadarDraw(RADAR_MAX_SPEED_M_S, RADAR_MAX_RANGE_M, radar_update['num_ant'], stop_event)
                             radar_plot.setup_plot()
                             last_radar_antennas = radar_update['num_ant']
                             # Reset plot history
                             radar_plot._h = []

                        if radar_data: # Ensure data is valid before drawing
                             radar_plot.draw(radar_data)

                except queue.Empty:
                    # No new radar data, maybe redraw last known state? Or do nothing.
                    # If using plt.pause() in draw, it handles updates.
                    pass # No new data is fine

                # Update Camera Window
                try:
                    camera_frame_bgr = camera_display_queue.get(block=False)
                    if camera_frame_bgr is not None: cv2.imshow("Camera AI (Live)", camera_frame_bgr)
                except queue.Empty: pass # Okay if no update

                # Handle Window Events & Quit Key
                key = cv2.waitKey(1) # Essential for GUI updates
                if key == ord('q') or key == 27:
                    logging.info("Quit key pressed."); stop_event.set(); break
                # Check if plot window was closed
                if radar_plot and not radar_plot._is_window_open:
                     if not stop_event.is_set(): # Avoid double log
                         logging.info("Radar plot closed by user. Stopping."); stop_event.set()
                     break

            else: # No display - just sleep briefly
                time.sleep(0.1)

        except KeyboardInterrupt:
             logging.info("Ctrl+C detected. Stopping."); stop_event.set(); break
        except Exception as e:
            logging.error(f"Error in main loop: {e}", exc_info=True); stop_event.set(); break


    # --- Shutdown Sequence ---
    logging.info("Initiating shutdown sequence...")
    stop_event.set()

    # Signal recorder (if running)
    if args.record and recording_thread and recording_thread.is_alive():
        logging.info("Signalling recording worker to finish...")
        try: recording_queue.put(None, block=True, timeout=1.0)
        except queue.Full: logging.warning("Could not add sentinel to recording queue.")

    logging.info("Waiting for worker threads to join...")
    threads_to_join = [radar_thread, camera_thread]
    if recording_thread: threads_to_join.append(recording_thread)

    for t in threads_to_join:
        join_timeout = 15.0 if t.name == "RecordingWorker" else 5.0
        t.join(timeout=join_timeout)
        if t.is_alive(): logging.warning(f"Thread {t.name} did not finish within {join_timeout}s.")

    logging.info("Stopping camera hardware...")
    if picam2:
        try: picam2.stop()
        except Exception as e: logging.error(f"Error stopping picam2: {e}")

    logging.info("Closing GUI windows...")
    cv2.destroyAllWindows()
    if radar_plot: radar_plot.close()
    if plt: plt.close('all')

    logging.info("Application finished.")

# --- Run Main ---
if __name__ == "__main__":
    main()
