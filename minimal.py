#!/usr/bin/env python3
"""
fusion_v4.py - Refactored Radar-Camera Fusion Recorder
Features:
 - Modes: --display (live view), --record (save data), --no-record
 - Robust thread lifecycle and queue management
 - Reduced blocking and drops under load
 - Configurable durations and session naming
 - Logging instead of print statements
"""
import argparse
import logging
import os
import sys
import time
import threading
import queue
import json
from datetime import datetime
from functools import lru_cache
# External imports
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import cv2
    import numpy as np
    from ifxradarsdk import get_version_full
    from ifxradarsdk.fmcw import DeviceFmcw
    from ifxradarsdk.fmcw.types import FmcwSimpleSequenceConfig
    import psutil # Keep for optional resource monitoring
    from picamera2 import Picamera2
    from picamera2.controls import Controls
    from picamera2.devices import IMX500
    from picamera2.devices.imx500 import (NetworkIntrinsics,
                                           postprocess_nanodet_detection)
    from picamera2.devices.imx500.postprocess import scale_boxes
    # from helpers.v3_helpers import parse_detections, draw_detections

except ImportError as e:
    logging.critical(f"Missing dependency: {e}")
    sys.exit(1)


import numpy as np

# ---------------------------------------------------
# Radar hyper-parameters — adjust to your board’s limits
RADAR_FRAME_RATE_HZ        = 10.0             # 10 Hz
RADAR_SAMPLE_RATE_HZ       = 1_000_000        # 1 MS/s
RADAR_NUM_SAMPLES_PER_CHIRP= 128
RADAR_NUM_CHIRPS_PER_FRAME = 64
RADAR_BANDWIDTH_HZ         = 460_000_000      # 460 MHz
RADAR_CENTER_FREQ_HZ       = 60_750_000_000   # 60.75 GHz
RADAR_TX_POWER             = 31               # max power level
RADAR_IF_GAIN_DB           = 33
LIGHT_SPEED = 299_792_458
RADAR_RANGE_RESOLUTION_M = LIGHT_SPEED / (2 * RADAR_BANDWIDTH_HZ)
RADAR_MAX_RANGE_M = (RADAR_NUM_SAMPLES_PER_CHIRP * LIGHT_SPEED) / (4 * RADAR_BANDWIDTH_HZ)
RADAR_MAX_SPEED_M_S = 3.0

# ---------------------------------------------------
AI_THRESHOLD = 0.55
AI_IOU = 0.65
AI_MAX_DETECTIONS = 10 # Max persons expected + buffer (e.g., up to 6 people)

imx500 = None
intrinsics = None
picam2 = None
last_camera_results = []

# Constants
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
RECORD_DIR = os.getenv("RECORD_DIR", "./recordings_v4")
QUEUE_SIZE = 100
METADATA_FLUSH = 50

# Initialize logging
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Shared resources
dash_event = threading.Event()
recording_event = threading.Event()
record_q = queue.Queue(maxsize=QUEUE_SIZE)

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
        self.box_xywh = imx500.convert_inference_coords(coords, metadata, cam_instance)


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

def draw_detections(frame, detection_results, net_intrinsics):
    """Draw detections onto frame copy. Returns drawn RGB frame and person count."""

    person_count = 0
    #display_frame = frame_rgb.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

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


# Worker threads
class RadarThread(threading.Thread):
    def __init__(self, display_q, record_q, event, display):
        super().__init__(daemon=True)
        self.display_q = display_q
        self.record_q = record_q
        self.stop_event = event
        self.display = display

    def run(self):
        logging.info("[Radar] Starting")
        try:
            with DeviceFmcw() as dev:
                # 1) build the simple FMCW sequence
                cfg = FmcwSimpleSequenceConfig()
                sequence = dev.create_simple_sequence(cfg)

                # 2) paper’s frame period: 100 ms → 10 Hz
                sequence.loop.repetition_time_s = 100e-3

                # 3) dive into the chirp loop: 64 chirps per frame, 391 µs apart
                chirp_loop = sequence.loop.sub_sequence.contents
                chirp_loop.loop.num_repetitions     = 64
                chirp_loop.loop.repetition_time_s   = 391e-6

                # 4) force your sweep specs
                chirp = chirp_loop.loop.sub_sequence.contents.chirp
                chirp.sample_rate_Hz     = 2_000_000        # 2 MHz
                chirp.num_samples        = 128
                chirp.start_frequency_Hz = 60.75e9 - 460e6/2
                chirp.end_frequency_Hz   = 60.75e9 + 460e6/2

                # 5) **enable exactly 3 Rx antennas** (bits 0,1,2 → mask=0b111)
                chirp.rx_mask        = 0x7
                chirp.tx_mask        = 0x1
                chirp.tx_power_level = RADAR_TX_POWER      # e.g. 31
                chirp.if_gain_dB     = RADAR_IF_GAIN_DB    # e.g. 33

                # 6) now upload *that* legal sequence
                dev.set_acquisition_sequence(sequence)

                frame_idx = 0
                while not self.stop_event.is_set():
                    ts = time.time()
                    frame = dev.get_next_frame(timeout_ms=1000)
                    if frame is None:
                        logging.warning("[Radar] Timeout")
                        continue
                    raw = frame[0].astype(np.complex64)
                    # Record
                    if recording_event.is_set():
                        try:
                            self.record_q.put_nowait(("radar", frame_idx, ts, raw))
                        except queue.Full:
                            logging.warning("[Radar] Record queue full, dropping")
                    # Display
                    if self.display:
                        mag = 20 * np.log10(np.abs(raw) + 1e-9)  # shape (num_rx, Nc, Ns)
                        maps = [mag[i,:,:] for i in range(mag.shape[0])]
                        msg = {"type":"data", "data":maps, "num_ant":len(maps)}
                        try:
                            self.display_q.put_nowait(msg)
                        except queue.Full:
                            pass
                    frame_idx = 1
        except Exception:
            logging.exception("[Radar] Error")
        finally:
            logging.info("[Radar] Stopped")

class CameraThread(threading.Thread):
    def __init__(self, display_q, record_q, event, display, stats, model_path, threshold, iou):
        super().__init__(daemon=True)
        self.display_q = display_q
        self.record_q = record_q
        self.stop_event = event
        self.display = display
        self.model_path = model_path
        self.threshold = threshold
        self.iou = iou
        self.stats   = stats
        # persistent FPS counters
        self._frame_count = 0
        self._start_time  = time.time()

    def run(self):
        logging.info("[Camera] Starting")

        try:
            imu   = imx500
            picam = picam2
            frame_idx = 0
            while not self.stop_event.is_set():
                ts = time.time()
                md = picam.capture_metadata()
                arr = picam.capture_array("main")
                # AI inference
                detections = parse_detections(md, picam, imu.network_intrinsics)
                # record
                if recording_event.is_set():
                    dto = [d.to_dict() for d in detections]
                    self.record_q.put_nowait(("camera", frame_idx, ts, arr, dto))

                # display
                if self.display:
                    # always draw detections
                    disp_frame, _ = draw_detections(arr, detections, intrinsics)
                    
                    # Convert to RGB for realistic view
                    disp_frame = cv2.cvtColor(disp_frame, cv2.COLOR_BGR2RGB)
                    # overlay stats if requested
                    if self.stats:
                        self._frame_count += 1
                        elapsed = time.time() - self._start_time
                        fps     = self._frame_count / elapsed if elapsed > 0 else 0.0
                        cpu     = psutil.cpu_percent(interval=None)
                        mem     = psutil.virtual_memory().percent
                        cv2.putText(disp_frame, f"FPS: {fps:5.1f}",    (10,30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (  0,255,  0), 2)
                        cv2.putText(disp_frame, f"CPU: {cpu:5.0f} %",  (10,60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (  0,255,255), 2)
                        cv2.putText(disp_frame, f"MEM: {mem:5.0f} %",  (10,90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (  0,255,255), 2)
                    try:
                        self.display_q.put(("camera", disp_frame), block=False)

                    except queue.Full:
                        # drop oldest
                        try: self.display_q.get(block=False)
                        except queue.Empty: pass
                        # retry once
                        try: self.display_q.put(("camera", disp_frame), block=False)
                        except queue.Full: pass

                frame_idx = 1
        except Exception:
            logging.exception("[Camera] Error")
        finally:
            if picam:
                picam.stop()
            logging.info("[Camera] Stopped")

class RecordThread(threading.Thread):
    def __init__(self, record_q, session):
        super().__init__(daemon=True)
        self.q = record_q
        self.dir = session
        self.stop_event = dash_event

    def run(self):
        logging.info("[Record] Starting")
        os.makedirs(os.path.join(self.dir, "radar"), exist_ok=True)
        os.makedirs(os.path.join(self.dir, "camera"), exist_ok=True)
        meta_path = os.path.join(self.dir, "metadata.jsonl")
        buf = []
        with open(meta_path, 'w') as mf:
            while not (self.stop_event.is_set() and self.q.empty()):
                try:
                    typ, idx, ts, data, *rest = self.q.get(timeout=0.5)
                except queue.Empty:
                    continue
                entry = {"type":typ, "index":idx, "ts":ts}
                if typ=="radar":
                    fn = f"radar_{idx:06d}_{ts:.6f}.npy"
                    np.save(os.path.join(self.dir, "radar", fn), data)
                    entry['file']=fn
                else:
                    fn = f"cam_{idx:06d}_{ts:.6f}.png"
                    cv2.imwrite(os.path.join(self.dir, "camera", fn), data[...,::-1])
                    entry['file']=fn; entry['dets']=rest[0]
                buf.append(entry)
                if len(buf)>=METADATA_FLUSH:
                    for e in buf: mf.write(json.dumps(e)+"\n")
                    mf.flush(); buf.clear()
            # final flush
            for e in buf: mf.write(json.dumps(e)+"\n")
            mf.flush()
        logging.info("[Record] Stopped")


# Main
def main():

    global picam2, imx500, intrinsics

    p = argparse.ArgumentParser()
    p.add_argument('--no-record', dest='record', action='store_false')
    p.add_argument('--display', action='store_true')
    p.add_argument('--duration', type=float, default=0)
    p.add_argument('--session', type=str, default="session")
    p.add_argument('--model', default="./imx500-models-backup/imx500_network_yolov8n_pp.rpk")
    p.add_argument('--threshold', type=float, default=0.55)
    p.add_argument('--iou', type=float, default=0.65)
    p.add_argument('--stats',    action='store_true',
                   help="Overlay FPS, CPU% and MEM% on camera view")
    args = p.parse_args()

    # --- Camera + AI Model Init (one time) ---
    try:
        imx500     = IMX500(args.model)
        intrinsics = imx500.network_intrinsics
        picam2     = Picamera2(imx500.camera_num)
        cfg        = picam2.create_preview_configuration(
                       main={"size": (640,480), "format": "RGB888"})
        picam2.configure(cfg)
        # this blocks until the network FW is loaded (v3 did this):
        imx500.show_network_fw_progress_bar()
        time.sleep(2.0)    # give it a moment
        picam2.start()

    except Exception as e:
        logging.critical(f"Camera/AI init failed: {e}")
        sys.exit(1)

    # Setup
    recording_event.set() if args.record else recording_event.clear()
    session_name = datetime.now().strftime("%Y%m%d_%H%M%S_") + args.session
    session_path = os.path.join(RECORD_DIR, session_name)
    if args.record:
        os.makedirs(session_path, exist_ok=True)

    # Queues for display
    disp_queue = queue.Queue(maxsize=10) if args.display else None
    # set up display queues
    radar_disp_queue = queue.Queue(maxsize=5)  if args.display else None
    if args.display:
        sensor_info = DeviceFmcw().get_sensor_information()
        num_rx      = sensor_info["num_rx_antennas"]
        radar_plot = RadarDraw(RADAR_MAX_SPEED_M_S,
                               RADAR_MAX_RANGE_M,
                               num_rx, dash_event, display_is_enabled=True)
        radar_plot.setup_plot()

    # Start threads
    threads = []
    threads = [
      RadarThread(radar_disp_queue, record_q, dash_event, display=args.display),
      CameraThread(disp_queue, record_q, dash_event,
                   display=args.display, stats=args.stats,
                   model_path=args.model,
                   threshold=args.threshold,
                   iou=args.iou)
    ]


    if args.record:
        threads.append(RecordThread(record_q, session_path))
    for t in threads: t.start()

    # Main loop for display
    start = time.time()
    try:
        while not dash_event.is_set():

            if args.display:
                try:
                    typ, frame = disp_queue.get(timeout=0.1)
                    if typ == "camera":
                        cv2.imshow("Camera AI (Live)", frame)

                except queue.Empty:
                    pass

                # radar matplotlib Range–Doppler
                try:
                    upd = radar_disp_queue.get_nowait()
                    if upd["type"] == "data":
                        radar_plot.draw(upd["data"])
                except queue.Empty:
                    pass

                # quit key
                if cv2.waitKey(1) == ord('q'):
                    dash_event.set()
                    break

    except KeyboardInterrupt:
        dash_event.set()
    finally:
        for t in threads: t.join(timeout=5)
        cv2.destroyAllWindows()
        logging.info("Shutdown complete")

if __name__ == '__main__':
    main()
