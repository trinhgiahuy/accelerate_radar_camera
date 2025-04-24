#!/usr/bin/env python3

import argparse
import sys
import time
import threading
import queue
from functools import lru_cache

# --- Matplotlib (Radar) Imports ---
import matplotlib.pyplot as plt
import numpy as np

# --- Radar SDK Imports ---
from ifxradarsdk import get_version_full
from ifxradarsdk.fmcw import DeviceFmcw
from ifxradarsdk.fmcw.types import FmcwSimpleSequenceConfig, FmcwMetrics
from helpers.DopplerAlgo import DopplerAlgo # Make sure helpers/DopplerAlgo.py is accessible

# --- Picamera2 (Camera) Imports ---
import cv2
import psutil
from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500 # Assuming IMX500 class handles device specifics
# Note: Removed specific device import as Picamera2 handles it based on camera_num
from picamera2.devices.imx500 import (NetworkIntrinsics,
                                       postprocess_nanodet_detection)
from picamera2.devices.imx500.postprocess import scale_boxes
# =================================================
# Shared Resources & Configuration
# =================================================
radar_queue = queue.Queue(maxsize=2) # Queue for Doppler maps
camera_queue = queue.Queue(maxsize=2) # Queue for processed camera frames
stop_event = threading.Event() # Event to signal threads to stop

# --- Camera AI Settings ---
AI_MODEL_PATH = "./imx500-models-backup/imx500_network_yolov8n_pp.rpk"
AI_THRESHOLD = 0.55
AI_IOU = 0.65
AI_MAX_DETECTIONS = 10
AI_LOG_DETECTIONS = True # Enable logging detections
AI_PRINT_OUT = False
AI_TARGET_FPS = 30 # Target frame rate for OpenCV preview window update delay

# --- Radar Settings ---
RADAR_NFRAMES = 0 # Run indefinitely until stopped
RADAR_FRATE = 5 # Hz
RADAR_RANGE_RES_M = 0.15
RADAR_MAX_RANGE_M = 4.8
RADAR_MAX_SPEED_M_S = 2.45
RADAR_SPEED_RES_M_S = 0.2
RADAR_CENTER_FREQ_HZ = 60_750_000_000

# Global references (will be initialized in main)
imx500 = None
intrinsics = None
picam2 = None
last_camera_results = []

# =================================================
# Radar Processing and Visualization
# =================================================

class RadarDraw:
    """Modified Draw class for radar visualization using Matplotlib."""
    def __init__(self, max_speed_m_s, max_range_m, num_ant, stop_signal_event):
        self._h = []
        self._max_speed_m_s = max_speed_m_s
        self._max_range_m = max_range_m
        self._num_ant = num_ant
        self._fig = None
        self._ax = None
        self._is_window_open = False
        self._stop_event = stop_signal_event # Reference to the main stop event

    def setup_plot(self):
        """Sets up the Matplotlib figure and axes. Call this from the main thread."""
        plt.ion()
        self._fig, ax = plt.subplots(nrows=1, ncols=self._num_ant, figsize=((self._num_ant + 1) // 2, 2))
        if self._num_ant == 1:
            self._ax = [ax]
        else:
            self._ax = ax

        self._fig.canvas.manager.set_window_title("Range-Doppler Map")
        self._fig.set_size_inches(3 * self._num_ant + 1, 3 + 1 / self._num_ant)
        # Connect close event to the main stop signal
        self._fig.canvas.mpl_connect('close_event', self.handle_close)
        self._is_window_open = True
        print("Radar plot initialized.")

    def _draw_first_time(self, data_all_antennas):
        if not self._is_window_open or not plt.fignum_exists(self._fig.number):
             print("Radar plot window closed or not ready for first draw.")
             return # Don't draw if window is closed

        minmin = min([np.min(data) for data in data_all_antennas])
        maxmax = max([np.max(data) for data in data_all_antennas])

        for i_ant in range(self._num_ant):
            data = data_all_antennas[i_ant]
            h = self._ax[i_ant].imshow(
                data,
                vmin=minmin, vmax=maxmax,
                cmap='hot',
                extent=(-self._max_speed_m_s, self._max_speed_m_s,
                        0, self._max_range_m),
                origin='lower',
                interpolation='nearest', # Often better for heatmaps
                aspect='auto' # Adjust aspect ratio
            )
            self._h.append(h)
            self._ax[i_ant].set_xlabel("Velocity (m/s)")
            self._ax[i_ant].set_ylabel("Distance (m)")
            self._ax[i_ant].set_title(f"Antenna #{i_ant}")

        self._fig.subplots_adjust(right=0.8)
        cbar_ax = self._fig.add_axes([0.85, 0.1, 0.03, 0.8]) # Adjust position/size as needed
        cbar = self._fig.colorbar(self._h[0], cax=cbar_ax)
        cbar.ax.set_ylabel("Magnitude (dB)")
        self._fig.tight_layout(rect=[0, 0, 0.83, 1]) # Adjust layout to prevent overlap with colorbar

    def _draw_next_time(self, data_all_antennas):
        if not self._is_window_open or not plt.fignum_exists(self._fig.number):
            print("Radar plot window closed or not ready for next draw.")
            return # Don't draw if window is closed

        minmin = min([np.min(data) for data in data_all_antennas]) # Update limits dynamically
        maxmax = max([np.max(data) for data in data_all_antennas])

        for i_ant in range(self._num_ant):
            data = data_all_antennas[i_ant]
            self._h[i_ant].set_data(data)
            self._h[i_ant].set_clim(vmin=minmin, vmax=maxmax) # Update color limits


    def draw(self, data_all_antennas):
        """Draws the plot. Called from the main thread."""
        if not self._is_window_open or not self._fig or not plt.fignum_exists(self._fig.number):
            # print("Radar plot window closed or not ready.")
            return # Exit if window isn't ready or is closed

        try:
            if not self._h:  # First run
                self._draw_first_time(data_all_antennas)
            else:
                self._draw_next_time(data_all_antennas)

            # self._fig.canvas.draw_idle() # Request redraw
            # self._fig.canvas.flush_events() # Process events
            plt.pause(0.001) # Crucial for updating plot in non-blocking manner

        except Exception as e:
            print(f"Error during radar plot drawing: {e}")
            self.close() # Close on error

    def handle_close(self, event=None):
        """Handles the close event from Matplotlib."""
        print("Radar window close requested.")
        self._is_window_open = False
        self._stop_event.set() # Signal the main thread/other threads to stop
        # Don't close the figure here, let the main thread handle cleanup

    def close(self):
        """Closes the plot resources. Called from the main thread during cleanup."""
        if self._is_window_open:
            self._is_window_open = False
            print("Closing radar plot window.")
            plt.close(self._fig)
            plt.close('all') # Ensure all figs are closed


def linear_to_dB(x):
    """Convert linear scale to dB."""
    # Handle potential log10(0)
    x_abs = np.abs(x)
    x_db = 20 * np.log10(np.maximum(x_abs, 1e-12)) # Use a small floor value
    return x_db

def radar_worker(r_queue, stop_flag):
    """Thread function for acquiring and processing radar data."""
    print("Radar thread started.")
    try:
        with DeviceFmcw() as device:
            print(f"Radar SDK Version: {get_version_full()}")
            print(f"Sensor: {device.get_sensor_type()}")

            num_rx_antennas = device.get_sensor_information()["num_rx_antennas"]

            # Configure metrics
            metrics = FmcwMetrics(
                range_resolution_m=RADAR_RANGE_RES_M,
                max_range_m=RADAR_MAX_RANGE_M,
                max_speed_m_s=RADAR_MAX_SPEED_M_S,
                speed_resolution_m_s=RADAR_SPEED_RES_M_S,
                center_frequency_Hz=RADAR_CENTER_FREQ_HZ,
            )

            # Create acquisition sequence
            sequence = device.create_simple_sequence(FmcwSimpleSequenceConfig())
            sequence.loop.repetition_time_s = 1 / RADAR_FRATE

            chirp_loop = sequence.loop.sub_sequence.contents
            device.sequence_from_metrics(metrics, chirp_loop)

            chirp = chirp_loop.loop.sub_sequence.contents.chirp
            chirp.sample_rate_Hz = 1_000_000
            chirp.rx_mask = (1 << num_rx_antennas) - 1
            chirp.tx_mask = 1
            chirp.tx_power_level = 31
            chirp.if_gain_dB = 33
            chirp.lp_cutoff_Hz = 500_000
            chirp.hp_cutoff_Hz = 80_000

            device.set_acquisition_sequence(sequence)

            doppler_algo = DopplerAlgo(chirp.num_samples, chirp_loop.loop.num_repetitions, num_rx_antennas)

            # Send necessary info for plotting to main thread (optional, could be hardcoded too)
            # plot_info = {'max_speed': metrics.max_speed_m_s, 'max_range': metrics.max_range_m, 'num_ant': num_rx_antennas}
            # r_queue.put({'type': 'info', 'data': plot_info})

            frame_count = 0
            while not stop_flag.is_set():
                try:
                    frame_contents = device.get_next_frame(timeout_ms=1000) # Add timeout
                    if frame_contents is None:
                        print("Radar frame timeout")
                        time.sleep(0.1) # Avoid busy-waiting on timeout
                        continue

                    frame_data = frame_contents[0]
                    data_all_antennas = []
                    for i_ant in range(num_rx_antennas):
                        mat = frame_data[i_ant, :, :]
                        # Compute Doppler map and convert to dB
                        doppler_map = doppler_algo.compute_doppler_map(mat, i_ant)
                        dfft_dbfs = linear_to_dB(doppler_map)
                        data_all_antennas.append(dfft_dbfs)

                    # Put the processed data into the queue
                    try:
                        r_queue.put({'type': 'data', 'data': data_all_antennas, 'num_ant': num_rx_antennas}, block=False) # Non-blocking put
                    except queue.Full:
                        # If queue is full, discard oldest frame and try again or just skip
                        try:
                            r_queue.get_nowait() # Discard oldest
                            r_queue.put({'type': 'data', 'data': data_all_antennas, 'num_ant': num_rx_antennas}, block=False)
                        except queue.Empty:
                            pass # Should not happen right after Full, but safety first
                        except queue.Full:
                            print("Radar queue still full, dropping frame.")


                    frame_count += 1
                    if RADAR_NFRAMES > 0 and frame_count >= RADAR_NFRAMES:
                         print("Radar reached target frame count.")
                         break

                    # Optional small sleep to yield control if needed, though get_next_frame might yield
                    # time.sleep(0.01)

                except Exception as e:
                    print(f"Error in radar loop: {e}")
                    time.sleep(0.5) # Avoid spamming errors

    except Exception as e:
        print(f"Failed to initialize or run radar: {e}")
    finally:
        print("Radar thread finished.")
        # Signal main thread to stop if radar fails critically
        stop_flag.set()


# =================================================
# Camera AI Processing
# =================================================

class Detection:
    """Represents a single detection."""
    def __init__(self, coords, category, conf, metadata, cam_instance, net_intrinsics):
        self.category = category
        self.conf = conf
        # Use the passed instances for coordinate conversion
        self.box = imx500.convert_inference_coords(coords, metadata, cam_instance)

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
    """Draw the detections onto the frame."""
    labels = get_labels(net_intrinsics)
    person_detected = False
    for detection in detection_results:
        try:
            x, y, w, h = detection.box
            label_index = int(detection.category)
            if 0 <= label_index < len(labels):
                 label_text = f"{labels[label_index]} ({detection.conf:.2f})"
                 # Check if the detected label is 'person'
                 if labels[label_index].lower() == 'person':
                     person_detected = True
                     color = (0, 0, 255) # Red for person
                 else:
                     color = (0, 255, 0) # Green for others

                 # Draw detection box
                 cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                 # Draw label background
                 (label_width, label_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                 cv2.rectangle(frame, (x, y - label_height - baseline), (x + label_width, y), color, cv2.FILLED)
                 # Draw label text
                 cv2.putText(frame, label_text, (x, y - baseline//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1) # White text
            else:
                print(f"Warning: Invalid category index {label_index} detected.")
        except Exception as e:
            print(f"Error drawing detection: {e}, Box: {detection.box}")
    return person_detected # Return whether a person was detected in this frame


def camera_worker(c_queue, stop_flag):
    """Thread function for capturing camera frames and running AI."""
    global picam2, imx500, intrinsics # Access global instances initialized in main
    print("Camera thread started.")

    if not picam2 or not imx500 or not intrinsics:
        print("Camera components not initialized!")
        stop_flag.set()
        return

    frame_count = 0
    start_time = time.time()
    labels = get_labels(intrinsics) # Cache labels

    while not stop_flag.is_set():
        try:
            # Capture frame and metadata together for better sync
            # Request metadata to ensure NN results are included
            frame_array = picam2.capture_array("main") # Capture from the 'main' stream
            metadata = picam2.capture_metadata()

            if frame_array is None:
                 print("Failed to capture camera frame.")
                 time.sleep(0.1)
                 continue

            # Process detections using the captured metadata
            current_results = parse_detections(metadata, picam2, intrinsics)

            # --- Drawing and Stats ---
            # Work on a copy if needed, but capture_array should provide a new numpy array
            # frame = cv2.cvtColor(frame_array, cv2.COLOR_YUV420p) # Adjust if needed, depends on format
            frame = frame_array # Assume RGB or BGR based on config, adjust cvtColor if needed
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # If capture is BGR, uncomment this

            person_present = draw_detections(frame, current_results, intrinsics)

            # Add FPS and resource info
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent()

            fps_text = f"FPS: {fps:.1f}"
            mem_use_text = f"Mem: {memory.percent:.1f}%"
            cpu_use_text = f"CPU: {cpu_percent:.1f}%"

            cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, mem_use_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
            cv2.putText(frame, cpu_use_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

             # Put the processed frame into the queue
            try:
                c_queue.put(frame, block=False)
            except queue.Full:
                try:
                    c_queue.get_nowait() # Discard oldest frame
                    c_queue.put(frame, block=False)
                except queue.Empty:
                    pass
                except queue.Full:
                     print("Camera queue still full, dropping frame.")


            # Log detections if enabled
            if AI_LOG_DETECTIONS and person_present: # Log only if person detected
                 print(f"Person detected at {time.strftime('%H:%M:%S')}")
                 # Optional: log details of person detections
                 # for result in current_results:
                 #     if labels[int(result.category)].lower() == 'person':
                 #         print(f"  - Conf: {result.conf:.2f}, Box: {result.box}")

            if AI_PRINT_OUT and frame_count % 30 == 0: # Print stats periodically
                print(f"Cam Stats - FPS: {fps:.1f}, Mem: {memory.percent:.1f}%, CPU: {cpu_percent:.1f}%")

            # time.sleep(0.001) # Small yield

        except Exception as e:
            print(f"Error in camera loop: {e}")
            time.sleep(0.5)

    print("Camera thread finished.")
    # No need to signal stop here unless camera fails critically,
    # rely on main thread or radar thread failure to signal stop.


# =================================================
# Main Execution Logic
# =================================================
if __name__ == "__main__":
    print("Starting Radar-Camera Sync Application...")

    # --- Initialize Camera First (often needs hardware access early) ---
    try:
        print(f"Loading AI model: {AI_MODEL_PATH}")
        imx500 = IMX500(AI_MODEL_PATH) # Must be called before Picamera2
        intrinsics = imx500.network_intrinsics
        print("IMX500 Initialized.")

        picam2 = Picamera2(imx500.camera_num)
        # Configure for preview and NN input - size needs to match model expected input if possible
        # Check model input size using imx500.get_input_size() if needed
        preview_config = picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}, # Use RGB for direct OpenCV display
            controls={}, # Add controls if needed (e.g., AE, AWB)
            queue=False, # Process frames sequentially in the thread
            buffer_count=6 # Reduced buffer count
        )
        picam2.configure(preview_config)
        imx500.show_network_fw_progress_bar() # Show firmware loading progress
        picam2.start()
        print("Picamera2 Started.")
        time.sleep(2) # Allow camera to settle

    except Exception as e:
        print(f"FATAL: Failed to initialize camera or AI model: {e}")
        sys.exit(1)

    # --- Initialize Radar ---
    # Radar initialization is done inside the radar_worker thread using 'with DeviceFmcw()'

    # --- Initialize Radar Plot ---
    # Fetch radar info needed for plotting (assuming BGT60TR13C typical values if SDK info not available yet)
    # These values might be slightly off until the radar thread starts and confirms
    num_rx_antennas_guess = 3 # Typical for BGT60TR13C, will be confirmed by radar thread
    radar_plot = RadarDraw(RADAR_MAX_SPEED_M_S, RADAR_MAX_RANGE_M, num_rx_antennas_guess, stop_event)
    radar_plot.setup_plot() # Setup the plot in the main thread

    # --- Start Worker Threads ---
    radar_thread = threading.Thread(target=radar_worker, args=(radar_queue, stop_event), daemon=True)
    camera_thread = threading.Thread(target=camera_worker, args=(camera_queue, stop_event), daemon=True)

    print("Starting worker threads...")
    radar_thread.start()
    camera_thread.start()

    # --- Main Loop (GUI Updates and Control) ---
    last_radar_antennas = num_rx_antennas_guess
    radar_data = None
    camera_frame = None

    while not stop_event.is_set():
        try:
            # --- Update Radar Plot ---
            try:
                radar_update = radar_queue.get(block=False) # Non-blocking get
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

            # --- Update Camera Window ---
            try:
                camera_frame = camera_queue.get(block=False)
                if camera_frame is not None:
                    cv2.imshow("Camera AI Detection", camera_frame)
            except queue.Empty:
                pass # No new frame is fine

            # --- Handle User Input (OpenCV Window) ---
            key = cv2.waitKey(10) # Process OpenCV events, short wait (ms)
            if key == ord('q') or key == 27: # 'q' or ESC key
                print("Quit key pressed.")
                stop_event.set()
                break

            # --- Check if Matplotlib window was closed (handled by callback now) ---
            # if not radar_plot._is_window_open:
            #    if not stop_event.is_set():
            #        print("Radar plot closed by user.")
            #        stop_event.set()


        except Exception as e:
            print(f"Error in main loop: {e}")
            stop_event.set() # Stop on critical error in main loop

        # Optional small sleep for main thread if CPU usage is high
        # time.sleep(0.005)


    # --- Shutdown Sequence ---
    print("Initiating shutdown...")
    stop_event.set() # Ensure it's set

    print("Waiting for camera thread to finish...")
    camera_thread.join(timeout=5.0) # Wait for thread to finish with timeout
    if camera_thread.is_alive():
        print("Warning: Camera thread did not finish gracefully.")

    print("Waiting for radar thread to finish...")
    radar_thread.join(timeout=5.0)
    if radar_thread.is_alive():
        print("Warning: Radar thread did not finish gracefully.")

    print("Stopping camera...")
    if picam2:
        picam2.stop()
        # picam2.close() # May not be needed if stop handles resource release

    print("Closing windows...")
    cv2.destroyAllWindows()
    radar_plot.close() # Close matplotlib plot cleanly
    plt.close('all') # Belt and braces

    print("Application finished.")
