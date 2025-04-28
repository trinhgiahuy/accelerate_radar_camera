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

def draw_detections(frame, detection_results, net_intrinsics):
    """Draw detections onto frame copy. Returns drawn RGB frame and person count."""
    #if not net_intrinsics_obj: return frame_rgb, 0
    #labels_cache_key = (tuple(net_intrinsics_obj.labels), net_intrinsics_obj.ignore_dash_labels)
    #labels = get_labels(labels_cache_key)
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
            # with DeviceFmcw() as dev:
            #     sequence = dev.create_simple_sequence(FmcwSimpleSequenceConfig())
            #     dev.set_acquisition_sequence(sequence)
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
                        # 1) convert complex→dB: shape (num_rx, Nc, Ns)
                        mag = 20 * np.log10(np.abs(raw) + 1e-9)
                        # 2) pick antenna 0 slice → 2D
                        img2d = mag[0, :, :]             # shape (Nc, Ns)
                        # 3) normalize to 0–255
                        norm = cv2.normalize(img2d, None, 0, 255, cv2.NORM_MINMAX)
                        u8   = norm.astype(np.uint8)
                        # 4) apply a colormap
                        color = cv2.applyColorMap(u8, cv2.COLORMAP_HOT)
                        # 5) tag as radar and queue
                        try:
                            self.display_q.put_nowait(("radar", color))
                        except queue.Full:
                            pass
                    frame_idx = 1
        except Exception:
            logging.exception("[Radar] Error")
        finally:
            logging.info("[Radar] Stopped")

class CameraThread(threading.Thread):
    def __init__(self, display_q, record_q, event, display, model_path, threshold, iou):
        super().__init__(daemon=True)
        self.display_q = display_q
        self.record_q = record_q
        self.stop_event = event
        self.display = display
        self.model_path = model_path
        self.threshold = threshold
        self.iou = iou

    def run(self):
        logging.info("[Camera] Starting")
        # cam = None
        # try:
        #     imu = IMX500(self.model_path)
        #     picam = Picamera2(imu.camera_num)
        #     cfg = picam.create_preview_configuration(main={"size":(640,480),"format":"RGB888"})
        #     picam.configure(cfg)
        #     imu.show_network_fw_progress_bar()      # v3 did this
        #     time.sleep(2.0)                         # give the IMX500 a moment to settle
        #     picam.start()
        try:
            imu   = imx500
            picam = picam2
            frame_idx = 0
            while not self.stop_event.is_set():
                ts = time.time()
                md = picam.capture_metadata()
                arr = picam.capture_array("main")
                # AI inference
                # out = imu.get_outputs(md, add_batch=True)
                # dets = []
                # if out is not None:
                #     try:
                #         boxes, scores, classes = postprocess_nanodet_detection(
                #             out[0],
                #             conf=self.threshold,
                #             iou_thres=self.iou,
                #             max_out_dets=10
                #         )
                #     except Exception as e:
                #         # sometimes the NN produces zero proposals → internal broadcast error
                #         logging.warning(f"postprocess_nanodet_detection failed, treating as no detections: {e}")
                #         boxes, scores, classes = np.empty((0,4)), [], []                    
                    
                #     for b, s, c in zip(boxes, scores, classes):
                #         if s >= self.threshold:
                #             dets.append((list(map(int,b)), float(s), int(c)))
                # Record
                detections = parse_detections(md, picam, imu.network_intrinsics)
                # record
                if recording_event.is_set():
                    dto = [d.to_dict() for d in detections]
                    self.record_q.put_nowait(("camera", frame_idx, ts, arr, dto))

                # display
                # if self.display:
                #     disp_frame, count = draw_detections(arr, detections, intrinsics)
                #     self.display_q.put_nowait(("camera", disp_frame))
                # display (always catch queue.Full)
                if self.display:
                    disp_frame, _ = draw_detections(arr, detections, intrinsics)
                    try:
                        self.display_q.put(("camera", disp_frame), block=False)
                    except queue.Full:
                        # drop oldest
                        try: self.display_q.get(block=False)
                        except queue.Empty: pass
                        # retry once
                        try: self.display_q.put(("camera", disp_frame), block=False)
                        except queue.Full: pass

                # if recording_event.is_set():
                #     try:
                #         dto = [d.to_dict() for d in detections]
                #         self.record_q.put_nowait(("camera", frame_idx, ts, arr, dto))
                #         # self.record_q.put_nowait(("camera", frame_idx, ts, arr, dets))
                #     except queue.Full:
                #         logging.warning("[Camera] Record queue full, dropping")
                # # Display
                # if self.display:
                #     # vis = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                #     # for box, score, cls in dets:
                #     #     x,y,w,h = box
                #     #     cv2.rectangle(vis, (x,y),(x+w,y+h),(0,255,0),2)
                #     display_frame, person_count = draw_detections(arr, detections, imu.network_intrinsics)
                #     self.display_q.put_nowait(("camera", display_frame))
                #     # try:
                #         # self.display_q.put_nowait(vis)
                #     # except queue.Full:
                #         # pass
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

    # Start threads
    threads = []
    threads.append(RadarThread(disp_queue, record_q, dash_event, args.display))
    threads.append(CameraThread(disp_queue, record_q, dash_event, args.display, args.model, args.threshold, args.iou))
    if args.record:
        threads.append(RecordThread(record_q, session_path))
    for t in threads: t.start()

    # Main loop for display
    start = time.time()
    try:
        while not dash_event.is_set():
            now = time.time()
            if args.duration and (now-start)>args.duration:
                dash_event.set(); break
            if args.display:
                # try: frame = disp_queue.get(timeout=0.1)
                # except queue.Empty: continue
                # if isinstance(frame, np.ndarray): cv2.imshow("Live", frame); cv2.waitKey(1)
                try:
                    typ, frame = disp_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if typ == "camera":
                    cv2.imshow("Camera AI (Live)", frame)
                elif typ == "radar":
                    cv2.imshow("Radar (Live)", frame)

                # must call waitKey to update both windows
                if cv2.waitKey(1) in (ord('q'), 27):
                    dash_event.set()
                    break            
            else:
                time.sleep(0.1)
    except KeyboardInterrupt:
        dash_event.set()
    finally:
        for t in threads: t.join(timeout=5)
        cv2.destroyAllWindows()
        logging.info("Shutdown complete")

if __name__ == '__main__':
    main()
