import sys
from functools import lru_cache
import cv2
import numpy as np
import time
import psutil
from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import (NetworkIntrinsics,
                                      postprocess_nanodet_detection)

# Settings
threshold = 0.55
iou = 0.65
max_detections = 10
save_images = False  # Disable saving images
log_detections = True  # Enable logging detections
print_out = False

frame_rate = 30  # Target frame rate for OpenCV preview

last_detections = []

class Detection:
    def __init__(self, coords, category, conf, metadata):
        """Create a Detection object, recording the bounding box, category and confidence."""
        self.category = category
        self.conf = conf
        self.box = imx500.convert_inference_coords(coords, metadata, picam2)


def parse_detections(metadata: dict):
    """Parse the output tensor into a number of detected objects, scaled to the ISP out."""
    global last_detections
    bbox_normalization = intrinsics.bbox_normalization

    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    input_w, input_h = imx500.get_input_size()
    if np_outputs is None:
        return last_detections

    if intrinsics.postprocess == "nanodet":
        boxes, scores, classes = postprocess_nanodet_detection(
            outputs=np_outputs[0], conf=threshold, iou_thres=iou, max_out_dets=max_detections
        )[0]
        from picamera2.devices.imx500.postprocess import scale_boxes
        boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
    else:
        boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
        if bbox_normalization:
            boxes = boxes / input_h

        boxes = np.array_split(boxes, 4, axis=1)
        boxes = zip(*boxes)

    last_detections = [
        Detection(box, category, score, metadata)
        for box, score, category in zip(boxes, scores, classes)
        if score > threshold
    ]
    return last_detections

@lru_cache
def get_labels():
    labels = intrinsics.labels
    if intrinsics.ignore_dash_labels:
        labels = [label for label in labels if label and label != "-"]
    return labels

def draw_detections(frame):
    """Draw the detections directly on the frame."""
    labels = get_labels()
    for detection in last_results:
        x, y, w, h = detection.box
        label = f"{labels[int(detection.category)]} ({detection.conf:.2f})"

        # Draw detection box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


if __name__ == "__main__":

    model = "./imx500-models-backup/imx500_network_yolov8n_pp.rpk"

    # This must be called before instantiation of Picamera2
    imx500 = IMX500(model)
    intrinsics = imx500.network_intrinsics

    picam2 = Picamera2(imx500.camera_num)

    config = picam2.create_preview_configuration(
        controls={},
        buffer_count=12,
    )
    config = picam2.create_preview_configuration({"size": (640, 480)})

    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=False)

    last_results = None
    labels = get_labels()	

    print("Started! Press 'q' to quit.")
    frame_count = 0
    start_time = time.time()

    while True:
        metadata = picam2.capture_metadata()
        last_results = parse_detections(metadata)

        # Capture frame
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fps = frame_count / (time.time() - start_time)
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent()
        fps_text = f"FPS: {fps:.2f}"
        mem_use_text= f"Memory Usage: {memory.percent}%"
        cpu_use_text = f"CPU Usage: {cpu_percent}%"
        cv2.putText(frame, fps_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255,0), 2)
        cv2.putText(frame, mem_use_text, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(frame, cpu_use_text, (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        draw_detections(frame)

        # Show the live preview using OpenCV
        cv2.imshow("Real-Time Detection", frame)

        # Log detections
        if log_detections and last_results:
            for result in last_results:
                label = f"{labels[int(result.category)]} ({result.conf:.2f})"
                print(f"Detected {label}")
        if print_out and frame_count % 10 == 0:
            print(f"FPS: {fps:.2f}")
            print(f"Memory Usage: {memory.percent}%")
            print(f"CPU Usage: {cpu_percent}%")

        # Save image if enabled
        if save_images:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            cv2.imwrite(f"frame_{timestamp}.jpg", frame)
        frame_count += 1

        # Break the loop on 'q' key
        if cv2.waitKey(1000 // frame_rate) & 0xFF == ord("q"):
            break

    picam2.stop()
    cv2.destroyAllWindows()
