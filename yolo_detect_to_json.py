import argparse
import ctypes
import cv2
import numpy as np
import json
import os
import logging
import glob
from collections import deque
from tqdm import tqdm
from yolov7_trt import YoLov7_TRT

# Detection and Tracking Thresholds
CONF_THRESH = 0.75
IOU_THRESHOLD = 0.2

# Define Class IDs
CLASS_SUSPICIOUS = 2

def parse_arguments():
    parser = argparse.ArgumentParser(description="Auto generate JSON annotations using YOLOv7")
    parser.add_argument("--yolov7_engine_file_path", default='engine_plugin/13062025_childabuse_re.engine', help="Path to YOLOv7 engine")
    parser.add_argument("--plugin_library", default="engine_plugin/libmyplugins.so", help="Path to plugin library")
    parser.add_argument("--temporal_size", type=int, default=24, help="Frame sequence length")
    parser.add_argument("--input_video_path", default=None, help="Input video path (for single video)")
    parser.add_argument("--video_folder_path", default=None, help="Path to folder containing videos (for batch processing)")
    parser.add_argument("--dataset_path", default="output_json", help="Output JSON directory")
    parser.add_argument("--frame_interval", type=int, default=60, help="Minimum frame interval between generated JSONs to avoid excessive output")
    return parser.parse_args()

def process_detections(result_boxes, result_scores, result_classid):
    ents = []
    id_counter = 0
    for box, score, cls_id in zip(result_boxes, result_scores, result_classid):
        if score < CONF_THRESH:
            continue
        class_id = int(cls_id)
        id_counter += 1
        original_box = box.tolist()
        ents.append({
            'original_bbox': original_box,
            'class': class_id,
            'score': float(score),
            'id': id_counter
        })
    return ents

def save_json_annotation(ent, video_name, frame_no, dataset_path, ppl_id):
    json_data = {
        "detection": [],
        "video_name": os.path.basename(video_name),
        "frame": frame_no
    }
    # Save original_bbox
    bbox = ent.get('original_bbox')
    x1, y1, x2, y2 = bbox
    detection = {
        "value": {
            "x": int(x1),
            "y": int(y1),
            "width": int(x2 - x1),
            "height": int(y2 - y1)
        }
    }
    json_data["detection"].append(detection)
    json_dir = os.path.join(dataset_path, "annotations")
    os.makedirs(json_dir, exist_ok=True)
    video_name_base = os.path.splitext(os.path.basename(video_name))[0]
    json_filename = f"{video_name_base}_frame{frame_no:06d}_ppl{ppl_id}.json"
    json_path = os.path.join(json_dir, json_filename)
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=4)
    logging.info(f"Saved JSON: {json_path}")

def process_single_video(video_path, args, yolov7_wrapper):
    if video_path is None:
        logging.error("No video path provided")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Failed to open video {video_path}")
        return
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    past_frames = deque(maxlen=args.temporal_size)
    frameno = 0
    detected_id_counter = 1
    last_json_frame = -args.frame_interval

    # Initialize tqdm progress bar
    progress_bar = tqdm(total=total_frames, desc=f"Processing {os.path.basename(video_path)}", unit="frame")

    while True:
        frameno += 1
        ret, frame = cap.read()
        progress_bar.update(1)  # Update progress bar
        if not ret:
            break
        img = cv2.cvtColor(cv2.normalize(frame, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F), cv2.COLOR_BGR2RGB)
        result_boxes, result_scores, result_classid, _ = yolov7_wrapper.infer(img)
        ents = process_detections(result_boxes, result_scores, result_classid)
        past_frames.append((frame.copy(), list(ents)))
        if len(ents) > 0 and len(past_frames) == args.temporal_size:
            if frameno - last_json_frame < args.frame_interval:
                logging.debug(f"Skipping JSON generation for frame {frameno} due to frame interval ({frameno - last_json_frame} < {args.frame_interval})")
                continue
            for ent in ents:
                # Skip non-suspicious class (cls0)
                if ent['class'] != CLASS_SUSPICIOUS:
                    continue
                save_json_annotation(ent, video_path, frameno, args.dataset_path, detected_id_counter)
                detected_id_counter += 1
            last_json_frame = frameno
    progress_bar.close()
    cap.release()

def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_arguments()
    ctypes.CDLL(args.plugin_library)
    yolov7_wrapper = YoLov7_TRT(args.yolov7_engine_file_path, CONF_THRESH, IOU_THRESHOLD)

    if args.video_folder_path:
        if not os.path.isdir(args.video_folder_path):
            logging.error(f"Video folder path {args.video_folder_path} is not a directory")
            return
        video_files = glob.glob(os.path.join(args.video_folder_path, "*.mp4")) + glob.glob(os.path.join(args.video_folder_path, "*.mkv"))
        if not video_files:
            logging.error(f"No videos found in {args.video_folder_path}")
            return
        for video_path in video_files:
            logging.info(f"Processing video: {video_path}")
            process_single_video(video_path, args, yolov7_wrapper)
    elif args.input_video_path:
        logging.info(f"Processing single video: {args.input_video_path}")
        process_single_video(args.input_video_path, args, yolov7_wrapper)
    else:
        logging.error("Must provide either --input_video_path or --video_folder_path")
        return

    yolov7_wrapper.destroy()

if __name__ == "__main__":
    main()
