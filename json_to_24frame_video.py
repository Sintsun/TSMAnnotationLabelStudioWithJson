import argparse
import cv2
import json
import os
import glob
import logging
import subprocess
import tempfile
from collections import defaultdict
from tqdm import tqdm
from boundbox_algo import crop_upper_bbox, make_square_bbox, expand_crop_bbox, resize_bbox, adjust_to_even_dimensions

def reencode_input_to_h264(input_path, reencode_dir):
    """Re-encode input video to H.264 for stability"""
    if reencode_dir is None:
        reencode_dir = tempfile.gettempdir()
    os.makedirs(reencode_dir, exist_ok=True)
    temp_output = os.path.join(reencode_dir, os.path.basename(input_path) + '_reencoded.mp4')
    cmd = [
        'ffmpeg', '-err_detect', 'ignore_err', '-i', input_path,
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-c:a', 'aac', '-y', temp_output
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            logging.info(f"Re-encoded {input_path} to H.264: {temp_output}")
            return temp_output
        else:
            logging.error(f"Failed to re-encode {input_path}: {result.stderr}")
            return input_path
    except Exception as e:
        logging.error(f"Error during re-encoding {input_path}: {e}")
        return input_path

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate short videos from JSON using FFmpeg")
    parser.add_argument("--video_folder_path", default="/path/to/videos", help="Video folder path")
    parser.add_argument("--dataset_path", default="output_json", help="Output JSON directory")
    parser.add_argument("--output_video_path", default="output_videos", help="Output short video directory")
    parser.add_argument("--temporal_size", type=int, default=24, help="Short video frame count")
    parser.add_argument("--square", action="store_true", help="Make bounding box square")
    parser.add_argument("--upper_crop", action="store_true", help="Enable upper body crop")
    parser.add_argument("--bbox_resize_factor", type=float, default=1.0, help="Bounding box resize factor")
    parser.add_argument("--crop_ratio", type=float, default=1.6, help="Crop ratio threshold")
    parser.add_argument("--expand_crop_factor", type=float, default=1.0, help="Expand crop factor")
    parser.add_argument("--frame_interval", type=int, default=60,
                        help="Minimum frame interval between generated videos to avoid excessive output")
    parser.add_argument("--frame_direction", type=str, default="future", choices=["past", "future", "hybrid"],
                        help="Direction of frames to crop: 'past' for previous frames, 'future' for upcoming frames, "
                             "'hybrid' for half past and half future frames (default: future)")
    parser.add_argument("--skip_reencode", action="store_true", help="Skip reencoding input videos to save space")
    parser.add_argument("--reencode_dir", type=str, default=None,
                        help="Directory to save re-encoded videos (if not skipping reencode). Defaults to system temp dir.")
    return parser.parse_args()

def check_directory_writable(directory):
    """Check if a directory is writable, create it if it doesn't exist"""
    try:
        os.makedirs(directory, exist_ok=True)
        temp_file = os.path.join(directory, ".test_write")
        with open(temp_file, 'w') as f:
            f.write("test")
        os.remove(temp_file)
        return True
    except (PermissionError, OSError) as e:
        logging.error(f"Directory {directory} is not writable: {e}")
        return False

def collect_frames(cap, frame_no, temporal_size, frame_direction):
    frames_to_write = []
    if frame_direction == "past":
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_no - temporal_size))
        for _ in range(temporal_size):
            ret, frame = cap.read()
            if ret:
                frames_to_write.append(frame.copy())
    elif frame_direction == "future":
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        for _ in range(temporal_size):
            ret, frame = cap.read()
            if ret:
                frames_to_write.append(frame.copy())
            else:
                break
    elif frame_direction == "hybrid":
        half = temporal_size // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_no - half))
        for _ in range(temporal_size):
            ret, frame = cap.read()
            if ret:
                frames_to_write.append(frame.copy())
            else:
                break
    if len(frames_to_write) < temporal_size:
        logging.warning(f"Insufficient frames for direction {frame_direction} at frame {frame_no}")
    return frames_to_write

def write_video_with_ffmpeg(frames, x1, y1, x2, y2, output_path, fps):
    clip_width, clip_height = adjust_to_even_dimensions(x2 - x1, y2 - y1)
    cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{clip_width}x{clip_height}',
        '-pix_fmt', 'bgr24',
        '-r', str(fps),
        '-i', '-',
        '-an',
        '-c:v', 'h264_nvenc',
        '-preset', 'slow',
        '-b:v', '5M',
        '-pix_fmt', 'yuv420p',
        output_path
    ]
    all_data = b''
    for frame in frames:
        if y2 > frame.shape[0] or x2 > frame.shape[1]:
            continue
        region = frame[y1:y2, x1:x2]
        if region.shape[0] != clip_height or region.shape[1] != clip_width:
            region = cv2.resize(region, (clip_width, clip_height))
        all_data += region.tobytes()
    if not all_data:
        logging.warning(f"No valid data for {output_path}")
        return False
    try:
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = p.communicate(input=all_data)
        if p.returncode != 0:
            logging.warning("h264_nvenc failed, trying libx264")
            cmd = [
                'ffmpeg', '-y',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-s', f'{clip_width}x{clip_height}',
                '-pix_fmt', 'bgr24',
                '-r', str(fps),
                '-i', '-',
                '-an',
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-pix_fmt', 'yuv420p',
                output_path
            ]
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = p.communicate(input=all_data)
            if p.returncode != 0:
                logging.error(f"FFmpeg failed for {output_path}: {stderr.decode()}")
                return False
        logging.info(f"Generated short video: {output_path}")
        return True
    except Exception as e:
        logging.error(f"Error writing video {output_path}: {e}")
        return False

def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_arguments()

    # Check if dataset_path and output_video_path are writable
    if not check_directory_writable(args.dataset_path):
        logging.error(f"Cannot proceed: {args.dataset_path} is not writable")
        return
    if not check_directory_writable(args.output_video_path):
        logging.error(f"Cannot proceed: {args.output_video_path} is not writable")
        return

    annotations_dir = os.path.join(args.dataset_path, 'annotations')
    json_files = glob.glob(os.path.join(annotations_dir, "*.json"))
    if not json_files:
        logging.error(f"No JSON files found in {annotations_dir}")
        return

    # Group JSON by video_name
    video_annotations = defaultdict(list)
    tasks = []  # Collect tasks for Label Studio import
    for json_path in json_files:
        with open(json_path, 'r') as f:
            data = json.load(f)
        video_name = data['video_name']
        frame_no = data['frame']
        video_annotations[video_name].append({'path': json_path, 'data': data, 'frame_no': frame_no})

    # Initialize tqdm progress bar for JSON files
    progress_bar = tqdm(total=len(json_files), desc="Processing JSON files", unit="file")

    for video_name, annos in video_annotations.items():
        # Sort by frame_no
        annos.sort(key=lambda x: x['frame_no'])
        video_path = os.path.join(args.video_folder_path, video_name)
        if not os.path.exists(video_path):
            logging.error(f"Video {video_path} not found, skipping")
            progress_bar.update(len(annos))
            continue
        if args.skip_reencode:
            reencoded_path = video_path
        else:
            reencoded_path = reencode_input_to_h264(video_path, args.reencode_dir)

        cap = cv2.VideoCapture(reencoded_path)
        if not cap.isOpened():
            logging.error(f"Failed to open video {reencoded_path}")
            progress_bar.update(len(annos))
            continue

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        last_video_frame = -args.frame_interval

        for anno in annos:
            data = anno['data']
            frame_no = data['frame']
            if frame_no - last_video_frame < args.frame_interval:
                logging.debug(f"Skipping frame {frame_no} due to interval ({frame_no - last_video_frame} < {args.frame_interval})")
                progress_bar.update(1)
                continue

            detection = data['detection'][0]['value']
            x1, y1 = detection['x'], detection['y']
            x2, y2 = x1 + detection['width'], y1 + detection['height']
            bbox = [x1, y1, x2, y2]
            if args.upper_crop:
                bbox = crop_upper_bbox(bbox, width, height, args.crop_ratio)
            if args.expand_crop_factor != 1.0:
                bbox = expand_crop_bbox(bbox, width, height, args.expand_crop_factor)
            if args.square:
                bbox = make_square_bbox(bbox, width, height)
            elif args.bbox_resize_factor != 1.0:
                bbox = resize_bbox(bbox, args.bbox_resize_factor, width, height)
            x1, y1, x2, y2 = map(int, bbox)

            frames_to_write = collect_frames(cap, frame_no, args.temporal_size, args.frame_direction)
            if len(frames_to_write) < args.temporal_size:
                progress_bar.update(1)
                continue

            video_filename = f"{os.path.splitext(video_name)[0]}_frame{frame_no:06d}.mp4"
            video_out_path = os.path.join(args.output_video_path, video_filename)
            if write_video_with_ffmpeg(frames_to_write, x1, y1, x2, y2, video_out_path, fps):
                # Add to Label Studio tasks.json with pre-annotation
                key_frame = (args.temporal_size // 2) + 1 if args.frame_direction == "hybrid" else 1
                task = {
                    "data": {
                        "video": os.path.abspath(video_out_path)  # Absolute path for local import
                    },
                    "predictions": [
                        {
                            "model_version": "yolo_auto",
                            "result": [
                                {
                                    "from_name": "label",
                                    "to_name": "video",
                                    "type": "rectanglelabels",
                                    "value": {
                                        "frame": key_frame,
                                        "x": (x1 / width) * 100,
                                        "y": (y1 / height) * 100,
                                        "width": ((x2 - x1) / width) * 100,
                                        "height": ((y2 - y1) / height) * 100,
                                        "rotation": 0,
                                        "rectanglelabels": ["Suspicious"]
                                    }
                                }
                            ]
                        }
                    ]
                }
                tasks.append(task)
            last_video_frame = frame_no
            progress_bar.update(1)

        cap.release()

    # Save tasks.json for Label Studio import
    tasks_path = os.path.join(args.output_video_path, 'tasks.json')
    with open(tasks_path, 'w') as f:
        json.dump(tasks, f, indent=4)
    logging.info(f"Generated Label Studio tasks.json: {tasks_path}")

    progress_bar.close()

if __name__ == "__main__":
    main()
