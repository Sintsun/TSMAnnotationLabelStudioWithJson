import os
import cv2
import json
import logging
import argparse
import glob
import re
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('video_to_images.log'),
        logging.StreamHandler()
    ]
)

# Class mapping
CLASS_MAP = {
    'normal': 0,  # cls0
    'hitting': 1, # cls1
    'shaking': 2  # cls2
}

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Extract frames from videos and organize by class from Label Studio JSON files.')
    parser.add_argument('--video_folder', type=str, required=True,
                        help='Path to folder containing video files (e.g., /path/to/videos)')
    parser.add_argument('--output_folder', type=str, required=True,
                        help='Root folder to save image subfolders (e.g., /path/to/image_folders)')
    parser.add_argument('--json_folder', type=str, required=True,
                        help='Path to folder containing Label Studio JSON files (e.g., /path/to/json_files)')
    parser.add_argument('--expected_frames', type=int, default=24,
                        help='Expected number of frames to extract per video (default: 24)')
    return parser.parse_args()

def normalize_video_name(video_name):
    """Normalize video filename by removing prefixes and extracting core name."""
    # Remove prefix before hyphen
    if '-' in video_name:
        video_name = video_name.split('-', 1)[1]
    # Remove additional prefixes (e.g., 'c3_', 'c4_', '1_', 'r13_')
    prefixes = ['c3_', 'c4_', '1_', 'r13_']
    for prefix in prefixes:
        if video_name.startswith(prefix):
            video_name = video_name[len(prefix):]
    # Use regex to extract core name (e.g., '2025_02_06__15_32_43_frame004653_ppl25')
    pattern = r'(\d{4}_\d{2}_\d{2}__\d{2}_\d{2}_\d{2}_frame\d+_ppl\d+)'
    match = re.search(pattern, video_name)
    if match:
        core_name = match.group(1)
        logging.debug(f"Normalized video name: {video_name} -> {core_name}")
        return core_name
    logging.warning(f"Could not normalize video name: {video_name}")
    return video_name

def load_labelstudio_json(json_path):
    """Load a single Label Studio JSON file and return video filename (normalized) to class label mapping."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        video_label_map = {}
        invalid_entries = 0
        total_entries = 0
        for item in data:
            total_entries += 1
            # Check if item is a valid dictionary and has required fields
            if not isinstance(item, dict):
                logging.warning(f"Skipping invalid JSON entry (not a dict) in {json_path}: {item}")
                invalid_entries += 1
                continue
            if 'video' not in item:
                logging.warning(f"Skipping entry with missing 'video' field in {json_path}: {item}")
                invalid_entries += 1
                continue
            if 'choice' not in item:
                logging.warning(f"Skipping entry with missing 'choice' field in {json_path}: {item.get('video', 'unknown')}")
                invalid_entries += 1
                continue
            video_path = item['video']
            choice = item['choice']
            # Validate choice
            if choice not in CLASS_MAP:
                logging.warning(f"Skipping entry with invalid choice '{choice}' for video {video_path} in {json_path}")
                invalid_entries += 1
                continue
            # Extract video filename
            video_name = Path(video_path).name
            # Normalize filename
            normalized_name = normalize_video_name(video_name)
            logging.debug(f"JSON video: {video_name}, normalized: {normalized_name}, label: {choice}")
            video_label_map[normalized_name] = choice
        logging.info(f"Processed {total_entries} entries, loaded {len(video_label_map)} valid video labels from {json_path}")
        if invalid_entries > 0:
            logging.warning(f"Skipped {invalid_entries} invalid entries in {json_path}")
        if not video_label_map:
            logging.error(f"No valid labels found in {json_path}")
        # Log all normalized names for debugging
        logging.debug(f"Normalized JSON video names from {json_path}: {list(video_label_map.keys())}")
        return video_label_map
    except Exception as e:
        logging.error(f"Failed to load JSON file {json_path}: {e}")
        return {}

def get_org_name(video_path):
    """Extract org_name from video filename, removing prefix before hyphen and keeping frame/ppl parts."""
    video_name = Path(video_path).stem
    if '-' in video_name:
        video_name = video_name.split('-', 1)[1]
    return video_name

def extract_frames(video_path, output_subfolder, expected_frames):
    """Extract frames from video and save as images."""
    try:
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Failed to open video: {video_path}")
            return False

        # Get video frame count
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < expected_frames:
            logging.error(f"Video {video_path} has only {total_frames} frames, expected {expected_frames}")
            cap.release()
            return False

        # Create output subfolder
        os.makedirs(output_subfolder, exist_ok=True)
        logging.info(f"Created output subfolder: {output_subfolder}")

        # Extract frames
        frame_count = 0
        saved_frames = 0
        while frame_count < total_frames and saved_frames < expected_frames:
            ret, frame = cap.read()
            if not ret:
                logging.warning(f"Failed to read frame {frame_count + 1} from {video_path}")
                break

            # Save frame as image
            frame_number = saved_frames + 1
            image_path = os.path.join(output_subfolder, f"img_{frame_number:03d}.jpg")
            cv2.imwrite(image_path, frame)
            saved_frames += 1
            logging.debug(f"Saved frame {frame_number} to {image_path}")
            frame_count += 1

        cap.release()

        # Verify saved frames
        if saved_frames != expected_frames:
            logging.error(f"Extracted {saved_frames} frames from {video_path}, expected {expected_frames}")
            return False

        logging.info(f"Successfully extracted {saved_frames} frames to {output_subfolder}")
        return True
    except Exception as e:
        logging.error(f"Error extracting frames from {video_path}: {e}")
        return False

def process_videos(video_folder, output_folder, json_folder, expected_frames):
    """Process all videos in the video folder using class labels from all JSON files in the json_folder."""
    # Find all JSON files in the json_folder
    json_files = glob.glob(os.path.join(json_folder, "*.json"))
    if not json_files:
        logging.error(f"No JSON files found in {json_folder}")
        return

    logging.info(f"Found {len(json_files)} JSON files in {json_folder}")

    # Load all JSON files and merge video label mappings
    video_label_map = {}
    for json_path in json_files:
        labels = load_labelstudio_json(json_path)
        for video_name, label in labels.items():
            if video_name in video_label_map and video_label_map[video_name] != label:
                logging.warning(f"Conflicting labels for video {video_name}: {video_label_map[video_name]} vs {label} in {json_path}")
            video_label_map[video_name] = label

    if not video_label_map:
        logging.error("No valid labels found across all JSON files. Exiting.")
        return

    # Find all video files (assuming .mp4)
    video_paths = glob.glob(os.path.join(video_folder, "*.mp4"))
    if not video_paths:
        logging.error(f"No video files found in {video_folder}")
        return

    logging.info(f"Found {len(video_paths)} videos in {video_folder}")

    # Process each video
    for video_path in video_paths:
        video_name = Path(video_path).name
        # Normalize filename for matching
        match_name = normalize_video_name(video_name)
        logging.info(f"Processing video: {video_name}, normalized: {match_name}")

        # Get class label from JSON
        class_label = video_label_map.get(match_name)
        if not class_label:
            logging.warning(f"No label found for video {match_name}, skipping")
            continue

        class_id = CLASS_MAP.get(class_label)
        if class_id is None:
            logging.warning(f"Invalid class label {class_label} for {match_name}, skipping")
            continue

        # Get org_name from video filename
        org_name = get_org_name(video_path)
        # Create subfolder name (e.g., cls0_temp_r1_2024_11_18__17_02_59_frame000691_ppl3)
        subfolder_name = f"cls{class_id}_{org_name}"
        output_subfolder = os.path.join(output_folder, subfolder_name)

        # Extract frames
        success = extract_frames(video_path, output_subfolder, expected_frames)
        if success:
            logging.info(f"Frame extraction completed for {video_path}")
        else:
            logging.error(f"Frame extraction failed for {video_path}")

def main():
    """Main function to process videos and extract frames."""
    args = parse_arguments()

    # Validate inputs
    if not os.path.exists(args.video_folder):
        logging.error(f"Video folder {args.video_folder} does not exist")
        return
    if not os.path.exists(args.output_folder):
        logging.info(f"Output folder {args.output_folder} does not exist, creating it")
        os.makedirs(args.output_folder, exist_ok=True)
    if not os.path.exists(args.json_folder):
        logging.error(f"JSON folder {args.json_folder} does not exist")
        return

    # Process videos
    process_videos(
        args.video_folder,
        args.output_folder,
        args.json_folder,
        args.expected_frames
    )

if __name__ == "__main__":
    main()