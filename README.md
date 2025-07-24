# Video Processing and Annotation Pipeline with YOLOv7 TensorRT

This project provides a comprehensive pipeline for processing videos using a YOLOv7 model optimized with NVIDIA TensorRT, generating short video clips from detected objects, and preparing annotations for Label Studio. The pipeline includes utilities for bounding box manipulation and video frame extraction based on labeled data.

## Overview

The pipeline is designed to:
1. Detect objects in videos using a YOLOv7 TensorRT model (`yolov7_trt.py`).
2. Manipulate bounding boxes with operations like cropping, resizing, squaring, and expanding (`boundbox_algo.py`).
3. Generate JSON annotations from YOLOv7 detections (`yolo_detect_to_json.py`).
4. Create 24-frame video clips from JSON annotations with configurable bounding box adjustments (`json_to_24frame_video.py`).
5. Extract and organize frames from videos based on class labels for dataset creation (`generate_tsm_dataset.py`).

## Prerequisites

- **Hardware**: NVIDIA GPU with CUDA and TensorRT support.
- **Software**:
  - Python 3.8 or higher.
  - FFmpeg installed and accessible in the system PATH.
  - Required Python packages:
    ```bash
    pip install numpy opencv-python pycuda tensorrt tqdm
    ```
- **Files**:
  - A pre-trained YOLOv7 TensorRT engine file (`.engine`).
  - Label Studio JSON annotation files for `generate_tsm_dataset.py`.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install FFmpeg:
   - **Ubuntu**: `sudo apt-get install ffmpeg`
   - **macOS**: `brew install ffmpeg`
   - **Windows**: Download from [FFmpeg website](https://ffmpeg.org/download.html) and add to PATH.

## Usage

### 1. Object Detection with YOLOv7 TensorRT

The `yolov7_trt.py` script provides a `YoLov7_TRT` class for running inference on video frames using a TensorRT-optimized YOLOv7 model. It filters detections for a specific class (e.g., "Suspicious", class ID 2).

**Example**:
```python
from yolov7_trt import YoLov7_TRT
import cv2

# Initialize the model
model = YoLov7_TRT(engine_file_path="engine_plugin/13062025_childabuse_re.engine", conf_threshold=0.75, iou_threshold=0.2)

# Load an image
image = cv2.imread("sample.jpg")

# Run inference
boxes, scores, class_ids, inference_time = model.infer(image)

# Draw bounding boxes
for box, score, class_id in zip(boxes, scores, class_ids):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.imwrite("output.jpg", image)

# Clean up
model.destroy()
```

### 2. Bounding Box Manipulation

The `boundbox_algo.py` script provides functions to adjust bounding boxes:
- `expand_crop_bbox`: Expands or shrinks a bounding box around its center.
- `resize_bbox`: Resizes a bounding box, optionally cropping to maintain a 1:2 width-to-height ratio and making it square.
- `crop_upper_bbox`: Crops the upper part of a bounding box if its height exceeds a specified width ratio.
- `make_square_bbox`: Adjusts a bounding box to be square, centered on the original box.
- `adjust_to_even_dimensions`: Ensures bounding box dimensions are even for video encoding compatibility.

### 3. Generating JSON Annotations

The `yolo_detect_to_json.py` script processes videos to generate JSON annotations for detected objects using YOLOv7 TensorRT.

**Command-line Arguments**:
- `--yolov7_engine_file_path`: Path to the YOLOv7 TensorRT engine file.
- `--plugin_library`: Path to the TensorRT plugin library.
- `--temporal_size`: Number of frames to buffer (default: 24).
- `--input_video_path`: Path to a single input video (optional).
- `--video_folder_path`: Path to a folder containing videos (optional).
- `--dataset_path`: Directory to save JSON annotations (default: `output_json`).
- `--frame_interval`: Minimum frame interval between JSON outputs (default: 60).

**Example Command**:
```bash
python yolo_detect_to_json.py \
  --yolov7_engine_file_path engine_plugin/13062025_childabuse_re.engine \
  --plugin_library engine_plugin/libmyplugins.so \
  --video_folder_path /path/to/videos \
  --dataset_path output_json \
  --temporal_size 24 \
  --frame_interval 60
```

**Output**:
- JSON files in `dataset_path/annotations/` with the format:
  ```json
  {
    "detection": [{"value": {"x": 100, "y": 200, "width": 300, "height": 400}}],
    "video_name": "video.mp4",
    "frame": 123
  }
  ```

### 4. Generating Short Video Clips

The `json_to_24frame_video.py` script generates 24-frame video clips from JSON annotations, applying bounding box adjustments and frame selection modes (`past`, `future`, or `hybrid`).

**Command-line Arguments**:
- `--video_folder_path`: Path to input videos.
- `--dataset_path`: Directory containing JSON annotations.
- `--output_video_path`: Directory for output video clips.
- `--temporal_size`: Number of frames per clip (default: 24).
- `--square`: Make bounding boxes square.
- `--upper_crop`: Crop the upper part of bounding boxes.
- `--bbox_resize_factor`: Resize bounding boxes by a factor (default: 1.0).
- `--crop_ratio`: Height-to-width ratio threshold for cropping (default: 1.6).
- `--expand_crop_factor`: Expand or shrink bounding boxes (default: 1.0).
- `--frame_interval`: Minimum frame interval between clips (default: 60).
- `--frame_direction`: Frame selection mode (`past`, `future`, or `hybrid`).
- `--skip_reencode`: Skip re-encoding input videos to H.264.
- `--reencode_dir`: Directory for re-encoded videos (default: system temp directory).

**Example Command**:
```bash
python json_to_24frame_video.py \
  --video_folder_path /path/to/videos \
  --dataset_path output_json \
  --output_video_path output_videos \
  --temporal_size 24 \
  --square \
  --upper_crop \
  --bbox_resize_factor 1.2 \
  --crop_ratio 1.6 \
  --expand_crop_factor 1.1 \
  --frame_interval 60 \
  --frame_direction hybrid
```

**Output**:
- Short video clips (e.g., `video_frame000123.mp4`) in `output_video_path`.
- A `tasks.json` file for Label Studio import with pre-annotations.

### 5. Creating a Dataset from Labeled Videos

The `generate_tsm_dataset.py` script extracts 24 frames from videos and organizes them into class-based subfolders using Label Studio JSON labels.

**Command-line Arguments**:
- `--video_folder`: Path to input videos.
- `--output_folder`: Root folder for image subfolders.
- `--json_path`: Path to Label Studio JSON file.
- `--expected_frames`: Number of frames to extract per video (default: 24).

**Example Command**:
```bash
python generate_tsm_dataset.py \
  --video_folder /path/to/videos \
  --output_folder /path/to/image_folders \
  --json_path project-1205-at-2025-07-09-02-55-b0f4c6c8.json \
  --expected_frames 24
```

**Output**:
- Subfolders in `output_folder` named `cls<class_id>_<video_name>` containing `img_001.jpg` to `img_024.jpg`.

## Notes

- **JSON Annotation Format**: Ensure JSON files follow the expected structure for `json_to_24frame_video.py` and `generate_tsm_dataset.py`.
- **Video Re-encoding**: Videos are re-encoded to H.264 by default for stability; use `--skip_reencode` to skip this step.
- **Codec Fallback**: The pipeline uses `h264_nvenc` for faster encoding with NVIDIA GPUs, falling back to `libx264` if needed.
- **Label Studio Integration**: Generated videos and `tasks.json` are compatible with Label Studio for further annotation.
- **Class Labels**: For `generate_tsm_dataset.py`, class labels are mapped as `normal: 0`, `hitting: 1`, `shaking: 2`.

## Troubleshooting

- **FFmpeg Errors**: Verify FFmpeg is installed and in the system PATH.
- **TensorRT Errors**: Ensure the `.engine` file is compatible with your TensorRT and CUDA versions.
- **Insufficient Frames**: Adjust `--frame_direction` or verify input video length.
- **Directory Permissions**: Ensure `dataset_path` and `output_video_path` are writable.
- **Invalid JSON Entries**: Check Label Studio JSON files for missing or invalid `video` or `choice` fields.

## License

This project is licensed under the MIT License.
