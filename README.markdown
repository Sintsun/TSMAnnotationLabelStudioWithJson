# YOLOv7 TensorRT Video Processing Pipeline

This project provides a pipeline for processing videos using a YOLOv7 model deployed with NVIDIA TensorRT, generating short video clips based on detected objects, and preparing annotations for Label Studio. The pipeline includes bounding box manipulation utilities and supports video re-encoding for stability.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Create TensorRT engine from your PyTorch model:**
   ```bash
   python simple_convert.py --input_model your_model.pt --output_engine your_model.engine
   ```

3. **Test your engine:**
   ```bash
   python test_simple_engine.py
   ```

4. **Process video and generate JSON annotations:**
   ```bash
   python yolo_detect_to_json.py \
     --yolov7_engine_file_path your_model.engine \
     --input_video_path your_video.mp4 \
     --dataset_path output_json
   ```

5. **Generate video clips from annotations:**
   ```bash
   python json_to_24frame_video.py \
     --video_folder_path /path/to/videos \
     --dataset_path output_json \
     --output_video_path output_videos
   ```

## Overview

The pipeline consists of several Python scripts designed to:
1. Perform object detection using a YOLOv7 model optimized with TensorRT (`yolov7_trt.py`).
2. Manipulate bounding boxes with operations like cropping, resizing, and squaring (`boundbox_algo.py`).
3. Generate 24-frame video clips from JSON annotations, with configurable bounding box adjustments and frame selection (`json_to_24frame_video.py` and `create_tsm_dataset_with_yolo.py`).

## Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA and TensorRT support
- Required Python packages:
  ```bash
  pip install numpy opencv-python pycuda tensorrt tqdm torch torchvision
  ```
- FFmpeg installed and accessible in the system PATH
- A pre-trained YOLOv7 TensorRT engine file (`.engine`)

## Creating TensorRT Engine

### Method 1: From PyTorch Model (.pt file)

If you have a PyTorch model file (`.pt`), you can convert it to a TensorRT engine using the provided conversion script:

```bash
python simple_convert.py --input_model your_model.pt --output_engine your_model.engine --input_size 640
```

**Parameters:**
- `--input_model`: Path to your PyTorch model file (.pt)
- `--output_engine`: Output path for the TensorRT engine (.engine)
- `--output_onnx`: Optional ONNX intermediate file path
- `--input_size`: Input image size (default: 640)
- `--use_dummy`: Use dummy model if original model fails to load

**Example:**
```bash
python simple_convert.py --input_model 28052025_child_new_class.pt --output_engine 28052025_child_new_class.engine
```

### Method 2: From ONNX Model

If you already have an ONNX model:

```bash
python convert_engine.py --onnx_model your_model.onnx --engine_file your_model.engine
```

### Testing Your Engine

After creating an engine, test it to ensure it works correctly:

```bash
python test_simple_engine.py
```

This will load your engine and run a simple inference test.

### Environment Setup

For CUDA 12.6 and TensorRT 10 compatibility, set these environment variables:

```bash
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/targets/aarch64-linux/lib:$LD_LIBRARY_PATH
```

### Troubleshooting Engine Creation

**Common Issues:**

1. **ModuleNotFoundError**: If you get missing module errors during conversion, the script will automatically create dummy modules to handle this.

2. **CUDA Version Mismatch**: Ensure your CUDA environment is properly set up:
   ```bash
   nvidia-smi  # Check CUDA version
   python -c "import torch; print(torch.cuda.is_available())"  # Test PyTorch CUDA
   ```

3. **TensorRT API Changes**: The conversion script is compatible with TensorRT 10. For older versions, you may need to modify the API calls.

4. **Memory Issues**: If you encounter memory errors, try reducing the input size or using a smaller batch size.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure FFmpeg is installed:
   - On Ubuntu: `sudo apt-get install ffmpeg`
   - On macOS: `brew install ffmpeg`
   - On Windows: Download from [FFmpeg website](https://ffmpeg.org/download.html) and add to PATH

## Usage

### 1. Object Detection with YOLOv7 TensorRT

#### Method A: Using the New Detection Script (Recommended)

The `yolo_detect_to_json.py` script provides a complete pipeline for video processing and JSON annotation generation:

```bash
python yolo_detect_to_json.py \
  --yolov7_engine_file_path your_model.engine \
  --input_video_path video.mp4 \
  --dataset_path output_json \
  --frame_interval 60 \
  --temporal_size 24
```

**Parameters:**
- `--yolov7_engine_file_path`: Path to your TensorRT engine file
- `--input_video_path`: Path to input video (single video)
- `--video_folder_path`: Path to folder containing videos (batch processing)
- `--dataset_path`: Output directory for JSON annotations
- `--frame_interval`: Minimum frame interval between detections (default: 60)
- `--temporal_size`: Number of frames to process (default: 24)

**Example:**
```bash
python yolo_detect_to_json.py \
  --yolov7_engine_file_path 28052025_child_new_class.engine \
  --input_video_path NVR_ch13_main_20250207113000_20250207115100.mp4 \
  --dataset_path nvr_test_output \
  --frame_interval 60
```

#### Method B: Using the Original YOLOv7 Class

The `yolov7_trt.py` script provides a `YoLov7_TRT` class for running inference on video frames using a TensorRT-optimized YOLOv7 model.

**Example:**
```python
from yolov7_trt import YoLov7_TRT
import cv2

# Initialize the model
model = YoLov7_TRT(engine_file_path="yolov7.engine", conf_threshold=0.5, iou_threshold=0.4)

# Load an image
image = cv2.imread("sample.jpg")

# Run inference
boxes, scores, class_ids, inference_time = model.infer(image)

# Process results
for box, score, class_id in zip(boxes, scores, class_ids):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.imwrite("output.jpg", image)

# Clean up
model.destroy()
```

### 2. Bounding Box Manipulation

The `boundbox_algo.py` script provides utilities for adjusting bounding boxes, such as cropping the upper part, making them square, resizing, or expanding them.

**Functions:**
- `crop_upper_bbox`: Crops the upper part of a bounding box if its height exceeds a specified width ratio.
- `make_square_bbox`: Adjusts a bounding box to be square, centered on the original box.
- `resize_bbox`: Resizes a bounding box while optionally cropping to maintain a specific aspect ratio.
- `expand_crop_bbox`: Expands or shrinks a bounding box around its center.
- `adjust_to_even_dimensions`: Ensures bounding box dimensions are even for video encoding compatibility.

### 3. Generating Short Video Clips

The `json_to_24frame_video.py` and `create_tsm_dataset_with_yolo.py` scripts process JSON annotations to generate 24-frame video clips centered on detected objects. They support bounding box adjustments and frame selection modes (`past`, `future`, or `hybrid`).

**Command-line Arguments:**
- `--video_folder_path`: Path to input videos.
- `--dataset_path`: Directory containing JSON annotations.
- `--output_video_path`: Directory for output video clips.
- `--temporal_size`: Number of frames in each clip (default: 24).
- `--square`: Make bounding boxes square.
- `--upper_crop`: Crop the upper part of bounding boxes.
- `--bbox_resize_factor`: Resize bounding boxes by a factor (default: 1.0).
- `--crop_ratio`: Height-to-width ratio threshold for cropping (default: 1.6).
- `--expand_crop_factor`: Expand or shrink bounding boxes (default: 1.0).
- `--frame_interval`: Minimum frame interval between clips (default: 60).
- `--frame_direction`: Frame selection mode (`past`, `future`, or `hybrid`).
- `--skip_reencode`: Skip re-encoding input videos to H.264.
- `--reencode_dir`: Directory for re-encoded videos (default: system temp directory).

**Example Command:**
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

**Output:**
- Short video clips (e.g., `video_frame000123.mp4`) in the specified output directory.
- A `tasks.json` file for importing clips into Label Studio with pre-annotations.

### JSON Annotation Format

The JSON annotations should have the following structure:
```json
{
  "video_name": "video.mp4",
  "frame": 123,
  "detection": [
    {
      "value": {
        "x": 100,
        "y": 200,
        "width": 300,
        "height": 400
      }
    }
  ]
}
```

## Notes

- Ensure the TensorRT engine file is compatible with your GPU and TensorRT version.
- The pipeline assumes JSON annotations are stored in `dataset_path/annotations/`.
- Videos are re-encoded to H.264 by default for stability; use `--skip_reencode` to skip this step.
- The `h264_nvenc` codec is used for faster encoding if an NVIDIA GPU is available; it falls back to `libx264` if needed.
- Generated videos are compatible with Label Studio for further annotation.

## Troubleshooting

### General Issues

- **FFmpeg errors**: Ensure FFmpeg is installed and accessible in the system PATH.
- **TensorRT errors**: Verify that the engine file matches your TensorRT version and CUDA device.
- **Insufficient frames**: Adjust `--frame_direction` or ensure the input video has enough frames.
- **Directory permissions**: Ensure `dataset_path` and `output_video_path` are writable.

### Engine Creation Issues

- **CUDA not available**: Check CUDA installation and environment variables:
  ```bash
  nvidia-smi
  python -c "import torch; print(torch.cuda.is_available())"
  ```

- **TensorRT version mismatch**: The scripts are compatible with TensorRT 10. For older versions, modify API calls in the conversion scripts.

- **Memory errors during conversion**: Reduce input size or use smaller batch sizes.

- **ModuleNotFoundError during model loading**: The conversion script automatically creates dummy modules to handle missing dependencies.

### Detection Issues

- **Plugin library errors**: The new `yolo_detect_to_json.py` script doesn't require the old plugin library. Use this instead of the old version.

- **Engine loading failures**: Ensure your engine was created with the same TensorRT version and CUDA environment.

- **Low detection accuracy**: Check your model's confidence threshold and ensure the input video quality is adequate.

### Performance Optimization

- **Slow inference**: Use TensorRT engines optimized for your specific GPU architecture.
- **High memory usage**: Reduce batch size or input resolution.
- **Video processing speed**: Adjust `--frame_interval` to skip frames and reduce processing time.

## License

This project is licensed under the MIT License.