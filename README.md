# TSM Annotation Label Studio with JSON

A comprehensive pipeline for video object detection, annotation, and TSM (Temporal Segment Networks) dataset generation using YOLOv7 TensorRT inference and Label Studio integration.

## 🚀 Features

- **YOLOv7 TensorRT Inference**: High-performance object detection using NVIDIA TensorRT
- **Multi-class Detection**: Support for multiple object classes with configurable confidence thresholds
- **Video Processing**: Generate short video clips from detected objects
- **Label Studio Integration**: Export annotations for Label Studio workflow
- **TSM Dataset Generation**: Create training datasets for Temporal Segment Networks
- **Bounding Box Manipulation**: Advanced algorithms for cropping, resizing, and adjusting detection boxes
- **High-Quality Video Output**: Optimized video encoding with FFmpeg

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Reference](#-api-reference)
- [Examples](#-examples)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)

## 🏃 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Test Your TensorRT Engine
```bash
python test_simple_engine.py
```

### 3. Process Video and Generate Annotations
```bash
python yolo_detect_to_json.py \
  --yolov7_engine_file_path engine_plugin/childabuse_28032025_re.engine \
  --input_video_path NVR_ch13_main_20250207113000_20250207115100.mp4 \
  --dataset_path nvr_test_output \
  --frame_interval 60 \
  --target_classes "2" \
  --conf_threshold 0.5
```

### 4. Generate Short Video Clips
```bash
python json_to_24frame_video.py \
  --video_folder_path . \
  --dataset_path nvr_test_output \
  --output_video_path test_output_videos \
  --temporal_size 24 \
  --frame_interval 60 \
  --upper_crop
```

### 5. Create TSM Dataset
```bash
python generate_tsm_dataset.py \
  --video_folder test_output_videos \
  --output_folder tsm_dataset_output \
  --json_folder . \
  --expected_frames 24
```

## 🔧 Installation

### Prerequisites
- Python 3.8+
- NVIDIA GPU with CUDA support
- TensorRT 10.x
- FFmpeg

### System Requirements
```bash
# Check CUDA installation
nvidia-smi

# Check TensorRT
python -c "import tensorrt; print(tensorrt.__version__)"

# Check FFmpeg
ffmpeg -version
```

### Install Python Dependencies
```bash
# Basic requirements
pip install -r requirements_basic.txt

# Full requirements (includes TensorRT)
pip install -r requirements.txt
```

### Environment Setup
```bash
# For CUDA 12.6 and TensorRT 10
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/targets/aarch64-linux/lib:$LD_LIBRARY_PATH
```

## 📖 Usage

### Object Detection Pipeline

#### 1. YOLOv7 TensorRT Detection
```bash
python yolo_detect_to_json.py \
  --yolov7_engine_file_path <engine_file> \
  --input_video_path <video_file> \
  --dataset_path <output_dir> \
  --frame_interval 60 \
  --target_classes "0,1,2" \
  --conf_threshold 0.5 \
  --iou_threshold 0.4
```

**Parameters:**
- `--yolov7_engine_file_path`: Path to TensorRT engine file
- `--input_video_path`: Input video file
- `--dataset_path`: Output directory for JSON annotations
- `--frame_interval`: Minimum frames between detections (default: 60)
- `--target_classes`: Comma-separated class IDs (default: "0,1,2")
- `--conf_threshold`: Confidence threshold (default: 0.5)
- `--iou_threshold`: IoU threshold for NMS (default: 0.4)

#### 2. Video Clip Generation
```bash
python json_to_24frame_video.py \
  --video_folder_path <video_dir> \
  --dataset_path <json_dir> \
  --output_video_path <output_dir> \
  --temporal_size 24 \
  --frame_interval 60 \
  --square \
  --upper_crop \
  --bbox_resize_factor 1.2 \
  --expand_crop_factor 1.1
```

**Parameters:**
- `--temporal_size`: Number of frames per clip (default: 24)
- `--square`: Make bounding boxes square
- `--upper_crop`: Crop upper part of bounding boxes
- `--bbox_resize_factor`: Resize factor for bounding boxes (default: 1.2)
- `--expand_crop_factor`: Expansion factor for cropping (default: 1.1)
- `--frame_direction`: Frame selection mode (past/future/hybrid)

#### 3. TSM Dataset Generation
```bash
python generate_tsm_dataset.py \
  --video_folder <video_dir> \
  --output_folder <output_dir> \
  --json_folder <json_dir> \
  --expected_frames 24
```

### TensorRT Engine Creation

#### From PyTorch Model
```bash
python convert_engine.py \
  --input_model your_model.pt \
  --output_engine your_model.engine \
  --input_size 640
```

#### From ONNX Model
```bash
python convert_engine.py \
  --onnx_model your_model.onnx \
  --engine_file your_model.engine
```

## 🔌 API Reference

### YoLov7_TRT Class
```python
from yolov7_trt import YoLov7_TRT

# Initialize model
model = YoLov7_TRT(
    engine_file_path="model.engine",
    conf_threshold=0.5,
    iou_threshold=0.4
)

# Run inference
boxes, scores, class_ids, inference_time = model.infer(image)

# Clean up
model.destroy()
```

### Bounding Box Algorithms
```python
from boundbox_algo import (
    crop_upper_bbox,
    make_square_bbox,
    resize_bbox,
    expand_crop_bbox
)

# Crop upper part
bbox = crop_upper_bbox(bbox, width, height, ratio_threshold=2.0)

# Make square
bbox = make_square_bbox(bbox, width, height)

# Resize with factor
bbox = resize_bbox(bbox, resize_factor=1.2, width, height)

# Expand around center
bbox = expand_crop_bbox(bbox, width, height, expand_factor=1.1)
```

## 💡 Examples

### Complete Workflow Example

```bash
# 1. Process video with detection
python yolo_detect_to_json.py \
  --yolov7_engine_file_path engine_plugin/childabuse_28032025_re.engine \
  --input_video_path sample_video.mp4 \
  --dataset_path annotations \
  --frame_interval 60 \
  --target_classes "2" \
  --conf_threshold 0.5

# 2. Generate video clips
python json_to_24frame_video.py \
  --video_folder_path . \
  --dataset_path annotations \
  --output_video_path video_clips \
  --temporal_size 24 \
  --upper_crop \
  --bbox_resize_factor 1.2

# 3. Create TSM dataset
python generate_tsm_dataset.py \
  --video_folder video_clips \
  --output_folder tsm_dataset \
  --json_folder . \
  --expected_frames 24
```

### Label Studio Integration

The pipeline generates `tasks.json` files compatible with Label Studio:

```json
{
  "data": {
    "video": "/path/to/video_clip.mp4"
  },
  "predictions": [
    {
      "result": [
        {
          "value": {
            "x": 100,
            "y": 200,
            "width": 300,
            "height": 400
          },
          "from_name": "bbox",
          "to_name": "video",
          "type": "rectangle"
        }
      ]
    }
  ]
}
```

## 🐛 Troubleshooting

### Common Issues

#### Engine Loading Failures
```bash
# Check TensorRT version compatibility
python -c "import tensorrt; print(tensorrt.__version__)"

# Verify CUDA environment
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

#### FFmpeg Errors
```bash
# Install FFmpeg
sudo apt-get install ffmpeg  # Ubuntu
brew install ffmpeg          # macOS

# Check installation
ffmpeg -version
```

#### Memory Issues
- Reduce `--frame_interval` to process fewer frames
- Use smaller input video resolution
- Reduce `--temporal_size` for shorter clips

#### Detection Issues
- Lower `--conf_threshold` for more detections
- Check `--target_classes` parameter
- Verify engine file compatibility

### Performance Optimization

#### GPU Memory
```bash
# Monitor GPU usage
nvidia-smi -l 1

# Reduce batch size in engine creation
python convert_engine.py --batch_size 1
```

#### Processing Speed
```bash
# Skip frame re-encoding
python json_to_24frame_video.py --skip_reencode

# Use hardware acceleration
python json_to_24frame_video.py --use_hw_accel
```

## 📁 Project Structure

```
TSMAnnotationLabelStudioWithJson/
├── 📄 Core Scripts
│   ├── yolo_detect_to_json.py      # Main detection pipeline
│   ├── json_to_24frame_video.py    # Video clip generation
│   ├── generate_tsm_dataset.py     # TSM dataset creation
│   └── yolov7_trt.py              # TensorRT inference
├── 🔧 Utilities
│   ├── boundbox_algo.py           # Bounding box algorithms
│   ├── convert_engine.py          # Engine conversion
│   └── test_simple_engine.py      # Engine testing
├── 📁 Models
│   ├── models/                    # YOLOv7 model files
│   └── utils/                     # Utility functions
├── 🎯 Examples
│   ├── engine_plugin/             # TensorRT engines
│   └── datasets_tsm/              # Sample datasets
└── 📋 Configuration
    ├── requirements.txt           # Dependencies
    ├── INSTALLATION_GUIDE.md     # Setup guide
    └── project-2-at-*.json       # Label Studio config
```

## 🎯 Use Cases

- **Video Surveillance**: Process security camera footage for anomaly detection
- **Behavioral Analysis**: Generate datasets for human action recognition
- **Quality Control**: Monitor industrial processes with computer vision
- **Research**: Create training datasets for temporal analysis models

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [YOLOv7](https://github.com/WongKinYiu/yolov7) for the object detection model
- [TensorRT](https://developer.nvidia.com/tensorrt) for GPU acceleration
- [Label Studio](https://labelstud.io/) for annotation workflow
- [FFmpeg](https://ffmpeg.org/) for video processing

## 📞 Support

For questions and support:
- Create an issue in the GitHub repository
- Check the [Troubleshooting](#-troubleshooting) section
- Review the [Examples](#-examples) for common use cases

---

**Made with ❤️ for the computer vision community**
