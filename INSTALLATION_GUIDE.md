# TSM Annotation Label Studio 專案安裝指南

## 已安裝的依賴項

您的專案現在已經安裝了以下 Python 函式庫：

### 核心依賴項
- **opencv-python** (4.12.0.88) - 電腦視覺和影像處理
- **numpy** (2.2.6) - 數值計算
- **tqdm** (4.67.1) - 進度條顯示
- **tensorrt** (10.3.0) - NVIDIA TensorRT 推理引擎
- **pycuda** (2025.1.2) - CUDA 程式設計介面

### 輔助依賴項
- **pathlib2** (2.3.7.post1) - 路徑操作
- **pytools** (2025.2.4) - 工具函式
- **platformdirs** (4.4.0) - 平台特定目錄
- **typing-extensions** (4.15.0) - 型別提示擴展

## 使用方法

### 1. 設置環境變數
每次使用前，請確保將 pip 安裝路徑添加到 PATH：

```bash
export PATH="$HOME/.local/bin:$PATH"
```

### 2. 運行 Python 腳本
現在您可以運行專案中的任何 Python 腳本：

```bash
# 生成 TSM 資料集
python3 generate_tsm_dataset.py --help

# 將 JSON 轉換為 24 幀影片
python3 json_to_24frame_video.py --help

# YOLO 檢測轉 JSON
python3 yolo_detect_to_json.py --help

# 使用工具
python3 tools/read_tsm_annotations.py --help
python3 tools/segments_select_8frames.py --help
```

### 3. 重新安裝依賴項
如果需要重新安裝，可以運行：

```bash
./install_dependencies.sh
```

## 專案結構

```
TSMAnnotationLabelStudioWithJson/
├── generate_tsm_dataset.py          # 從影片和 JSON 生成 TSM 資料集
├── json_to_24frame_video.py         # 從 JSON 註解生成 24 幀影片
├── yolo_detect_to_json.py           # YOLO 檢測結果轉 JSON
├── yolov7_trt.py                    # YOLOv7 TensorRT 推理
├── boundbox_algo.py                 # 邊界框演算法
├── tools/                           # 工具腳本
│   ├── read_tsm_annotations.py      # 讀取 TSM 註解
│   └── segments_select_8frames.py   # 選擇 8 幀片段
├── engine_plugin/                   # TensorRT 引擎檔案
├── output_json/                     # JSON 輸出目錄
├── output_videos/                   # 影片輸出目錄
├── datasets_tsm/                    # TSM 資料集
└── requirements_basic.txt           # 基本依賴項清單
```

## 注意事項

1. **TensorRT 引擎檔案**：確保 `engine_plugin/` 目錄中有正確的 `.engine` 檔案
2. **CUDA 支援**：此專案需要 NVIDIA GPU 和 CUDA 支援
3. **記憶體需求**：處理大型影片檔案時可能需要大量記憶體
4. **權限**：某些操作可能需要適當的檔案系統權限

## 故障排除

### 如果遇到 "command not found" 錯誤：
```bash
export PATH="$HOME/.local/bin:$PATH"
```

### 如果遇到模組導入錯誤：
```bash
python3 -m pip install --upgrade --user [模組名稱]
```

### 如果遇到 CUDA 相關錯誤：
確保您的系統已正確安裝 NVIDIA 驅動程式和 CUDA 工具包。

## 支援

如果遇到任何問題，請檢查：
1. Python 版本（需要 3.7+）
2. 系統是否支援 CUDA
3. 所有依賴項是否正確安裝
4. 檔案路徑和權限設置
