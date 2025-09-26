#!/bin/bash

# 使用 Docker 安裝 Label Studio 的腳本
# 這是最穩定的安裝方法

echo "開始使用 Docker 安裝 Label Studio..."

# 檢查 Docker 是否安裝
if ! command -v docker &> /dev/null; then
    echo "錯誤：未找到 Docker，請先安裝 Docker"
    echo "安裝 Docker 的指令："
    echo "curl -fsSL https://get.docker.com -o get-docker.sh"
    echo "sudo sh get-docker.sh"
    echo "sudo usermod -aG docker \$USER"
    echo "然後重新登入系統"
    exit 1
fi

# 創建 Label Studio 資料目錄
LABEL_STUDIO_DATA_DIR="$HOME/label_studio_data"
mkdir -p "$LABEL_STUDIO_DATA_DIR"

echo "創建 Label Studio 資料目錄：$LABEL_STUDIO_DATA_DIR"

# 創建 Docker Compose 檔案
cat > "$LABEL_STUDIO_DATA_DIR/docker-compose.yml" << 'EOF'
version: '3.8'

services:
  label-studio:
    image: heartexlabs/label-studio:latest
    container_name: label-studio
    ports:
      - "8080:8080"
    volumes:
      - ./data:/label-studio/data
      - ./media:/label-studio/media
    environment:
      - LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
      - LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/label-studio/media
    restart: unless-stopped
EOF

# 創建啟動腳本
cat > "$LABEL_STUDIO_DATA_DIR/start_label_studio.sh" << 'EOF'
#!/bin/bash
echo "啟動 Label Studio..."
docker-compose up -d
echo "Label Studio 正在啟動中..."
echo "請等待幾分鐘，然後在瀏覽器中打開：http://localhost:8080"
echo ""
echo "查看日誌：docker-compose logs -f"
echo "停止服務：docker-compose down"
EOF

# 創建停止腳本
cat > "$LABEL_STUDIO_DATA_DIR/stop_label_studio.sh" << 'EOF'
#!/bin/bash
echo "停止 Label Studio..."
docker-compose down
echo "Label Studio 已停止"
EOF

chmod +x "$LABEL_STUDIO_DATA_DIR/start_label_studio.sh"
chmod +x "$LABEL_STUDIO_DATA_DIR/stop_label_studio.sh"

echo "Label Studio Docker 安裝完成！"
echo ""
echo "使用方法："
echo "1. 進入 Label Studio 目錄：cd $LABEL_STUDIO_DATA_DIR"
echo "2. 啟動 Label Studio：./start_label_studio.sh"
echo "3. 停止 Label Studio：./stop_label_studio.sh"
echo "4. 在瀏覽器中打開：http://localhost:8080"
echo ""
echo "注意：首次啟動可能需要下載 Docker 映像檔，請耐心等待"


