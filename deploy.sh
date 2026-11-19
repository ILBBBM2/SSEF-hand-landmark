#!/usr/bin/env bash
set -e
PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJ_DIR"

echo "=== Step 1: Re-quantizing RKNN models with exact MediaPipe ranges ==="
/home/ultra_potato/rknn-env/bin/python convert_hands_onnx.py

echo "=== Step 2: Cross-compiling ssef_vis for Luckfox RV1106 ==="
RKNN_LIB_DIR="/home/ultra_potato/luckfox-pico/media/iva/iva/librockiva/rockiva-rv1106-Linux/lib"
/home/ultra_potato/luckfox-pico/tools/linux/toolchain/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc \
  -O3 -Wall ssef_vis.c \
  -I/home/ultra_potato/luckfox-pico/project/app/rk_smart_door/smart_door/common/face/algo \
  -L"$RKNN_LIB_DIR" -Wl,-rpath-link,"$RKNN_LIB_DIR" \
  -lrknnmrt -lpthread -ldl -lm -o ssef_vis

echo "=== Step 3: Deploying binary and models to Luckfox (172.32.0.93) ==="
scp ssef_vis models/mediapipe_rknn/hand_detector.rknn models/mediapipe_rknn/hand_landmarks_detector.rknn root@172.32.0.93:/userdata/
echo "=== DONE! Now run on Luckfox: LD_LIBRARY_PATH=/oem/usr/lib:\$LD_LIBRARY_PATH /userdata/ssef_vis ==="
