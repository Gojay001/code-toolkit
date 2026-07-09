#!/bin/bash
# extract_alpha_and_apply.sh

SEQ1_DIR="/Users/bigo10295/Downloads/seq/zuo-huoyan"  # 包含RGBA帧的文件夹
SEQ2_DIR="/Users/bigo10295/Downloads/seq/zuo-huoyan-red"  # 包含RGB帧的文件夹
OUTPUT_DIR="/Users/bigo10295/Downloads/seq/zuo-huoyan-red-rgba"  # 输出文件夹

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "将 $SEQ1_DIR 的Alpha通道应用到 $SEQ2_DIR..."
echo "=============================================="

# 方法1：使用alphamerge滤镜
ffmpeg \
  -i "$SEQ2_DIR/%05d.png" \
  -i "$SEQ1_DIR/zuo-huoyan_%05d.png" \
  -filter_complex "[1]format=rgba,alphaextract[alpha];[0][alpha]alphamerge[out]" \
  -map "[out]" \
  -y \
  "$OUTPUT_DIR/%05d.png"

echo "✅ 完成！输出在: $OUTPUT_DIR"