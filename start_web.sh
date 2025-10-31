#!/bin/bash

echo "🌐 MotionSync - 位姿提取与动作估计软件启动脚本"
echo "========================================"

# 检查端口是否被占用
if lsof -Pi :8080 -sTCP:LISTEN -t >/dev/null ; then
    echo "端口8080被占用，正在停止现有进程..."
    pkill -f web_app.py
    sleep 2
fi

# 激活conda环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate yolo

echo "启动Web服务器..."
echo "访问地址: http://localhost:8080"
echo "或者: http://$(hostname -I | awk '{print $1}'):8080"
echo "========================================"

# 启动Web应用
python web_app.py
