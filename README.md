# MotionSync - 位姿提取与动作估计软件

## 🎯 功能特性

- **位姿提取**: 使用YOLOv8-pose模型提取人体关键点位姿信息
- **动作估计**: 基于关键点序列进行动作识别和估计
- **格式转换**: 支持TXT ↔ COCO JSON ↔ LabelMe JSON转换
- **模型训练**: 基于现有数据训练自定义位姿检测模型和动作分类模型
- **视频动作标注**: 从视频中提取关键点并标注动作（W/A/S/D/空格/静止）
- **实时游戏标注**: 实时捕获游戏画面并标注动作
- **动作识别**: 实时识别视频中的动作并显示3D火柴人
- **Web界面**: 现代化浏览器界面，支持实时进度显示

## 🚀 快速开始

### 1. 环境准备
```bash
# 创建conda环境
conda create -n yolo python=3.10
conda activate yolo

# 安装依赖
pip install torch torchvision torchaudio
pip install ultralytics flask PyQt5
```

### 2. 启动Web版本
```bash
conda activate yolo
python web_app.py
```

### 3. 访问界面
打开浏览器访问: http://localhost:8080

**说明**: 
- `localhost:8080` 是本地地址，每个人在自己的电脑上运行，访问的是自己本地的服务，不会互相冲突
- 如果需要在同一台机器上运行多个实例，可以修改 `web_app.py` 第2814行的端口号（如改为8081、8082等）

## 🌐 Web界面使用

Web界面包含10个功能标签页，按功能分组如下：

### 📸 基础功能组

#### 1. 位姿提取
从图片中提取人体关键点位姿信息。

**使用步骤**:
1. 点击"位姿提取"标签页
2. 设置参数：
   - **图片目录**: `./datasets/images/train`（包含待处理图片的目录）
   - **模型路径**: `./models/yolov8n-pose.pt`（YOLOv8-pose模型）
   - **输出目录**: `./datasets/labels/train`（输出TXT标签文件）
   - **置信度阈值**: `0.5`（0.1-1.0，越高越严格）
3. 点击"开始位姿提取"
4. 观察进度条和实时日志输出

**输出格式**: YOLO格式的TXT文件，包含边界框和17个关键点坐标

#### 2. 格式转换
在不同标注格式之间转换（TXT ↔ COCO JSON ↔ LabelMe JSON）。

**使用步骤**:
1. 点击"格式转换"标签页
2. 选择转换类型：
   - `TXT → COCO JSON`: 将YOLO格式转换为COCO格式
   - `COCO JSON → TXT`: 将COCO格式转换为YOLO格式
   - `TXT → LabelMe JSON`: 将YOLO格式转换为LabelMe格式
   - `LabelMe JSON → TXT`: 将LabelMe格式转换为YOLO格式
3. 设置路径：
   - **输入路径**: 源文件或目录路径
   - **输出路径**: 目标文件路径
   - **图片目录**: 对应的图片目录（用于生成JSON）
4. 点击"开始转换"
5. 等待转换完成

#### 3. JSON验证与可视化
验证JSON文件结构并可视化关键点。

**使用步骤**:
1. 点击"JSON验证"标签页
2. 输入JSON文件路径（如 `./datasets/labels_json/1.json`）
3. 点击"验证JSON"检查文件结构是否正确
4. 点击"可视化关键点"查看关键点在图片上的标注
5. 使用图片索引输入框切换不同的图片

#### 4. 数据集信息
查看数据集的统计信息。

**使用步骤**:
1. 点击"数据集信息"标签页
2. 点击"刷新信息"更新统计数据
3. 查看：
   - 训练集/验证集图片数量
   - 标签文件数量
   - 数据集配置信息

#### 5. COCO Annotator管理
启动和管理COCO Annotator工具，用于可视化编辑标注。

**使用步骤**:
1. 点击"COCO工具"标签页
2. 点击"检查状态"查看COCO Annotator是否运行
3. 点击"启动COCO Annotator"启动服务（首次启动可能需要较长时间）
4. 等待启动完成后，点击"打开COCO Annotator"在新窗口打开
5. 在COCO Annotator中验证和修改JSON文件
6. 完成后点击"停止COCO Annotator"释放资源

### 🎬 视频与动作标注组

#### 6. 视频动作标注
从视频中提取关键点并手动标注动作。

**使用步骤**:
1. 点击"视频标注"标签页
2. 设置参数：
   - **视频路径**: `./1.mp4`（待处理的视频文件）
   - **模型路径**: `./models/yolov8n-pose.pt`
   - **输出JSON路径**: 留空则自动生成（格式：`视频名_action_labels.json`）
   - **置信度阈值**: `0.5`
   - **序列长度**: `30`（每次标注的持续时间，帧数）
   - **标注覆盖帧数**: `1`（1=逐帧标注，>1=批量标注）
   - **ROI区域**（可选）: `x,y,w,h`格式，限制检测区域
3. 点击"开始提取关键点"，等待关键点提取完成
4. 在视频播放界面进行标注：
   - 使用播放控制按钮或键盘方向键浏览视频
   - 点击按钮或按键盘标注动作：
     - **W**: 前进
     - **A**: 左移
     - **S**: 后退
     - **D**: 右移
     - **空格**: 跳跃
     - **I**: 静止
5. 标注完成后点击"保存标注"保存JSON文件

**提示**: 标注覆盖帧数设置为5时，每5帧标注一次，适合快速标注长视频

#### 7. 实时游戏标注
实时捕获游戏画面并自动标注动作（需要键盘监听）。

**使用步骤**:
1. **安装额外依赖**（首次使用）:
   ```bash
   pip install mss pynput
   ```
2. 点击"实时标注"标签页
3. 设置参数：
   - **模型路径**: `./models/yolov8n-pose.pt`
   - **屏幕区域**（可选）: `x,y,width,height`格式，留空则全屏捕获
   - **置信度阈值**: `0.5`
   - **捕获帧率**: `30`（10-60 FPS）
   - **输出JSON路径**（可选）: 留空则自动生成
   - **序列长度**: `30`
4. 点击"开始实时标注"
5. **在3秒内切换到游戏窗口**
6. 按住键盘进行动作标注：
   - 按住 **W/A/S/D/空格** 进行相应动作
   - 系统自动捕获画面并标注
7. 按 **ESC键** 停止标注并保存

**注意**: 
- Linux系统可能需要sudo权限运行（用于键盘监听）
- 确保游戏窗口在指定屏幕区域内

### 🤖 模型训练与识别组

#### 8. 动作分类模型训练
训练LSTM或ST-GCN模型用于动作分类。

**使用步骤**:
1. 点击"模型训练"标签页
2. 设置数据路径：
   - **标注数据路径**: 
     - 可以是包含多个`*_action_labels.json`文件的目录（如 `./action_labels/`）
     - 或单个JSON文件（如 `./1_action_labels.json`）
3. 选择模型类型：
   - **LSTM**: 双向LSTM模型，适合时序动作识别
   - **ST-GCN**: 时空图卷积网络，适合复杂动作识别
4. 设置训练参数：
   - **训练轮数**: `100`（epochs）
   - **批次大小**: `32`
   - **学习率**: `0.001`
   - **序列长度**: `30`（关键点序列长度）
   - **验证集比例**: `0.2`（20%）
5. 点击"开始训练"
6. 观察训练进度和实时日志
7. 训练完成后查看：
   - 训练曲线图
   - 混淆矩阵
   - 模型文件保存在 `./models/action_classifier/`

**输出文件**:
- `best_lstm_action_classifier.pth` 或 `best_stgcn_action_classifier.pth`
- `training_curves_*.png`（训练曲线）
- `confusion_matrix_*.png`（混淆矩阵）
- `training_log_*.txt`（训练日志）

#### 9. 动作识别（3D火柴人）
实时识别视频中的动作并显示3D火柴人可视化。

**使用步骤**:
1. 点击"动作识别"标签页
2. 设置视频路径：`./1.mp4` 或 `./datasets/videos/demo.mp4`
3. 点击"加载第一帧并选择ROI"（可选，用于限制检测区域）
   - 在第一帧图片上拖拽选择感兴趣区域
   - 点击"确认ROI"保存区域
4. 设置参数：
   - **模型路径**: `./models/yolov8n-pose.pt`（关键点检测模型）
   - **动作识别模型路径**（可选）: `./models/action_classifier/best_lstm_action_classifier.pth`
   - **置信度阈值**: `0.5`
   - **ROI区域**（可选）: 手动输入 `x,y,w,h` 格式
5. 点击"开始识别"
6. 在右侧查看：
   - **实时视频画面**（带关键点标注）
   - **3D火柴人可视化**（实时更新）
   - **动作识别结果**（如果提供了动作识别模型）
7. 点击"停止识别"结束

**功能说明**:
- 实时提取视频中的关键点
- 3D火柴人实时显示人体姿态
- 如果提供了训练好的动作分类模型，会实时显示识别的动作类别

### 🔧 工具与日志组

#### 10. 运行日志
查看Web应用的运行日志。

**使用步骤**:
1. 点击"运行日志"标签页
2. 点击"刷新日志"查看最新日志
3. 点击"清空日志"清空当前显示的日志（不影响日志文件）
4. 日志文件保存在 `./web_app.log`

## 📁 项目结构

```
sofatware/
├── web_app.py              # Web版本主程序 ⭐
├── start_web.sh            # Web版本启动脚本
├── auto_label.py            # 自动标定模块
├── txt_coco_json.py         # TXT与COCO JSON转换
├── txt_json.py              # TXT与LabelMe JSON转换
├── train_pose.py            # 模型训练
├── requirements.txt         # 依赖包列表
├── README.md                # 使用说明
├── WEB_GUIDE.md             # Web版本详细指南
├── datasets/                # 数据集目录
│   ├── images/
│   │   ├── train/           # 训练图片
│   │   └── val/             # 验证图片
│   ├── labels/
│   │   ├── train/           # 训练标签
│   │   └── val/             # 验证标签
│   └── train.yaml           # 数据集配置
├── models/                  # 模型文件
│   ├── yolov8n-pose.pt      # 预训练模型
│   ├── yolov8l-pose.pt      # 大模型
│   ├── yolo11n.pt           # YOLO11模型
│   └── action_classifier/   # 动作分类模型
│       ├── best_lstm_action_classifier.pth
│       ├── best_stgcn_action_classifier.pth
│       └── training_*.png   # 训练结果图表
└── templates/               # Web界面模板
    └── index.html           # 主页面
```

## 🔧 命令行使用

如果需要在命令行中使用，可以直接调用功能模块：

```bash
# 位姿提取
python -c "from auto_label import auto_label_yolo_format; auto_label_yolo_format('./datasets/images/train', './models/yolov8n-pose.pt', './datasets/labels/train', 0.5)"

# 格式转换
python -c "from txt_coco_json import txt_to_coco; txt_to_coco('./datasets/images/train', './datasets/labels/train', './output.json')"
```

## 📊 数据集格式

### YOLO格式
```
class_id x_center y_center width height kpt1_x kpt1_y kpt1_v ...
```

### 关键点定义 (17个关键点)
```
0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
```

## 🎉 总结

**推荐使用Web版本**，它提供了：
- 🌐 现代化浏览器界面
- ⚡ 实时进度显示和日志输出
- 🔧 完整的位姿提取、动作标注和识别功能
- 📱 无需安装客户端，通过浏览器即可使用
- 🎬 支持视频标注和实时游戏画面标注
- 🤖 集成动作分类模型训练和推理

**立即开始**: 
```bash
python web_app.py
# 或使用启动脚本
bash start_web.sh
```
然后访问 http://localhost:8080