### 下载yolo

```bash
conda create -n yolo python=3.10
pip install torch torchvision torchaudio
pip install ultralytics
测试：
yolo pose predict model=yolov8l-pose.pt source='./1.mp4' show
```
### 训练自己的数据集（基于yolov8n-pose）
17 个关键点的顺序和定义与 COCO 一致：
```bash
序号	名称	        说明
0	nose	        鼻尖
1	left_eye	左眼
2	right_eye	右眼
3	left_ear	左耳
4	right_ear	右耳
5	left_shoulder	左肩
6	right_shoulder	右肩
7	left_elbow	左肘
8	right_elbow	右肘
9	left_wrist	左手腕
10	right_wrist	右手腕
11	left_hip	左髋
12	right_hip	右髋
13	left_knee	左膝
14	right_knee	右膝
15	left_ankle	左踝
16	right_ankle	右踝
```

### 自动标注
auto_label.py

效果图：
![alt text](./images/1.png)

### txt-COCO JSON
```bash
 python txt-COCO\ JSON.py --mode txt2coco     --img_dir ./datasets/images/train     --txt_dir ./datasets/labels/train     --save_dir ./datasets/labels_json/train_coco.json
 python txt-COCO\ JSON.py --mode coco2txt     --json_path ./datasets/labels_json/test_1-1.json     --save_dir ./datasets/labels/train
```
### COCO Annotator
安装 docker-compose（如果没安装） 
```bash
sudo apt update
sudo apt install -y docker-compose
```

进入 coco-annotator 目录
```bash
cd ~/coco-annotator-master
```
启动服务
```bash
docker-compose up -d
```

关闭服务
```bash
docker-compose down
```
浏览器访问
```bash
http://localhost:5000
```
需要在Categories下新建特征点类别，比如person


修改文件夹所有权
```bash
sudo chown -R $USER:$USER datasets
```
删除数据集目录

1.退出docker \
2.删除docker volume rm $(docker volume ls -q) （COCO Annotator 默认会创建一个 MongoDB 的持久化卷）

追加训练
```bash
yolo pose train model=yolov8n-pose.pt data=./datasets/train.yaml epochs=20 imgsz=640
```