# 钢铁表面缺陷检测系统（YOLOv8）

## 项目简介  
基于 **YOLOv8** 构建的工业级钢铁表面缺陷检测方案，深度优化**小目标识别**与**少样本场景适配**，覆盖**裂纹、夹杂、斑块、麻点** 4 类核心缺陷，助力产线质量自动化筛查。  


## 核心价值  
- **场景适配**：专为钢铁产线设计，解决**小缺陷难检测**、**数据稀缺**痛点  
- **全流程覆盖**：打通「**数据处理**→**模型训练**→**推理部署**」完整链路  
- **高效实用**：推理速度 ~0.8s/张，支持产线**实时检测**  


## 技术架构  
```mermaid
graph TD
    A[原始数据<br>(图像+标注)] --> B[自动化预处理<br>(OpenCV+数据增强)]
    B --> C[YOLOv8 模型训练<br>(GhostNet骨干+CBAM注意力)]
    C --> D[ONNX推理优化<br>(0.8s/张)]
    D --> E[产线部署<br>(实时缺陷预警)]


### 1. 环境准备  
bash
# 克隆仓库
git clone https://github.com/[你的用户名]/defect_demo.git
cd defect_demo

# 安装依赖
pip install -r requirements.txt

### 2. 数据准备
plaintext
data/
├── train/
│   ├── images/  # 训练图像（.jpg 格式）
│   └── labels/  # 标注文件（.txt 格式，YOLO 格式）
├── val/         # 验证集（结构同 train/）
└── test/        # 测试集（结构同 train/）

3. 模型训练
修改 config/defect.yaml 配置训练参数（如迭代次数、批次大小等），执行训练：
bash
python main.py --train
训练日志与权重文件默认保存至 runs/train/ 目录。

4. 缺陷检测
使用训练好的模型（默认路径 runs/train/exp/weights/best.pt ），对测试集执行检测：
bash
python main.py --detect \
  --weights runs/train/exp/weights/best.pt \
  --source data/test/images  # 测试图像路径

## 📊 检测结果展示  

### 1. 模型训练收敛情况  
模型在测试集上的 F1 分数最终达到 **92.3%** ，训练曲线显示收敛稳定：  

![F1分数变化曲线](runs/train/yolov8n_steel_defect/BoxF1_curve.png)  


### 2. 缺陷检测实际效果  
验证集图片的检测可视化，绿色框为模型预测的缺陷位置：  

![验证集检测效果](runs/train/yolov8n_steel_defect/val_batch0_pred.png)  


### 3. 各类缺陷检测精度对比  
混淆矩阵展示了 4 类缺陷的检测准确率：  

![类别混淆矩阵](runs/train/yolov8n_steel_defect/confusion_matrix.png)  
