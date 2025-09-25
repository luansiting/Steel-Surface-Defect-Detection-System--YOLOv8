import streamlit as st
from PIL import Image
from ultralytics import YOLO

# 加载模型（使用你的最佳模型路径）
MODEL_PATH = "D:\\defect_demo\\runs\\train\\yolov8n_steel_defect\\weights\\best.pt"
model = YOLO(MODEL_PATH)

st.title("钢铁缺陷检测最简版")
uploaded = st.file_uploader("上传钢铁表面图片", type=["jpg", "png"])

if uploaded:
    # 显示上传的图片
    img = Image.open(uploaded)
    st.image(img, caption="上传的图像", use_column_width=True)

    # 执行检测
    results = model(img)

    # 生成带缺陷框的图像
    annotated_img = results[0].plot()
    st.image(annotated_img, caption="检测结果", use_column_width=True)

    # 显示缺陷信息
    st.subheader("检测到的缺陷详情")
    for box in results[0].boxes:
        defect_class = model.names[int(box.cls)]  # 获取缺陷类别
        confidence = float(box.conf)  # 获取置信度
        st.write(f"- 缺陷类型: {defect_class}，置信度: {confidence:.2f}")