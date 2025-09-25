import os
import random
import shutil
from pathlib import Path

class TestSetSplitter:
    def __init__(self, raw_data_root: str = "data/raw"):
        self.raw_data_root = Path(raw_data_root)  # 原始数据根目录
        self.test_ratio = 0.1  # 测试集占训练集的比例（10%）
        self.image_suffix = ".jpg"  # 图像格式（根据实际修改，如 .png）
        self.label_suffix = ".txt"  # 标注格式（改为 .txt）

    def split(self):
        # 1. 创建测试集目录
        test_dir = self.raw_data_root / "test"
        test_dir.mkdir(exist_ok=True)
        (test_dir / "images").mkdir(exist_ok=True)
        (test_dir / "labels").mkdir(exist_ok=True)

        # 2. 获取训练集图像
        train_images_dir = self.raw_data_root / "train" / "images"
        # 只获取指定格式的图像（如 .jpg）
        train_images = list(train_images_dir.glob(f"*{self.image_suffix}"))
        random.shuffle(train_images)  # 打乱训练集图像

        # 3. 计算测试集数量并拆分
        test_count = int(len(train_images) * self.test_ratio)
        test_images = train_images[:test_count]  # 选前10%作为测试集

        # 4. 移动图像和标注到测试集
        for img_path in test_images:
            img_name = img_path.name
            # 构建标注文件名（.txt 格式）
            label_name = img_name.replace(self.image_suffix, self.label_suffix)
            label_path = self.raw_data_root / "train" / "labels" / label_name

            # 移动图像
            shutil.move(img_path, test_dir / "images" / img_name)
            # 移动标注（若存在）
            if label_path.exists():
                shutil.move(label_path, test_dir / "labels" / label_name)

        print(f"✅ 测试集拆分完成！共移动 {test_count} 组（图像+标注）到 {test_dir}")

if __name__ == "__main__":
    splitter = TestSetSplitter()
    splitter.split()