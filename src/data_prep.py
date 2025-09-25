import os
import xml.etree.ElementTree as ET
from PIL import Image
import shutil


class DataPreprocessor:
    def __init__(self, data_root=None):
        # 原始数据根目录（支持自定义）
        self.raw_data_root = os.path.join(data_root or "data", "raw")
        # 预处理后的数据保存目录（支持自定义）
        self.processed_data_root = os.path.join(data_root or "data", "processed")
        # 缺陷类别映射（需与数据集中标注一致）
        self.class_mapping = {
            "crazing": 0,
            "inclusion": 1,
            "patches": 2,
            "pitted_surface": 3,
        }
        # 处理的数据集类型（新增test）
        self.splits = ["train", "valid", "test"]

    def process_all_splits(self):
        """统一处理训练集、验证集和测试集"""
        for split in self.splits:
            print(f"开始处理 {split} 数据集...")
            self._process_single_split(split)
        print("所有数据集预处理处理完成！")

    def _process_single_split(self, split_name):
        """处理单个数据集（训练/验证/测试）"""
        # 创建预处理目录
        img_dir, label_dir = self._create_directories(split_name)

        # 获取原始数据路径
        raw_img_path = os.path.join(self.raw_data_root, split_name, "images")
        raw_label_path = os.path.join(self.raw_data_root, split_name, "labels")

        # 遍历所有标注文件
        for xml_file in os.listdir(raw_label_path):
            if not xml_file.endswith(".xml"):
                continue  # 只处理XML文件

            # 标注文件和图像文件路径
            xml_full_path = os.path.join(raw_label_path, xml_file)
            img_file = xml_file.replace(".xml", ".jpg")  # 假设图像为JPG格式
            img_full_path = os.path.join(raw_img_path, img_file)

            # 转换标注格式（VOC XML -> YOLO TXT）
            yolo_labels = self._convert_voc_to_yolo(xml_full_path, img_full_path)

            # 保存YOLO格式标注
            self._save_labels(yolo_labels, label_dir, xml_file)

            # 复制图像到处理后目录
            shutil.copy2(img_full_path, os.path.join(img_dir, img_file))

    def _create_directories(self, split_name):
        """创建预处理所需的文件夹结构"""
        img_dir = os.path.join(self.processed_data_root, "images", split_name)
        label_dir = os.path.join(self.processed_data_root, "labels", split_name)

        # 递归创建目录（已存在则不报错）
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)

        return img_dir, label_dir

    def _convert_voc_to_yolo(self, xml_path, img_path):
        """将VOC格式的XML标注转换为YOLO格式的TXT标注"""
        # 解析XML文件
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 获取图像尺寸（用于归一化）
        with Image.open(img_path) as img:
            img_width, img_height = img.size

        yolo_labels = []
        # 遍历所有目标
        for obj in root.iter("object"):
            # 获取类别名称和ID
            class_name = obj.find("name").text
            if class_name not in self.class_mapping:
                print(f"警告：未知类别 {class_name}，已跳过")
                continue
            class_id = self.class_mapping[class_name]

            # 获取边界框坐标（VOC格式：xmin, ymin, xmax, ymax）
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            # 转换为YOLO格式（中心坐标+宽高，归一化到0-1）
            center_x = (xmin + xmax) / (2 * img_width)
            center_y = (ymin + ymax) / (2 * img_height)
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height

            # 保留6位小数，避免精度丢失
            yolo_labels.append([
                class_id,
                round(center_x, 6),
                round(center_y, 6),
                round(width, 6),
                round(height, 6)
            ])

        return yolo_labels

    def _save_labels(self, labels, label_dir, xml_file):
        """保存YOLO格式的标注文件"""
        # 标注文件名与图像名一致，后缀改为txt
        txt_file = xml_file.replace(".xml", ".txt")
        txt_path = os.path.join(label_dir, txt_file)

        # 写入标注内容
        with open(txt_path, "w", encoding="utf-8") as f:
            for label in labels:
                # 格式：class_id center_x center_y width height
                f.write(f"{label[0]} {label[1]} {label[2]} {label[3]} {label[4]}\n")


# 执行预处理
if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessor.process_all_splits()
