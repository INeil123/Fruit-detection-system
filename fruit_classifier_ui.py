import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, 
                           QVBoxLayout, QHBoxLayout, QWidget, QFileDialog)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

# 定义标签字典（与VGG16.py中相同）
Labels = {
    '粑粑柑': 0, '白兰瓜': 1, '白萝卜': 2, '白心火龙果': 3, '百香果': 4, '菠萝': 5, '菠萝莓': 6, '菠萝蜜': 7, '草莓': 8, '车厘子': 9, 
    '番石榴-百': 10, '番石榴-红': 11, '佛手瓜': 12, '甘蔗': 13, '桂圆': 14, '哈密瓜': 15, '黑莓': 16, '红苹果': 17, '红心火龙果': 18, 
    '胡萝卜': 19, '黄桃': 20, '金桔': 21, '橘子': 22, '蓝莓': 23, '梨': 24, '李子': 25, '荔枝': 26, '莲雾': 27, '榴莲': 28, '芦柑': 29, 
    '芒果': 30, '毛丹': 31, '猕猴桃': 32, '木瓜': 33, '柠檬': 34, '牛油果': 35, '蟠桃': 36, '枇杷': 37, '葡萄-白': 38, '葡萄-红': 39, 
    '脐橙': 40, '青柠': 41, '青苹果': 42, '人参果': 43, '桑葚': 44, '沙果': 45, '沙棘': 46, '砂糖橘': 47, '山楂': 48, '山竹': 49, 
    '蛇皮果': 50, '圣女果': 51, '石榴': 52, '柿子': 53, '树莓': 54, '水蜜桃': 55, '酸角': 56, '甜瓜-白': 57, '甜瓜-金': 58, '甜瓜-绿': 59, 
    '甜瓜-伊丽莎白': 60, '沃柑': 61, '无花果': 62, '西瓜': 63, '西红柿': 64, '西梅': 65, '西柚': 66, '香蕉': 67, '香橼': 68, '杏': 69, 
    '血橙': 70, '羊角蜜': 71, '羊奶果': 72, '杨梅': 73, '杨桃': 74, '腰果': 75, '椰子': 76, '樱桃': 77, '油桃': 78, '柚子': 79, '枣': 80
}

# 创建反向映射字典
Labels_reverse = {v: k for k, v in Labels.items()}

class FruitClassifierUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.loadModel()
        
    def initUI(self):
        self.setWindowTitle('水果识别系统')
        self.setGeometry(100, 100, 800, 600)
        
        # 创建主窗口部件和布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()
        
        # 创建图片显示标签
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 400)
        self.image_label.setStyleSheet("border: 2px dashed #aaa;")
        layout.addWidget(self.image_label)
        
        # 创建按钮布局
        button_layout = QHBoxLayout()
        
        # 上传图片按钮
        self.upload_btn = QPushButton('上传图片')
        self.upload_btn.clicked.connect(self.uploadImage)
        button_layout.addWidget(self.upload_btn)
        
        # 预测按钮
        self.predict_btn = QPushButton('开始预测')
        self.predict_btn.clicked.connect(self.predictImage)
        self.predict_btn.setEnabled(False)
        button_layout.addWidget(self.predict_btn)
        
        layout.addLayout(button_layout)
        
        # 创建结果显示标签
        self.result_label = QLabel('请上传图片进行识别')
        self.result_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.result_label)
        
        main_widget.setLayout(layout)
        
    def loadModel(self):
        # 加载模型
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        num_ftrs = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_ftrs, len(Labels))
        
        # 加载训练好的权重
        if os.path.exists('model.pth'):
            self.model.load_state_dict(torch.load('model.pth', map_location=self.device))
            self.model = self.model.to(self.device)
            self.model.eval()
        else:
            self.result_label.setText('错误：找不到模型文件 model.pth')
            
        # 定义图像转换
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def uploadImage(self):
        file_name, _ = QFileDialog.getOpenFileName(self, '选择图片', '', 
                                                 'Image files (*.jpg *.jpeg *.png *.bmp *.gif)')
        if file_name:
            # 显示图片
            pixmap = QPixmap(file_name)
            scaled_pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
            self.current_image_path = file_name
            self.predict_btn.setEnabled(True)
            self.result_label.setText('图片已上传，点击"开始预测"进行识别')
            
    def predictImage(self):
        if hasattr(self, 'current_image_path'):
            # 加载和预处理图片
            image = Image.open(self.current_image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # 进行预测
            with torch.no_grad():
                outputs = self.model(image_tensor)
                _, predicted = torch.max(outputs, 1)
                predicted_class = Labels_reverse[predicted.item()]
                
            # 显示结果
            self.result_label.setText(f'预测结果：{predicted_class}')
        else:
            self.result_label.setText('请先上传图片')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = FruitClassifierUI()
    ex.show()
    sys.exit(app.exec_()) 