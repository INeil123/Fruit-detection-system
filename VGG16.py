# -*- coding: utf-8 -*-
import os
from PIL import Image
from torch.utils import data
from torchvision import transforms as T
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision.models import vgg16, VGG16_Weights
import matplotlib.pyplot as plt

# 定义标签
# Labels = {'粑粑柑': 0, '白兰瓜': 1, '白萝卜': 2, '白心火龙果': 3, '百香果': 4, '菠萝': 5, '菠萝莓': 6, '菠萝蜜': 7, '草莓': 8, '车厘子': 9, '番石榴-百': 10, '番石榴-红': 11, '佛手瓜': 12, '甘蔗': 13, '桂圆': 14, '哈密瓜': 15, '黑莓': 16, '红苹果': 17, '红心火龙果': 18, '胡萝卜': 19, '黄桃': 20, '金桔': 21, '橘子': 22, '蓝莓': 23, '梨': 24, '李子': 25, '荔枝': 26, '莲雾': 27, '榴莲': 28, '芦柑': 29, '芒果': 30, '毛丹': 31, '猕猴桃': 32, '木瓜': 33, '柠檬': 34, '牛油果': 35, '蟠桃': 36, '枇杷': 37, '葡萄-白': 38, '葡萄-红': 39, '脐橙': 40, '青柠': 41, '青苹果': 42, '人参果': 43, '桑葚': 44, '沙果': 45, '沙棘': 46, '砂糖橘': 47, '山楂': 48, '山竹': 49, '蛇皮果': 50, '圣女果': 51, '石 '人参果': '人 '人参果': 43, '桑葚': 44, '沙果': 45, '沙棘': 46, '砂糖橘': 47, '山楂': 48, '山竹': 49, '蛇皮果': 50, '圣女果': 51, '石榴': 52, '柿子': 53, '树莓': 54, '水蜜桃': 55, '酸角': 56, '甜瓜-白': 57, '甜瓜-金': 58, '甜瓜-绿': 59, '甜瓜-伊丽莎白': 60, '沃柑': 61, '无花果': 62, '西瓜': 63, '西红柿': 64, '西梅': 65, '西柚': 66, '香蕉': 67, '香橼': 68, '杏': 69, '血橙': 70, '羊角蜜': 71, '羊奶果': 72, '杨梅': 73, '杨桃': 74, '腰果': 75, '椰子': 76, '樱桃': 77, '油桃': 78, '柚子': 79, '枣': 80}
# 自定义数据集类
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

class SeedlingData(data.Dataset):
    def __init__(self, root, transforms=None, train=True, test=False):
        self.test = test
        self.transforms = transforms
        # 加载图像路径
        if self.test:
            self.imgs = [os.path.join(root, img) for img in os.listdir(root)]
        else:
            imgs_labels = [os.path.join(root, img) for img in os.listdir(root)]
            imgs = []
            for imglable in imgs_labels:
                for imgname in os.listdir(imglable):
                    imgpath = os.path.join(imglable, imgname)
                    imgs.append(imgpath)
            trainval_files, val_files = train_test_split(imgs, test_size=0.3, random_state=42)
            self.imgs = trainval_files if train else val_files

    def __getitem__(self, index):
        img_path = self.imgs[index]
        if self.test:
            label = -1
        else:
            labelname = os.path.basename(os.path.dirname(img_path))
            label = Labels[labelname]
        data = Image.open(img_path).convert('RGB')
        data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)

# 超参数
modellr = 1e-4
BATCH_SIZE = 32
EPOCHS = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet 标准化参数
])

# 加载数据集
train_dataset = SeedlingData(root='best.1/fruit81_full', transforms=transform, train=True)
val_dataset = SeedlingData(root='best.1/fruit81_full', transforms=transform, train=False)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 加载预训练的 VGG16 模型
model_ft = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
num_ftrs = model_ft.classifier[6].in_features
model_ft.classifier[6] = nn.Linear(num_ftrs, len(Labels))
model_ft = model_ft.to(DEVICE)

# 定义优化器和损失函数
optimizer = optim.Adam(model_ft.parameters(), lr=modellr)
criterion = nn.CrossEntropyLoss()

# 学习率调整函数
def adjust_learning_rate(optimizer, epoch):
    modellrnew = modellr * (0.1 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = modellrnew

# 训练函数
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    sum_loss = 0
    correct = 0
    total_num = len(train_loader.dataset)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sum_loss += loss.item()
        _, pred = torch.max(output, 1)
        correct += pred.eq(target).sum().item()

        if (batch_idx + 1) % 10 == 0:
            print(f'Train Epoch: {epoch} [{(batch_idx + 1) * len(data)}/{total_num} ({100. * (batch_idx + 1) / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    avg_loss = sum_loss / len(train_loader)
    accuracy = 100. * correct / total_num
    print(f'Epoch: {epoch}, Loss: {avg_loss:.6f}, Accuracy: {accuracy:.2f}%')

# 验证函数
def val(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total_num = len(test_loader.dataset)

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            _, pred = torch.max(output, 1)
            correct += pred.eq(target).sum().item()

    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total_num
    print(f'\nValidation set: Average loss: {avg_loss:.4f}, Accuracy: {correct}/{total_num} ({accuracy:.2f}%)\n')

# 训练和验证
for epoch in range(1, EPOCHS + 1):
    adjust_learning_rate(optimizer, epoch)
    train(model_ft, DEVICE, train_loader, optimizer, epoch)
    val(model_ft, DEVICE, val_loader)

# 保存模型
torch.save(model_ft.state_dict(), 'model.pth')