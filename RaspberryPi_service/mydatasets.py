import json
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from glob import glob


class CIFAR10_IMG(Dataset):

    def __init__(self, root, train=True, transform=None, target_transform=None):
        super(CIFAR10_IMG, self).__init__()
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        # 如果是训练则加载训练集，如果是测试则加载测试集
        if self.train:
            file_annotation = root + '/annotations/train.json'
            img_folder = root + '/train/'
        else:
            file_annotation = root + '/annotations/test.json'
            img_folder = root + '/test/'
        # fp = open(file_annotation, 'r')
        # data_dict = json.load(fp)

        with open(file_annotation, 'r') as f:
            data_dict = json.loads(json.load(f))
        # 如果图像数和标签数不匹配说明数据集标注生成有问题，报错提示
        assert len(data_dict['images']) == len(data_dict['categories'])
        num_data = len(data_dict['images'])

        self.filenames = []
        self.labels = []
        self.img_folder = img_folder
        for i in range(num_data):
            self.filenames.append(data_dict['images'][i])
            self.labels.append(data_dict['categories'][i])

    def __getitem__(self, index):
        img_name = self.img_folder + self.filenames[index]
        label = self.labels[index]

        img1 = Image.open(img_name)
        # new_img=img1.convert('RGB')
        # new_img = img1.resize((224, 224))
        new_img = img1.resize((32, 32))
        img = self.transform(new_img)  # 可以根据指定的转化形式对数据集进行转换

        # return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容
        return img, label

    def __len__(self):
        return len(self.filenames)


class MNIST_IMG(Dataset):

    def __init__(self, root, train=True, transform=None, target_transform=None):
        super(CIFAR10_IMG, self).__init__()
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        # 如果是训练则加载训练集，如果是测试则加载测试集
        if self.train:
            file_annotation = root + '/annotations/cifar10_train.json'
            img_folder = root + '/train_cifar10/'
        else:
            file_annotation = root + '/annotations/cifar10_test.json'
            img_folder = root + '/test_cifar10/'
        fp = open(file_annotation, 'r')
        data_dict = json.load(fp)

        # 如果图像数和标签数不匹配说明数据集标注生成有问题，报错提示
        assert len(data_dict['images']) == len(data_dict['categories'])
        num_data = len(data_dict['images'])

        self.filenames = []
        self.labels = []
        self.img_folder = img_folder
        for i in range(num_data):
            self.filenames.append(data_dict['images'][i])
            self.labels.append(data_dict['categories'][i])

    def __getitem__(self, index):
        img_name = self.img_folder + self.filenames[index]
        label = self.labels[index]

        img = plt.imread(img_name)
        img = self.transform(img)  # 可以根据指定的转化形式对数据集进行转换

        # return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容
        return img, label

    def __len__(self):
        return len(self.filenames)


class Chinese_IMG(Dataset):

    def __init__(self, root, train=True, transform=None, target_transform=None):
        super(Chinese_IMG, self).__init__()
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.filenames = []
        self.labels = []
        # 如果是训练则加载训练集，如果是测试则加载测试集
        train_dic = {'福': 0, '如': 1, '东': 2, '海': 3, '类': 4, '犬': 5, '画': 6, '虎': 7, '花': 8, '世': 9, '界': 10, '欢': 11,
                     '天': 12, '喜': 13, '地': 14, '皆': 15, '拾': 16, '是': 17, '俯': 18, '前': 19, '月': 20, '下': 21, '回': 22,
                     '光': 23, '返': 24, '照': 25, '蛇': 26, '添': 27, '足': 28, '短': 29, '寸': 30, '尺': 31, '长': 32, '呼': 33,
                     '雀': 34, '跃': 35, '崇': 36, '洋': 37, '媚': 38, '外': 39, '冤': 40, '家': 41, '付': 42, '之': 43, '流': 44,
                     '肠': 45, '断': 46, '愁': 47, '见': 48, '薄': 49, '识': 50, '论': 51, '齿': 52, '余': 53, '牙': 54, '草': 55,
                     '不': 56, '生': 57, '夺': 58, '目': 59, '彩': 60, '古': 61, '今': 62, '中': 63, '容': 64, '貌': 65, '赤': 66,
                     '心': 67, '报': 68, '国': 69, '缓': 70, '兵': 71, '计': 72, '慌': 73, '张': 74, '皇': 75, '亲': 76, '戚': 77,
                     '稀': 78, '年': 79, '双': 80, '无': 81, '至': 82, '臭': 83, '可': 84, '当': 85, '错': 86, '综': 87, '复': 88,
                     '杂': 89, '小': 90, '精': 91, '悍': 92, '灰': 93, '丧': 94, '气': 95, '一': 96, '炬': 97, '华': 98, '而': 99,
                     '实': 100, '话': 101, '投': 102, '机': 103}

        for folder in os.listdir(root):
            folders = root + "/" + folder
            if os.path.isdir(folders):
                file_names = glob(folders + '/*.jpg')
                for file in file_names:
                    self.filenames.append(file)
                    self.labels.append(train_dic[folder])

    def __getitem__(self, index):
        img_name = self.filenames[index]
        label = self.labels[index]

        img1 = Image.open(img_name)
        # new_img=img1.convert('RGB')
        # new_img = img1.resize((224, 224))
        new_img = img1.resize((32, 32))
        img = self.transform(new_img)  # 可以根据指定的转化形式对数据集进行转换

        # return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容
        return img, label

    def __len__(self):
        return len(self.filenames)
