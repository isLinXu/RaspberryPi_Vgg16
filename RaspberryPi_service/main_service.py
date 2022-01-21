import torch
import os
from PIL import Image
import cv2
import numpy as np
import time
from flask import request, Flask
import json
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import mydatasets
from vgg import vgg
import argparse

app = Flask(__name__)
app.config.update(RESTFUL_JSON=dict(ensure_ascii=False))

class Recognition():
    def __init__(self, MODEL_NAME):
        checkpoint = torch.load(MODEL_NAME+"pruned.pth.tar",map_location=torch.device('cpu'))
        self.model = vgg(cfg=checkpoint['cfg'])
        self.model.load_state_dict(checkpoint['state_dict'])
        optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

        checkpoint = torch.load(MODEL_NAME+"model_best.pth.tar",map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        self.model.eval()
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        with open("classes.txt", "r") as f:
            self.categories = [s.strip() for s in f.readlines()]
    def predict(self, input_image, filename="1.png"):
        new_img = input_image.resize((32, 32))
        img = self.transform(new_img)
        data = Variable(torch.unsqueeze(img, dim=0).float())
        output = self.model(data)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top5_prob, top5_catid = torch.topk(probabilities, 1)
        # value_ = 5
        # print(top5_catid[0])
        print(self.categories[top5_catid[0]], top5_prob[0].item())
        # for i in range(top5_prob.size(0)):
        #     print(top5_catid[i])
        return self.categories[top5_catid[0]], top5_prob[0].item()
@app.route("/upload_image", methods=['POST'])
def upload_image():
    global detection
    filename = request.form['filename']
    upload_file = request.files['file']
    file_name = "image/" + filename
    # file_paths = os.path.join(file_path, file_name)
    # 保存接收的图片到桌面
    upload_file.save(file_name)
    input_image = Image.open(file_name)
    result_name,result_id=detection.predict(input_image)
    info = {"code": "0",
            "info": {"name":result_name},
            "error": "",
            "msg": ""}
    return json.dumps(info, ensure_ascii=False)
def main():
    global detection
    detection = Recognition('./')
    #################测试##################
    app.run("0.0.0.0", port=10088, debug=False)  # 端口为8081


if __name__ == "__main__":
    main()