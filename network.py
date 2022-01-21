import torch.nn as nn
import torchvision.models as models
class feature_net(nn.Module):
    def __init__(self, model, dim, n_classes):
        super(feature_net, self).__init__()

        if model == 'vgg':
            vgg = models.vgg19(pretrained=True)
            self.feature = nn.Sequential(*list(vgg.children())[:-1])
            self.feature.add_module('global average', nn.AvgPool2d(3))


        #选择resnet50网络模型
        if model == 'resnet50':
            resnet = models.resnet50(pretrained=True)
            self.feature = nn.Sequential(*list(resnet.children())[:-1])
            print("resnet:",resnet)
            print("--------------------------------------------------------------------------------------")

        # # 选择inceptionv3网络模型
        #     if model == 'inceptionv3':
        #         inception = models.inception_v3(pretrained=True)
        #         self.feature = nn.Sequential(*list(inception.children())[:-1])
        #         self.feature._modules.pop('13')
        #         self.feature.add_module('global average', nn.AvgPool2d(18))


        self.classifier = nn.Sequential(
            nn.Linear(dim, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, n_classes)
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
# model = feature_net('vgg19',10,2)
model = feature_net('resnet50',10,2)  #model='resnet50',dim=10，n_classes=2
# model = feature_net('inceptionv3',10,2)  #model='resnet50',dim=10，n_classes=2
print(model)
