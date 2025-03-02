import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F


# resnet对比学习
class Resnet152(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Resnet152, self).__init__()
        res_pretrained_features = models.resnet152(pretrained=True)
        self.slice1 = torch.nn.Sequential(*list(res_pretrained_features.children())[:-5])
        self.slice2 = torch.nn.Sequential(*list(res_pretrained_features.children())[-5:-4])
        self.slice3 = torch.nn.Sequential(*list(res_pretrained_features.children())[-4:-3])
        self.slice4 = torch.nn.Sequential(*list(res_pretrained_features.children())[-3:-2])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        return [h_relu1, h_relu2, h_relu3, h_relu4]


class ContrastLoss_res_formal(nn.Module):
    def __init__(self, ablation=False):
        super(ContrastLoss_res_formal, self).__init__()
        self.vgg = Resnet152().cuda()
        self.l1 = nn.L1Loss()
        self.weights2 = [1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.ab = ablation

    def forward(self, a, p, n1):
        # 分别是对比1，输入，对比2
        a_vgg, p_vgg, n1_vgg = self.vgg(a), self.vgg(p), self.vgg(n1)
        loss = 0
        for i in range(len(a_vgg)):
            d_ap = self.l1(a_vgg[i], p_vgg[i].detach())
            if not self.ab:
                d_an1 = self.l1(a_vgg[i], n1_vgg[i].detach())
                contrastive = d_ap / (d_an1 + 1e-7)
            else:
                contrastive = d_ap

            loss += self.weights2[i] * contrastive
        return loss


class ContrastLoss_res(nn.Module):
    def __init__(self, weights, ablation=False):
        super(ContrastLoss_res, self).__init__()
        self.vgg = Resnet152().cuda()
        self.l1 = nn.L1Loss()
        self.weights = weights
        self.weights2 = [1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.ab = ablation

    def forward(self, a, p, n1, n2, n3):
        # 分别是对比1，输入，对比2
        a_vgg, p_vgg, n1_vgg = self.vgg(a), self.vgg(p), self.vgg(n1)
        n2_vgg, n3_vgg = self.vgg(n2), self.vgg(n3)
        loss = 0
        for i in range(len(a_vgg)):
            d_ap = self.l1(a_vgg[i], p_vgg[i].detach())
            if not self.ab:
                d_an1 = self.l1(a_vgg[i], n1_vgg[i].detach())
                d_an2 = self.l1(a_vgg[i], n2_vgg[i].detach())
                d_an3 = self.l1(a_vgg[i], n3_vgg[i].detach())
                contrastive = d_ap / (d_an1 + d_an2 * self.weights[0] + d_an3 * self.weights[1] + 1e-7)
            else:
                contrastive = d_ap

            loss += self.weights2[i] * contrastive
        return loss