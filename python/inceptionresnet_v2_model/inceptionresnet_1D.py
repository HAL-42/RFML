#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: inceptionresnet_1D.py
@time: 2019/9/2 8:33
@desc:
"""
import thop
import torch
import torch.nn as nn
import torch.nn.functional as F
from my_py_tools.quick_init import QuickInit


class BasicConv1d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False) # verify bias false
        self.bn = nn.BatchNorm1d(out_planes,
                                 eps=0.001, # value found in tensorflow
                                 momentum=0.1, # default pytorch value
                                 affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Stem(nn.Module):

    def __init__(self, num_input_channels):
        super(Stem, self).__init__()
        # * Def Reduction0 in stem
        self.red0_pad = nn.ConstantPad1d((0, 1), 0)
        self.red0_bconv = BasicConv1d(num_input_channels, 32, 3, stride=2)
        # * Def Reduction1 in stem
        self.red1_bconv1 = BasicConv1d(32, 32, 3)
        self.red1_bconv2 = BasicConv1d(32, 64, 3, padding=1)
        self.red1_pad = nn.ConstantPad1d((0, 1), 0)
        self.red1_branch0 = nn.MaxPool1d(3, stride=2)
        self.red1_branch1 = BasicConv1d(64, 96, 3, stride=2)
        # * Def Reduction2 in stem
        self.red2_branch0 = nn.Sequential(
            BasicConv1d(160, 64, 1),
            BasicConv1d(64, 96, 3)
        )
        self.red2_branch1 = nn.Sequential(
            BasicConv1d(160, 64, 1),
            BasicConv1d(64, 64, 7, padding=3),
            BasicConv1d(64, 96, 3)
        )
        # * Def Reduction3 in stem
        self.red3_branch0 = nn.MaxPool1d(3, stride=2)
        self.red3_branch1 = BasicConv1d(192, 128, 3, stride=2)

    def forward(self, x):
        # * Red0
        x = self.red0_pad(x)
        x = self.red0_bconv(x)
        # * Red1
        x = self.red1_bconv1(x)
        x = self.red1_bconv2(x)
        x = self.red1_pad(x)
        x0 = self.red1_branch0(x)
        x1 = self.red1_branch1(x)
        x = torch.cat((x0, x1), dim=1)
        # * Red2
        x0 = self.red2_branch0(x)
        x1 = self.red2_branch1(x)
        x = torch.cat((x0, x1), dim=1)
        # * Red3
        x0 = self.red3_branch0(x)
        x1 = self.red3_branch1(x)
        x = torch.cat((x0, x1), dim=1)
        return x


class InceptionA(nn.Module):

    def __init__(self, scale=1.0):
        super(InceptionA, self).__init__()

        self.scale = scale

        self.branch0 = BasicConv1d(320, 32, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv1d(320, 32, kernel_size=1, stride=1),
            BasicConv1d(32, 32, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv1d(320, 32, kernel_size=1, stride=1),
            BasicConv1d(32, 48, kernel_size=3, stride=1, padding=1),
            BasicConv1d(48, 64, kernel_size=3, stride=1, padding=1)
        )

        self.conv = nn.Conv1d(128, 320, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class ReductionA(nn.Module):

    def __init__(self):
        super(ReductionA, self).__init__()
        self.pad = nn.ConstantPad1d((0, 1), 0)
        
        self.branch0 = BasicConv1d(320, 384, kernel_size=3, stride=2)

        self.branch1 = nn.Sequential(
            BasicConv1d(320, 256, kernel_size=1, stride=1),
            BasicConv1d(256, 256, kernel_size=3, stride=1, padding=1),
            BasicConv1d(256, 384, kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool1d(3, stride=2)

    def forward(self, x):
        x = self.pad(x)
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class InceptionB(nn.Module):

    def __init__(self, scale=1.0):
        super(InceptionB, self).__init__()
        self.pad = nn.ConstantPad1d((0, 1), 0)
        
        self.scale = scale

        self.branch0 = BasicConv1d(1088, 192, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv1d(1088, 128, kernel_size=1, stride=1),
            BasicConv1d(128, 192, kernel_size=7, stride=1, padding=3)
        )

        self.conv = nn.Conv1d(384, 1088, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.pad(x)
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out
 
    
class ReductionBPlus(nn.Module):

    def __init__(self):
        super(ReductionBPlus, self).__init__()
        # Reduction 0
        self.red0_branch0 = nn.Sequential(
            BasicConv1d(1088, 256, kernel_size=1, stride=1),
            BasicConv1d(256, 384, kernel_size=3, stride=2)
        )
        self.red0_branch1 = nn.Sequential(
            BasicConv1d(1088, 256, kernel_size=1, stride=1),
            BasicConv1d(256, 288, kernel_size=3, stride=2)
        )
        self.red0_branch2 = nn.Sequential(
            BasicConv1d(1088, 256, kernel_size=1, stride=1),
            BasicConv1d(256, 288, kernel_size=3, stride=1, padding=1),
            BasicConv1d(288, 320, kernel_size=3, stride=2)
        )
        self.red0_branch3 = nn.MaxPool1d(3, stride=2)
        # Reduction 1
        self.red1_pad = nn.ConstantPad1d((0, 1), 0)
        self.red1_branch0 = nn.Sequential(
            BasicConv1d(2080, 520, 1),
            nn.MaxPool1d(3, stride=2)
        )
        self.red1_branch1 = nn.Sequential(
            BasicConv1d(2080, 256, 1),
            BasicConv1d(256, 520, 3, stride=2)
        )
        self.red1_branch2 = nn.Sequential(
            BasicConv1d(2080, 256, 1),
            BasicConv1d(256, 520, 3, stride=2)
        )
        self.red1_branch3 = nn.Sequential(
            BasicConv1d(2080, 256, 1),
            BasicConv1d(256, 520, 3, stride=2)
        )

    def forward(self, x):
        x0 = self.red0_branch0(x)
        x1 = self.red0_branch1(x)
        x2 = self.red0_branch2(x)
        x3 = self.red0_branch3(x)
        x = torch.cat((x0, x1, x2, x3), 1)
        x = self.red1_pad(x)
        x0 = self.red1_branch0(x)
        x1 = self.red1_branch1(x)
        x2 = self.red1_branch2(x)
        x3 = self.red1_branch3(x)
        x = torch.cat((x0, x1, x2, x3), dim=1)
        return x


class InceptionC(nn.Module):

    def __init__(self, scale=1.0, noReLU=False):
        super(InceptionC, self).__init__()

        self.scale = scale
        self.noReLU = noReLU

        self.branch0 = BasicConv1d(2080, 192, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv1d(2080, 192, kernel_size=1, stride=1),
            BasicConv1d(192, 256, kernel_size=3, stride=1, padding=1)
        )

        self.conv = nn.Conv1d(448, 2080, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out


class ReductionCPlus(nn.Module):

    def __init__(self):
        super(ReductionCPlus, self).__init__()
        # * Def Reduction 0
        self.red0_branch0 = nn.Sequential(
            nn.MaxPool1d(3, stride=2),
            BasicConv1d(2080, 520, 1)
        )
        self.red0_branch1 = nn.Sequential(
            BasicConv1d(2080, 520, 1),
            BasicConv1d(520, 390, 3, padding=1),
            BasicConv1d(390, 260, 3, stride=2)
        )
        self.red0_branch2 = nn.Sequential(
            BasicConv1d(2080, 520, 1),
            BasicConv1d(520, 260, 3, stride=2)
        )
        # * Def Reduction 1
        self.red1_pad = nn.ConstantPad1d((0, 1), 0)
        self.red1_branch0 = nn.Sequential(
            BasicConv1d(1040, 270, 1),
            nn.AvgPool1d(2, padding=1, count_include_pad=False)
        )
        self.red1_branch1=nn.Sequential(
            BasicConv1d(1040, 270, 1),
            nn.ConstantPad1d((0, 1), 0),
            BasicConv1d(270, 270, 3, stride=2, padding=1)
        )
        # * Def Reduction 2
        self.red2_avg_pool8 = nn.AvgPool1d(8, count_include_pad=False)
        self.red2_bconv = BasicConv1d(540, 250, 1)

    def forward(self, x):
        # * Red0
        x0 = self.red0_branch0(x)
        x1 = self.red0_branch1(x)
        x2 = self.red0_branch2(x)
        x = torch.cat((x0, x1, x2), dim=1)
        # * Red1
        x0 = self.red1_branch0(x)
        x1 = self.red1_branch1(x)
        x = torch.cat((x0, x1), dim=1)
        # * Red2
        x = self.red2_avg_pool8(x)
        x = self.red2_bconv(x)
        x = x.view(x.shape[0], -1)
        return x


class InceptionResNet1D(nn.Module):

    def __init__(self, num_classes, num_input_channels=1, len_sample=10000, batch_size=None,
                 num_incept_A=10, num_incept_B=20, num_incept_C=10,
                 scale_A=0.17, scale_B=0.1, scale_C=0.2,
                 mean = None):
        super(InceptionResNet1D, self).__init__()
        # * Special attributes
        self.model_init_dict = QuickInit(self, locals())

        self.input_size = (batch_size, num_input_channels, len_sample)

        # * Modules
        self.stem = Stem(num_input_channels)
        # ** A
        self.inception_A_units = [InceptionA(scale=scale_A) for i in range(num_incept_A)]
        self.inception_A = nn.Sequential(*self.inception_A_units)
        self.reduce_A = ReductionA()
        # ** B
        self.inception_B_units = [InceptionB(scale=scale_B) for i in range(num_incept_B)]
        self.inception_B = nn.Sequential(*self.inception_B_units)
        self.reduce_B_plus = ReductionBPlus()
        # ** C
        self.relu_inception_C_units = [InceptionC(scale=scale_C) for i in range(num_incept_C - 1)]
        self.relu_inception_C = nn.Sequential(*self.relu_inception_C_units)
        self.no_relu_inception_C = InceptionC(scale=scale_C, noReLU=True)
        self.reduce_C_plus = ReductionCPlus()
        # ** Generate logits
        self.logits_linear = nn.Linear(1250, num_classes)

    def forward(self, x):
        if self.mean:
            x -= self.mean
        # * stem
        x = self.stem(x)
        # * A
        x = self.inception_A(x)
        x = self.reduce_A(x)
        # * B
        x = self.inception_B(x)
        x = self.reduce_B_plus(x)
        # * C
        x = self.relu_inception_C(x)
        x = self.no_relu_inception_C(x)
        x = self.reduce_C_plus(x)
        # * Generate logits
        x = self.logits_linear(x)
        return x

    def _get_logits(self, input):
        # assert input.shape[1:] == self.input_size[1:], "input.size() != self.input_size."
        return self.forward(input)

    def _get_PR_classes(self, input, need_logits=False):
        logits = self._get_logits(input)
        if need_logits:
            return F.softmax(input, dim=1), logits
        else:
            return F.softmax(input, dim=1)

    def get_cross_entropy_loss(self, input, target, is_expanded_target=True, need_PR=False):
        if is_expanded_target:
            target = torch.argmax(target, dim=1)
        logits = self._get_logits(input)
        if need_PR:
            return F.cross_entropy(logits, target), F.softmax(torch.detach(logits), dim=1)
        else:
            return F.cross_entropy(logits, target)

    def predict(self, input, return_PR=False):
        with torch.no_grad:
            if return_PR:
                return self._get_PR_classes(input)
            else:
                PR_classes = self._get_PR_classes(input)
                predict = torch.argmax(PR_classes).numpy()
                return predict


if __name__ == "__main__":
    net = InceptionResNet1D(43, num_input_channels=1, batch_size=1,
                            num_incept_A=5, num_incept_B=10, num_incept_C=5,
                            scale_A=0.34, scale_B=0.2, scale_C=0.4)
    input = torch.randn(net.input_size)
    net(input)
    flops, params = thop.profile(net, inputs=(input, ))

    for name, child in net.named_children():
        param_num = 0
        for param in child.parameters():
            param_num += param.data.numel()
        print(f"Param num of {name} \n {param_num}")

    print(f"FLOPs:{flops} ------- params num:{params}")