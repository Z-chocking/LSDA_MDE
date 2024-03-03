import numpy as np
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from thop import profile



class ChannelAttention(nn.Module):
    def __init__(self,channel,reduction=16):
        super().__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.se=nn.Sequential(
            nn.Conv2d(channel,channel//reduction,1,bias=False),
            nn.ReLU(),
            nn.Conv2d(channel//reduction,channel,1,bias=False)
        )
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result=self.maxpool(x)
        avg_result=self.avgpool(x)
        max_out=self.se(max_result)
        avg_out=self.se(avg_result)
        output=self.sigmoid(max_out+avg_out)
        return output

class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=7):
        super().__init__()
        self.conv=nn.Conv2d(2,1,kernel_size=kernel_size,padding=kernel_size//2)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result,_=torch.max(x,dim=1,keepdim=True) # 沿通道做maxpool
        avg_result=torch.mean(x,dim=1,keepdim=True) # 沿通道做avgpool
        result=torch.cat([max_result,avg_result],1)
        output=self.conv(result)
        output=self.sigmoid(output)
        return output



class CBAMBlock(nn.Module):

    def __init__(self, channel=512,reduction=16,kernel_size=49):
        super().__init__()
        self.ca=ChannelAttention(channel=channel,reduction=reduction)
        self.sa=SpatialAttention(kernel_size=kernel_size)



    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual=x
        out=x*self.ca(x)
        out=out*self.sa(out)
        return out+residual


if __name__ == '__main__':
    # input=torch.randn(50,512,7,7)
    # kernel_size=input.shape[2]
    # cbam = CBAMBlock(channel=512,reduction=16,kernel_size=kernel_size)
    # output=cbam(input)
    # print(output.shape)
    feats = []
    for i in range(1, 5):
        input = torch.randn(1, 192 * 2 ** (i - 1), 480 // (2 ** (i + 1)), 640 // (2 ** (i + 1)))
        feats.append(input)

    # 组装特征
    new_feats = []
    for i in range(3, 0, -1):
        input = F.interpolate(feats[i], scale_factor=2 ** i, mode="bilinear", align_corners=False)
        new_feats.append(input)
    new_feats.append(feats[0])
    new_feats = new_feats[::-1]
    input = torch.cat(new_feats, dim=1)

    cbam = CBAMBlock(channel=2880, reduction=16, kernel_size=7)
    output = cbam(input)
    print(output.shape)
    flops, params = profile(cbam, inputs=(input,))
    print(flops)
    print(params)

    # 还原分辨率
    af_feats = []
    layer1 = nn.AvgPool2d(2, stride=2)
    f0, f1, f2, f3 = torch.split(output, [192, 384, 768, 1536], dim=1)
    af_feats.extend([f0, f1, f2, f3])
    for i in range(1, 4):
        af_feats[i] = F.interpolate(af_feats[i], scale_factor=(2**-i), mode="bilinear", align_corners=False)