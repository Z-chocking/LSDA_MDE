import torch
import torch.nn as nn
import torch.nn.functional as F

from .swin_transformer import SwinTransformer
from .newcrf_layers import NewCRF
from .uper_crf_head import PSP

from .CAM import CAM

from  .CBAM import CBAMBlock

########################################################################################################################


class NewCRFDepth(nn.Module):
    """
    Depth network based on neural window FC-CRFs architecture.
    """
    def __init__(self, version=None, inv_depth=False, pretrained=None, 
                    frozen_stages=-1, min_depth=0.1, max_depth=100.0, **kwargs):
        super().__init__()

        self.inv_depth = inv_depth
        self.with_auxiliary_head = False
        self.with_neck = False

        norm_cfg = dict(type='BN', requires_grad=True)
        # norm_cfg = dict(type='GN', requires_grad=True, num_groups=8)

        window_size = int(version[-2:])

        if version[:-2] == 'base':
            embed_dim = 128
            depths = [2, 2, 18, 2]
            num_heads = [4, 8, 16, 32]
            in_channels = [128, 256, 512, 1024]
        elif version[:-2] == 'large':
            embed_dim = 192
            depths = [2, 2, 18, 2]
            num_heads = [6, 12, 24, 48]
            in_channels = [192, 384, 768, 1536]
        elif version[:-2] == 'tiny':
            embed_dim = 96
            depths = [2, 2, 6, 2]
            num_heads = [3, 6, 12, 24]
            in_channels = [96, 192, 384, 768]

        self.en_channels =in_channels

        backbone_cfg = dict(
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            ape=False,
            drop_path_rate=0.3,
            patch_norm=True,
            use_checkpoint=False,
            frozen_stages=frozen_stages
        )

        embed_dim = 512
        decoder_cfg = dict(
            in_channels=in_channels,
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=embed_dim,
            dropout_ratio=0.0,
            num_classes=32,
            norm_cfg=norm_cfg,
            align_corners=False
        )

        self.backbone = SwinTransformer(**backbone_cfg)
        v_dim = decoder_cfg['num_classes']*4


        cam_dims = [1024, 512, 256, 128]  # CAM前卷积后的通道数
        # q_dims = [1536, 768, 384, 192]  # swin-large
        q_dims = in_channels[::-1]
        v_dims = [512, 256, 128, 64]  # K通道数
        cam_n_h = [32, 16, 8, 4]  # 头数
        lsda_flag = [0, 0, 1, 1]
        self.cam4 = CAM(q_dim=v_dims[0], kv_dim=q_dims[0], embed_dim=cam_dims[0], lsda_flag=lsda_flag[0], group_size=7,
                        num_heads=cam_n_h[0], interval=8)
        self.cam3 = CAM(q_dim=v_dims[1], kv_dim=q_dims[1], embed_dim=cam_dims[1], lsda_flag=lsda_flag[1], group_size=7,
                        num_heads=cam_n_h[1], interval=8)
        self.cam2 = CAM(q_dim=v_dims[2], kv_dim=q_dims[2], embed_dim=cam_dims[2], lsda_flag=lsda_flag[2], group_size=7,
                        num_heads=cam_n_h[2], interval=8)
        self.cam1 = CAM(q_dim=v_dims[3], kv_dim=q_dims[3], embed_dim=cam_dims[3], lsda_flag=lsda_flag[3], group_size=7,
                        num_heads=cam_n_h[3], interval=8)



        self.decoder = PSP(**decoder_cfg)
        self.disp_head1 = DispHead(input_dim=cam_dims[3])

        inp_dim = 0
        for i in range(len(q_dims)):
            inp_dim = inp_dim + q_dims[i]
        oup_dim = inp_dim


        # CBAM
        self.multi_scale_en = CBAMBlock(channel=inp_dim, reduction=16, kernel_size=7)


        self.up_mode = 'bilinear'
        if self.up_mode == 'mask':
            self.mask_head = nn.Sequential(
                nn.Conv2d(crf_dims[0], 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 16*9, 1, padding=0))

        self.min_depth = min_depth
        self.max_depth = max_depth

        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        print(f'== Load encoder backbone from: {pretrained}')
        self.backbone.init_weights(pretrained=pretrained)
        self.decoder.init_weights()
        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head, nn.ModuleList):
                for aux_head in self.auxiliary_head:
                    aux_head.init_weights()
            else:
                self.auxiliary_head.init_weights()

    def upsample_mask(self, disp, mask):
        """ Upsample disp [H/4, W/4, 1] -> [H, W, 1] using convex combination """
        N, _, H, W = disp.shape
        mask = mask.view(N, 1, 9, 4, 4, H, W)
        mask = torch.softmax(mask, dim=2)

        up_disp = F.unfold(disp, kernel_size=3, padding=1)
        up_disp = up_disp.view(N, 1, 9, 1, 1, H, W)

        up_disp = torch.sum(mask * up_disp, dim=2)
        up_disp = up_disp.permute(0, 1, 4, 2, 5, 3)
        return up_disp.reshape(N, 1, 4*H, 4*W)

    def forward(self, imgs):

        feats = self.backbone(imgs)
        if self.with_neck:
            feats = self.neck(feats)


        ppm_out = self.decoder(feats)

        # 处理编码器特征，以作为层级特征输入
        # 组装特征
        new_feats = []
        for i in range(3, 0, -1):
            input = F.interpolate(feats[i], scale_factor=2 ** i, mode="bilinear", align_corners=False)
            new_feats.append(input)
        new_feats.append(feats[0])
        new_feats = new_feats[::-1]
        input = torch.cat(new_feats, dim=1)

        output = self.multi_scale_en(input)

        # 还原分辨率
        af_feats = []
        layer1 = nn.AvgPool2d(2, stride=2)
        f0, f1, f2, f3 = torch.split(output, self.en_channels, dim=1)
        af_feats.extend([f0, f1, f2, f3])
        for i in range(1, 4):
            af_feats[i] = F.interpolate(af_feats[i], scale_factor=(2 ** -i), mode="bilinear", align_corners=False)

        e3 = self.cam4(ppm_out, af_feats[3], af_feats[3].size()[2], af_feats[3].size()[3])
        e3 = nn.PixelShuffle(2)(e3)
        e2 = self.cam3(e3, af_feats[2], af_feats[2].size()[2], af_feats[2].size()[3])
        e2 = nn.PixelShuffle(2)(e2)
        e1 = self.cam2(e2, af_feats[1], af_feats[1].size()[2], af_feats[1].size()[3])
        e1 = nn.PixelShuffle(2)(e1)
        e0 = self.cam1(e1, af_feats[0], af_feats[0].size()[2], af_feats[0].size()[3])


        if self.up_mode == 'mask':
            mask = self.mask_head(e0)
            d1 = self.disp_head1(e0, 1)
            d1 = self.upsample_mask(d1, mask)
        else:
            d1 = self.disp_head1(e0, 4)

        depth = d1 * self.max_depth

        return depth


class DispHead(nn.Module):
    def __init__(self, input_dim=100):
        super(DispHead, self).__init__()
        # self.norm1 = nn.BatchNorm2d(input_dim)
        self.conv1 = nn.Conv2d(input_dim, 1, 3, padding=1)
        # self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, scale):
        # x = self.relu(self.norm1(x))
        x = self.sigmoid(self.conv1(x))
        if scale > 1:
            x = upsample(x, scale_factor=scale)
        return x


class DispUnpack(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=128):
        super(DispUnpack, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 16, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.pixel_shuffle = nn.PixelShuffle(4)

    def forward(self, x, output_size):
        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x)) # [b, 16, h/4, w/4]
        # x = torch.reshape(x, [x.shape[0], 1, x.shape[2]*4, x.shape[3]*4])
        x = self.pixel_shuffle(x)

        return x


def upsample(x, scale_factor=2, mode="bilinear", align_corners=False):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=align_corners)