import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

NEG_INF = -1000000

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DynamicPosBias(nn.Module):
    def __init__(self, dim, num_heads, residual):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )
    def forward(self, biases):
        if self.residual:
            pos = self.pos_proj(biases) # 2Gh-1 * 2Gw-1, heads
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos


class Attention(nn.Module):
    r""" Multi-head self attention module with relative position bias.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 position_bias=True):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.position_bias = position_bias
        if self.position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads, residual=False)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, H, W, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Gh*Gw, Gh*Gw) or None
        """
        group_size = (H, W)
        B_, N, C = q.shape
        assert H*W == N
        q = self.q(q).reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        kv = self.kv(k).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q=q[0]
        k, v = kv[0], kv[1]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)) # (B, self.num_heads, N, N), N = H*W

        if self.position_bias:
            # generate mother-set
            position_bias_h = torch.arange(1 - group_size[0], group_size[0], device=attn.device)
            position_bias_w = torch.arange(1 - group_size[1], group_size[1], device=attn.device)
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))  # 2, 2Gh-1, 2Gw-1 -> 第一个矩阵为h，第二个矩阵为w
            biases = biases.flatten(1).transpose(0, 1).contiguous().float() # (2Gh-1*2Gw-1), 2 ->第一列为h，且先按h遍历

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(group_size[0], device=attn.device)
            coords_w = torch.arange(group_size[1], device=attn.device)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Gh, Gw
            coords_flatten = torch.flatten(coords, 1)  # 2, Gh*Gw
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Gh*Gw, Gh*Gw 列行相减
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Gh*Gw, Gh*Gw, 2
            relative_coords[:, :, 0] += group_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += group_size[1] - 1 # 行列各加M-1
            relative_coords[:, :, 0] *= 2 * group_size[1] - 1 # 行*2M-1
            relative_position_index = relative_coords.sum(-1)  # Gh*Gw, Gh*Gw ->行+列

            pos = self.pos(biases) # 2Gh-1 * 2Gw-1, heads
            # select position bias
            relative_position_bias = pos[relative_position_index.view(-1)].view(
                group_size[0] * group_size[1], group_size[0] * group_size[1], -1)  # Gh*Gw,Gh*Gw,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Gh*Gw, Gh*Gw
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nG = mask.shape[0]
            attn = attn.view(B_ // nG, nG, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0) # (B, nG, nHead, N, N)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C) # Bn_w,g_s^2,C
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CAMBlock(nn.Module):
    r""" CrossFormer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        group_size (int): Window size.
        lsda_flag (int): use SDA or LDA, 0 for SDA and 1 for LDA.Default: 0
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, group_size=7, interval=8, lsda_flag=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.group_size = group_size
        self.interval = interval
        self.lsda_flag = lsda_flag
        self.mlp_ratio = mlp_ratio

        self.norm_q = norm_layer(dim)
        self.norm_k = norm_layer(dim)

        self.attn = Attention(
            dim, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            position_bias=True)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, q, k, H, W):
        """
        Args:
            q: patch, with size (B,n_p,C)
            k: patch, with size (B,n_p,C)
            H: Height for x
            W: width for x

        Returns:
            patch, with size (B,n_p,C)
        """
        B, L, C = q.shape
        assert L == H * W, "input feature has wrong size %d, %d, %d" % (L, H, W)

        if min(H, W) <= self.group_size:
            # if window size is larger than input resolution, we don't partition windows
            self.lsda_flag = 0
            self.group_size = min(H, W)

        shortcut_q = q # 暂时没用到
        q = self.norm_q(q)
        q = q.view(B, H, W, C)

        shortcut_k = k
        k = self.norm_k(k)
        k = k.view(B, H, W, C)


        # padding 给patch打补丁以适配group_size
        size_div = self.interval if self.lsda_flag == 1 else self.group_size  # 就是w_s：SDA时group_size；LDA时interval
        pad_l = pad_t = 0
        pad_r = (size_div - W % size_div) % size_div
        pad_b = (size_div - H % size_div) % size_div
        q = F.pad(q, (0, 0, pad_l, pad_r, pad_t, pad_b))
        k = F.pad(k, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = q.shape

        mask = torch.zeros((1, Hp, Wp, 1), device=q.device)
        if pad_b > 0:
            mask[:, -pad_b:, :, :] = -1
        if pad_r > 0:
            mask[:, :, -pad_r:, :] = -1

        # group embeddings and generate attn_mask
        if self.lsda_flag == 0:  # SDA
            G = Gh = Gw = self.group_size
            # Patch->Window
            q = q.reshape(B, Hp // G, G, Wp // G, G, C).permute(0, 1, 3, 2, 4, 5).contiguous()  # B, Hp/G, Wp/G, G, G, C
            q = q.reshape(B * Hp * Wp // G ** 2, G ** 2, C)
            k = k.reshape(B, Hp // G, G, Wp // G, G, C).permute(0, 1, 3, 2, 4, 5).contiguous()  # B, Hp/G, Wp/G, G, G, C
            k = k.reshape(B * Hp * Wp // G ** 2, G ** 2, C)
            nG = Hp * Wp // G ** 2  # n_w
            # attn_mask
            if pad_r > 0 or pad_b > 0:
                mask = mask.reshape(1, Hp // G, G, Wp // G, G, 1).permute(0, 1, 3, 2, 4,
                                                                          5).contiguous()  # 1,Hp/G,Wp/G,G,G,1
                mask = mask.reshape(nG, 1, G * G)
                attn_mask = torch.zeros((nG, G * G, G * G), device=q.device)
                attn_mask = attn_mask.masked_fill(mask < 0, NEG_INF)  # 权重蒙版。与swin的不同，其只遮住pad部分的权重
            else:
                attn_mask = None
        else:  # LDA
            I, Gh, Gw = self.interval, Hp // self.interval, Wp // self.interval
            # patch->window 有点像patchMerging的第一步
            q = q.reshape(B, Gh, I, Gw, I, C).permute(0, 2, 4, 1, 3, 5).contiguous()  # B, I, I, Gh, Gw, C
            q = q.reshape(B * I * I, Gh * Gw, C)
            k = k.reshape(B, Gh, I, Gw, I, C).permute(0, 2, 4, 1, 3, 5).contiguous()  # B, I, I, Gh, Gw, C
            k = k.reshape(B * I * I, Gh * Gw, C)
            nG = I ** 2
            # attn_mask
            if pad_r > 0 or pad_b > 0:
                mask = mask.reshape(1, Gh, I, Gw, I, 1).permute(0, 2, 4, 1, 3, 5).contiguous()
                mask = mask.reshape(nG, 1, Gh * Gw)
                attn_mask = torch.zeros((nG, Gh * Gw, Gh * Gw), device=q.device)
                attn_mask = attn_mask.masked_fill(mask < 0, NEG_INF)
            else:
                attn_mask = None

        # multi-head self-attention
        x = self.attn(q, k, Gh, Gw, mask=attn_mask)  # nG*B, G*G, C

        # ungroup embeddings
        # 不完全Patch->window
        if self.lsda_flag == 0:
            x = x.reshape(B, Hp // G, Wp // G, G, G, C).permute(0, 1, 3, 2, 4,
                                                                5).contiguous()  # B, Hp//G, G, Wp//G, G, C
        else:
            x = x.reshape(B, I, I, Gh, Gw, C).permute(0, 3, 1, 4, 2, 5).contiguous()  # B, Gh, I, Gw, I, C
        x = x.reshape(B, Hp, Wp, C)

        # remove padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut_k + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class CAM(nn.Module):
    """ CAM.

    Args:
        q_dim (int): Number of input channels for Q.
        kv_dim (int): Number of input channels for K.
        embed_dim (int): The numbers of channels that wants to be fixed in.
        num_heads (int): Number of attention heads.
        group_size (int): Group size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(self, q_dim=96, kv_dim=96, embed_dim=512, lsda_flag=0, group_size=7, num_heads=2, interval=8,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, norm_layer=nn.LayerNorm
                 ):

        super().__init__()
        self.embed_dim = embed_dim
        self.drop_path = drop_path

        # fix channel
        # self.proj_q = nn.Conv2d(q_dim, embed_dim, 3, padding=1)
        # self.proj_kv = nn.Conv2d(kv_dim, embed_dim, 3, padding=1)

        self.proj_q = nn.Conv2d(q_dim, embed_dim, 3, padding=1)
        self.proj_kv_1 = nn.Conv2d(kv_dim, embed_dim//2, 3, padding=1)
        self.proj_kv_2 = nn.Conv2d(embed_dim//2, embed_dim, 3, padding=1)


        # build blocks
        self.blocks = CAMBlock(dim=embed_dim,
                             num_heads=num_heads, group_size=group_size, interval=interval,
                             lsda_flag=lsda_flag,
                             mlp_ratio=mlp_ratio,
                             qkv_bias=qkv_bias, qk_scale=qk_scale,
                             drop=drop, attn_drop=attn_drop,
                             drop_path=drop_path,
                             norm_layer=norm_layer,
                             )



        # 类mlp
        self.converge1 = nn.Conv2d(embed_dim,embed_dim//2,3,1,1)
        self.converge2 = nn.Conv2d(embed_dim//2, embed_dim , 3, 1, 1)
        self.act = nn.GELU()

        layer = norm_layer(embed_dim)
        layer_name = 'norm_cam'
        self.add_module(layer_name, layer)


    def forward(self, q, k, H, W):
        """
        Args:
            q: Feature with size(B,C,H,W)
            k: Feature with size(B,C,H,W)

        Returns:
            with size((B,C,H,W))

        """



        q=self.proj_q(q)
        k=self.proj_kv_2((self.proj_kv_1(k)))

        q_proj=q
        k_proj=k

        # converge = torch.cat((q,k),dim=1)
        converge = q + k
        converge = self.converge1(converge)
        converge  = self.act(converge)
        converge = self.converge2(converge)
        converge = self.act(converge)
        # patch 化（可改进）
        # Ph, Pw = q.size(2), q.size(3)
        # q = q.flatten(2).transpose(1, 2)
        # k = k.flatten(2).transpose(1, 2)
        converge = converge.flatten(2).transpose(1, 2)

        # LDA/SDA
        output = self.blocks(converge, converge, H, W)

        norm_layer = getattr(self, f'norm_cam')
        output = norm_layer(output)
        output = output.view(-1, H, W, self.embed_dim).permute(0, 3, 1, 2).contiguous()  # B,C,H,W

        return output + k_proj + q_proj  # 最后的残差