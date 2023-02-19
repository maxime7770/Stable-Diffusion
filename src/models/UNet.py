from torch import nn
import torch
import numpy as np




class Block(nn.Module):

    def __init__(self, c_in, c_out, c_mid, residual=False):
        super().__init__()
        self.residual  = residual
        if not c_mid:
            c_mid = c_out
        self.double_conv = nn.Sequential([
            nn.Conv2d(c_in, c_mid, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, c_mid),
            nn.ReLU(),
            nn.Conv2d(c_mid, c_out, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, c_out),
        ])


    def forward(self, x):
        if self.residual:
            x = nn.ReLu()(x + self.double_conv(x))
            return x
        else:
            return self.double_conv(x)



class DownBlock(nn.Module):

    def __init__(self, c_in, c_out, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential([
            nn.MaxPool2d(2),
            Block(c_in, c_out, residual=True),
            Block(c_in, c_out),
        ])

        self.emb_layer = nn.Sequential([
            nn.SiLU(),
            nn.Linear(emb_dim, c_out)
        ])

    def forward(self, x, emb):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(emb)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb



class UpBlock(nn.Module):

    def __init__(self, c_in, c_out, emb_dim=256):
        super().__init__()
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential([
            Block(c_in, c_out, residual=True),
            Block(c_in, c_out, c_in//2)
        ])

        self.emb_layer = nn.Sequential([
            nn.SiLU(),
            nn.Linear(emb_dim, c_out)
        ])

    def forward(self, x, skip_x, emb):
        x = self.up_sample(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(emb)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb



class SelfAttention(nn.Module):

    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels        
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        size = x.shape[-1]
        x = x.view(-1, self.channels, size * size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, size, size)



class UNet(nn.Module):

    def __init__(self, c_in=3, c_out=3, emb_dim=256):
        super().__init__()
        self.emb_dim = emb_dim
        self.input = Block(c_in, 64)
        self.down1 = DownBlock(64, 128)
        self.self_attn1 = SelfAttention(128)
        self.down2 = DownBlock(128, 256)
        self.self_attn2 = SelfAttention(256)
        self.down3 = DownBlock(256, 256)
        self.self_attn3 = SelfAttention(256)
        self.bot1 = Block(256, 512)
        self.bot2 = Block(512, 512)
        self.up1 = UpBlock(512, 256)
        self.self_attn4 = SelfAttention(256)
        self.up2 = UpBlock(256, 128)
        self.self_attn5 = SelfAttention(128)
        self.up3 = UpBlock(128, 64)
        self.self_attn6 = SelfAttention(64)
        self.output = nn.Conv2d(64, c_out, kernel_size=1)

    def pos_encoding(self, emb, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2).float() / channels)
        )
        pos_enc_a = torch.sin(emb.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(emb.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, emb):

        emb = emb.unsqueeze(-1)
        emb = self.pos_encoding(emb, self.emb_dim)

        x1 = self.input(x)
        x2 = self.down1(x1, emb)
        x2 = self.self_attn1(x2)
        x3 = self.down2(x2, emb)
        x3 = self.self_attn2(x3)
        x4 = self.down3(x3, emb)
        x4 = self.self_attn3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)

        x = self.up1(x4, x3, emb)
        x = self.self_attn4(x)
        x = self.up2(x, x2, emb)
        x = self.self_attn5(x)
        x = self.up3(x, x1, emb)
        x = self.self_attn6(x)
        x = self.output(x)

        return x

