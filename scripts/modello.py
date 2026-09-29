import torch
import torch.nn as nn
import torch.nn.functional as F


class BloccoConv(nn.Module):
    def __init__(self, canali_in, canali_out):
        super().__init__()
        self.conv1 = nn.Conv2d(canali_in, canali_out, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(canali_out, canali_out, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class LivelloEncoder(nn.Module):
    def __init__(self, canali_in, canali_out):
        super().__init__()
        self.blocco = BloccoConv(canali_in, canali_out)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        skip = self.blocco(x)
        x_ridotto = self.pool(skip)
        return x_ridotto, skip


class Encoder(nn.Module):
    def __init__(self, canali_in=1, canali_base=16):
        super().__init__()
        self.livello1 = LivelloEncoder(canali_in, canali_base)
        self.livello2 = LivelloEncoder(canali_base, canali_base * 2)
        self.livello3 = LivelloEncoder(canali_base * 2, canali_base * 4)

    def forward(self, x):
        x, skip1 = self.livello1(x)
        x, skip2 = self.livello2(x)
        x, skip3 = self.livello3(x)
        return x, [skip1, skip2, skip3]


class Bottleneck(nn.Module):
    def __init__(self, canali_in, canali_out):
        super().__init__()
        self.blocco = BloccoConv(canali_in, canali_out)

    def forward(self, x):
        return self.blocco(x)


class LivelloDecoder(nn.Module):
    def __init__(self, canali_in, canali_out):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.riduci_canali = nn.Conv2d(canali_in, canali_out, kernel_size=1)
        self.blocco = BloccoConv(canali_out * 2, canali_out)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = self.riduci_canali(x)
        x = torch.cat([x, skip], dim=1)
        x = self.blocco(x)
        return x


class Decoder(nn.Module):
    def __init__(self, canali_base=16):
        super().__init__()
        self.livello1 = LivelloDecoder(canali_base * 8, canali_base * 4)
        self.livello2 = LivelloDecoder(canali_base * 4, canali_base * 2)
        self.livello3 = LivelloDecoder(canali_base * 2, canali_base)

    def forward(self, x, skips):
        skip1, skip2, skip3 = skips
        x = self.livello1(x, skip3)
        x = self.livello2(x, skip2)
        x = self.livello3(x, skip1)
        return x


class LivelloOutput(nn.Module):
    def __init__(self, canali_in):
        super().__init__()
        self.conv_finale = nn.Conv2d(canali_in, 1, kernel_size=1)
        self.attivazione = nn.Softplus()

    def forward(self, x):
        x = self.conv_finale(x)
        x = self.attivazione(x)
        return x


class UNetImmunogold(nn.Module):
    def __init__(self, canali_in=1, canali_base=16):
        super().__init__()
        self.encoder = Encoder(canali_in=canali_in, canali_base=canali_base)
        self.bottleneck = Bottleneck(canali_in=canali_base * 4, canali_out=canali_base * 8)
        self.decoder = Decoder(canali_base=canali_base)
        self.output = LivelloOutput(canali_in=canali_base)

    def forward(self, x):
        x, skips = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, skips)
        x = self.output(x)
        return x
