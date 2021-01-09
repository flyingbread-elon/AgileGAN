import math
import random
import torch
from torch import nn
from lib.encoder.stylegan2.model import PixelNorm,EqualLinear,ConstantInput,StyledConv,ToRGB

class DualGenerator(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
    ):
        super().__init__()

        self.size = size

        self.style_dim = style_dim

        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )

        self.style = nn.Sequential(*layers)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(
            self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel
        )
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convsA = nn.ModuleList()
        self.to_rgbsA = nn.ModuleList()

        self.convsB = nn.ModuleList()
        self.to_rgbsB = nn.ModuleList()

        self.convsC = nn.ModuleList()
        self.to_rgbsC = nn.ModuleList()


        self.noises = nn.Module()

        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f'noise_{layer_idx}', torch.randn(*shape))



        for i in range(3, self.log_size + 1):
            if(i<7):
                #A
                out_channel = self.channels[2 ** i]
                self.convsA.append(
                    StyledConv(
                        in_channel,
                        out_channel,
                        3,
                        style_dim,
                        upsample=True,
                        blur_kernel=blur_kernel,
                    )
                )
                self.convsA.append(
                    StyledConv(
                        out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel
                    )
                )
                self.to_rgbsA.append(ToRGB(out_channel, style_dim))
                #B
                self.convsB.append(
                    StyledConv(
                        in_channel,
                        out_channel,
                        3,
                        style_dim,
                        upsample=True,
                        blur_kernel=blur_kernel,
                    )
                )
                self.convsB.append(
                    StyledConv(
                        out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel
                    )
                )
                self.to_rgbsB.append(ToRGB(out_channel, style_dim))
            else:
                out_channel = self.channels[2 ** i]
                self.convsC.append(
                    StyledConv(
                        in_channel,
                        out_channel,
                        3,
                        style_dim,
                        upsample=True,
                        blur_kernel=blur_kernel,
                    )
                )
                self.convsC.append(
                    StyledConv(
                        out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel
                    )
                )
                self.to_rgbsC.append(ToRGB(out_channel, style_dim))
            in_channel = out_channel
        self.n_latent = self.log_size * 2 - 2

    def make_noise(self):
        device = self.input.input.device

        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

        return noises

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.input.input.device
        )
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, input):
        return self.style(input)


    def forward(
        self,
        styles,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,
    ):
        if not input_is_latent:
            styles = [self.style(s) for s in styles]

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f'noise_{i}') for i in range(self.num_layers)
                ]

        if truncation < 1:
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = style_t

        if len(styles) < 2:
            inject_index = self.n_latent

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

            else:
                latent = styles[0]

        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)

        out = self.input(latent)


        out = self.conv1(out, latent[:, 0], noise=noise[0])
        skip = self.to_rgb1(out, latent[:, 1])


        #
        outA, outB = out.clone(),out.clone()
        skipA, skipB = skip.clone(),skip.clone()
        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convsA[::2], self.convsA[1::2], noise[1::2], noise[2::2], self.to_rgbsA
        ):
            outA = conv1(outA, latent[:, i], noise=noise1)
            outA = conv2(outA, latent[:, i + 1], noise=noise2)
            skipA = to_rgb(outA, latent[:, i + 2], skipA)
            i += 2

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convsB[::2], self.convsB[1::2], noise[1::2], noise[2::2], self.to_rgbsB
        ):
            outB = conv1(outB, latent[:, i], noise=noise1)
            outB = conv2(outB, latent[:, i + 1], noise=noise2)
            skipB = to_rgb(outB, latent[:, i + 2], skipB)
            i += 2


        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convsC[::2], self.convsC[1::2], noise[1::2], noise[2::2], self.to_rgbsC
        ):
            outA = conv1(outA, latent[:, i], noise=noise1)
            outA = conv2(outA, latent[:, i + 1], noise=noise2)
            skipA = to_rgb(outA, latent[:, i + 2], skipA)

            outB = conv1(outB, latent[:, i], noise=noise1)
            outB = conv2(outB, latent[:, i + 1], noise=noise2)
            skipB = to_rgb(outB, latent[:, i + 2], skipB)
            i += 2


        imageA = skipA
        imageB = skipB

        if return_latents:
            return imageA,imageB, latent

        else:
            return imageA, imageB, None

