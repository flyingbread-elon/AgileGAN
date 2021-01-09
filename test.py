import torch
from torchvision import utils
from lib.encoder.Encoder import EncoderModel
from argparse import Namespace
import cv2
import torchvision.transforms as transforms
from lib.normal_image import  Normal_Image
from lib.generator.model_dual import DualGenerator
from lib.genderdetect import GenderDetection
import argparse

model_paths = {
	'cartoon': 'pretrain/cartoon.pt',
    'wuxia': 'pretrain/wuxia.pt',
    'oil': 'pretrain/oil.pt',
    'scarlett': 'pretrain/scarlett.pt',
    'jackie': 'pretrain/jackie.pt'
}


class Tester():
    def __init__(self, low_path='pretrain/pixar.pt'):
        self.low_path =low_path
        g_ema = DualGenerator(1024, 512, 8).to('cuda')
        checkpoint = torch.load(self.low_path)
        g_ema.load_state_dict(checkpoint["g_ema"], strict=False)
        g_ema.eval()
        self.g_ema = g_ema

        ckpt = torch.load('pretrain/encoder.pt', map_location='cpu')
        opts = ckpt['opts']
        opts['checkpoint_path'] = 'pretrain/encoder.pt'
        if 'learn_in_w' not in opts:
            opts['learn_in_w'] = False
        opts = Namespace(**opts)
        pspnet = EncoderModel(opts)
        pspnet.eval()
        pspnet.to('cuda')
        self.pspnet = pspnet

        self.normal = Normal_Image()

        self.transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        self.gender_detect = GenderDetection()

        print('init done')

    def run(self, img_path):
        img = cv2.imread(img_path)
        if (img is not None):
            is_female = self.gender_detect.detect(img)
            img = self.normal.run(img)
        else:
            print('img is None')
            return 0

        if (img is not None):
            img = img.convert("RGB")
            transformed_image = self.transforms(img)
            _, _, mu = self.pspnet.encoder(transformed_image.unsqueeze(0).to("cuda").float())
            latent = [self.pspnet.decoder.style(s) for s in mu]
            latent = [torch.stack(latent, dim=0)]
            fake_img_A, fake_img_B, _ = self.g_ema(latent, input_is_latent=True, noise=None)
            if (is_female):
                utils.save_image(
                    fake_img_A,
                    "output.jpg",
                    nrow=1,
                    normalize=True,
                    range=(-1, 1),
                )
            else:
                utils.save_image(
                    fake_img_B,
                    "output.jpg",
                    nrow=1,
                    normalize=True,
                    range=(-1, 1),
                )
        else:
            print('img is None')
            return 0

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='examples/29899.jpg')
parser.add_argument('--style', type=str, default='cartoon')
input_opts = parser.parse_args()

T=Tester(model_paths[input_opts.style])
print("start to infer.")
T.run(input_opts.path)
print("done.")