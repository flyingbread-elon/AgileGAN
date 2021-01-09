
import torch
from torch import nn
from lib.encoder import model
from lib.encoder.stylegan2.model import Generator



def get_keys(d, name):
	if 'state_dict' in d:
		d = d['state_dict']
	d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
	return d_filt


class EncoderModel(nn.Module):

	def __init__(self, opts):
		super(EncoderModel, self).__init__()
		self.set_opts(opts)
		# Define architecture
		self.encoder = model.VAEStyleEncoder(50, self.opts)
		self.decoder = Generator(1024, 512, 8)
		self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
		# Load weights if needed
		self.load_weights()

	def load_weights(self):
		if self.opts.checkpoint_path is not None:
			print('Loading model from checkpoint: {}'.format(self.opts.checkpoint_path))
			ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
			self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
			self.__load_latent_avg(ckpt)
			self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
		else:
			print('Loading fail')


	def set_opts(self, opts):
		self.opts = opts

	def __load_latent_avg(self, ckpt, repeat=None):
		if 'latent_avg' in ckpt:
			self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
			if repeat is not None:
				self.latent_avg = self.latent_avg.repeat(repeat, 1)
		else:
			self.latent_avg = None
