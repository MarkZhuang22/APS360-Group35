import torch
from torch import nn

# Slightly modified from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection
class AuxiliaryConvolutions(nn.Module):
	"""
	Additional convolutions to produce higher-level feature maps.
	"""

	def __init__(self):
		super(AuxiliaryConvolutions, self).__init__()

		# Auxiliary/additional convolutions on top of the VGG base
		self.conv8 = nn.Sequential(
			nn.Conv2d(1024, 256, kernel_size=1, padding=0),
			nn.ReLU(),
			nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
			nn.ReLU()
			)
		
		self.conv9 = nn.Sequential(
			nn.Conv2d(512, 128, kernel_size=1, padding=0),
			nn.ReLU(),
			nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
			nn.ReLU()
			)
		
		self.conv10 = nn.Sequential(
			nn.Conv2d(256, 128, kernel_size=1, padding=0),
			nn.ReLU(),
			nn.Conv2d(128, 256, kernel_size=3, padding=0),
			nn.ReLU()
			)
		
		self.conv11 = nn.Sequential(
			nn.Conv2d(256, 128, kernel_size=1, padding=0),
			nn.ReLU(),
			nn.Conv2d(128, 256, kernel_size=3, padding=0),
			nn.ReLU()
			)
		
		# Initialize convolutions' parameters
		self.init_conv2d()

	def init_conv2d(self):
		"""
		Initialize convolution parameters.
		"""
		for c in self.children():
			if isinstance(c, nn.Conv2d):
				nn.init.xavier_uniform_(c.weight)
				nn.init.constant_(c.bias, 0.)

	def forward(self, conv7_feats):
		"""
		Forward propagation.

		:param conv7_feats: lower-level conv7 feature map, a tensor of dimensions (N, 1024, 19, 19)
		:return: higher-level feature maps conv8_2, conv9_2, conv10_2, and conv11_2
		"""
		output = self.conv8(conv7_feats)
		
		conv8_2_feats = output  # (N, 512, 10, 10)

		output = self.conv9(output)
	   
		conv9_2_feats = output  # (N, 256, 5, 5)

		output = self.conv10(output)
		
		conv10_2_feats = output  # (N, 256, 3, 3)

		conv11_2_feats = self.conv11(output)
		

		# Higher-level feature maps
		return conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats