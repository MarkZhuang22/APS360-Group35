
import torch
from torch import nn
import torch.nn.functional as F
from math import sqrt
from itertools import product as product
import torchvision


class VGGBase(nn.Module):
	"""
	VGG base convolutions to produce lower-level feature maps.
	"""

	def __init__(self):
		super(VGGBase, self).__init__()

		# Standard convolutional layers in VGG16
		self.conv1 = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2)
			)
	   

		self.conv2 = nn.Sequential(
			nn.Conv2d(64, 128, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(128, 128, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2)
			)
		

		self.conv3 = nn.Sequential(
			nn.Conv2d(128, 256, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(256, 256, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(256, 256, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
			)
		

		self.conv4 = nn.Sequential(
			nn.Conv2d(256, 512, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.ReLU(),
			# nn.MaxPool2d(kernel_size=2, stride=2)
			)

		
		self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

		self.conv5 = nn.Sequential(
			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
			)
		
		# Replacements for FC6 and FC7 in VGG16
		self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)  # atrous convolution

		self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

		# Load pretrained layers
		self.load_pretrained_layers()

	def forward(self, image):
		"""
		Forward propagation.

		:param image: images, a tensor of dimensions (N, 3, 300, 300)
		:return: lower-level feature maps conv4_3 and conv7
		"""
		output = self.conv1(image)
		

		output = self.conv2(output)
		

		output = self.conv3(output)
		

		output = self.conv4(output)
		
		conv4_3_feats = output  # (N, 512, 38, 38)
		output = self.pool4(output)  # (N, 512, 19, 19)

		output = self.conv5(output)
	 

		output = F.relu(self.conv6(output))  # (N, 1024, 19, 19)

		conv7_feats = F.relu(self.conv7(output))  # (N, 1024, 19, 19)

		# Lower-level feature maps
		return conv4_3_feats, conv7_feats

	def load_pretrained_layers(self):
		"""
		As in the paper, we use a VGG-16 pretrained on the ImageNet task as the base network.
		There's one available in PyTorch, see https://pytorch.org/docs/stable/torchvision/models.html#torchvision.models.vgg16
		We copy these parameters into our network. It's straightforward for conv1 to conv5.
		However, the original VGG-16 does not contain the conv6 and con7 layers.
		Therefore, we convert fc6 and fc7 into convolutional layers, and subsample by decimation. See 'decimate' in utils.py.
		"""
		# Current state of base
		state_dict = self.state_dict()
		param_names = list(state_dict.keys())

		# Pretrained VGG base
		pretrained_state_dict = torchvision.models.vgg16(pretrained=True).state_dict()
		pretrained_param_names = list(pretrained_state_dict.keys())

		# Transfer conv. parameters from pretrained model to current model
		for i, param in enumerate(param_names[:-4]):  # excluding conv6 and conv7 parameters
			state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]

		# Convert fc6, fc7 to convolutional layers, and subsample (by decimation) to sizes of conv6 and conv7
		# fc6
		conv_fc6_weight = pretrained_state_dict['classifier.0.weight'].view(4096, 512, 7, 7)  # (4096, 512, 7, 7)
		conv_fc6_bias = pretrained_state_dict['classifier.0.bias']  # (4096)
		state_dict['conv6.weight'] = decimate(conv_fc6_weight, m=[4, None, 3, 3])  # (1024, 512, 3, 3)
		state_dict['conv6.bias'] = decimate(conv_fc6_bias, m=[4])  # (1024)
		# fc7
		conv_fc7_weight = pretrained_state_dict['classifier.3.weight'].view(4096, 4096, 1, 1)  # (4096, 4096, 1, 1)
		conv_fc7_bias = pretrained_state_dict['classifier.3.bias']  # (4096)
		state_dict['conv7.weight'] = decimate(conv_fc7_weight, m=[4, 4, None, None])  # (1024, 1024, 1, 1)
		state_dict['conv7.bias'] = decimate(conv_fc7_bias, m=[4])  # (1024)

		# Note: an FC layer of size (K) operating on a flattened version (C*H*W) of a 2D image of size (C, H, W)...
		# ...is equivalent to a convolutional layer with kernel size (H, W), input channels C, output channels K...
		# ...operating on the 2D image of size (C, H, W) without padding

		self.load_state_dict(state_dict)

		print("\nLoaded base model.\n")