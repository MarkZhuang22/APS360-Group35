from networkx import selfloop_edges
import torch
from torch import nn
import torch.nn.functional as F


class FtAuxiliaryConvolutions(nn.Module):
	"""
	Additional convolutions to produce higher-level feature maps.
	"""

	def __init__(self):
		super(FtAuxiliaryConvolutions, self).__init__()

		self.convf1 = nn.Sequential(
			nn.Conv2d(768, 256, kernel_size=1, padding=0),
			nn.ReLU(),
			nn.Conv2d(256, 512, kernel_size=3, padding=1),
			nn.ReLU()
			)
		
		self.convf2 = nn.Sequential(
			nn.Conv2d(512, 256, kernel_size=1, padding=0),
			nn.ReLU(),
			nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), #19*19*512
			nn.ReLU()
			)
		
		self.convf3 = nn.Sequential(
			nn.Conv2d(512, 128, kernel_size=1, padding=0),
			nn.ReLU(),
			nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), #10*10*256
			nn.ReLU() 
			)
		
		self.convf4 = nn.Sequential(
			nn.Conv2d(256, 128, kernel_size=1, padding=0),
			nn.ReLU(),
			nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),#5*5*256
			nn.ReLU()
			)
		
		self.convf5 = nn.Sequential(
			nn.Conv2d(256, 128, kernel_size=1, padding=0),
			nn.ReLU(),
			nn.Conv2d(128, 256, kernel_size=3, padding=0),
			nn.ReLU()
			)
		
		self.convf6 = nn.Sequential(
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

	def forward(self, ft):
		"""
		Forward propagation.

		:param conv7_feats: lower-level conv7 feature map, a tensor of dimensions (N, 1024, 19, 19)
		:return: higher-level feature maps conv8_2, conv9_2, conv10_2, and conv11_2
		"""
		#print(ft.shape)
		output = self.convf1(ft)
		
		convft1_feats = output  # (N, 512, 10, 10)
		
		#conv9_Reduced = self.conv9_Reduced(output)
        
		output = self.convf2(output)
	   
		convft2_feats = output  # (N, 256, 5, 5)
        
		output = self.convf3(output)
		
		conv1ft3_feats = output  # (N, 256, 3, 3)

		output = self.convf4(output)
		
		convft4_feats = output  # (N, 256, 3, 3)

		output = self.convf5(output)
		
		convft5_feats = output
		
		convft6_feats = self.convf6(output)
		#print(convft1_feats.shape,convft2_feats.shape,conv1ft3_feats.shape,convft4_feats.shape,convft5_feats.shape,convft6_feats.shape)

		# Higher-level feature maps
        
		return convft1_feats, convft2_feats, conv1ft3_feats, convft4_feats,convft5_feats,convft6_feats