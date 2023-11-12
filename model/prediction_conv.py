import torch
from torch import nn


class PredictionConvolutions(nn.Module):
	"""
	Convolutions to predict class scores and bounding boxes using lower and higher-level feature maps.

	The bounding boxes (locations) are predicted as encoded offsets w.r.t each of the 8732 prior (default) boxes.
	See 'cxcy_to_gcxgcy' in utils.py for the encoding definition.

	The class scores represent the scores of each object class in each of the 8732 bounding boxes located.
	A high score for 'background' = no object.
	"""

	def __init__(self, n_classes):
		"""
		:param n_classes: number of different types of objects
		"""
		super(PredictionConvolutions, self).__init__()

		self.n_classes = n_classes

		# Number of prior-boxes we are considering per position in each feature map
		n_boxes = {'conv4_3': 4,
				   'conv7': 6,
				   'conv8_2': 6,
				   'conv9_2': 6,
				   'conv10_2': 4,
				   'conv11_2': 4}
		# 4 prior-boxes implies we use 4 different aspect ratios, etc.

		# Localization prediction convolutions (predict offsets w.r.t prior-boxes)
		self.loc_conv4_3 = nn.Conv2d(512, n_boxes['conv4_3'] * 4, kernel_size=3, padding=1)
		self.loc_conv7 = nn.Conv2d(1024, n_boxes['conv7'] * 4, kernel_size=3, padding=1)
		self.loc_conv8_2 = nn.Conv2d(512, n_boxes['conv8_2'] * 4, kernel_size=3, padding=1)
		self.loc_conv9_2 = nn.Conv2d(256, n_boxes['conv9_2'] * 4, kernel_size=3, padding=1)
		self.loc_conv10_2 = nn.Conv2d(256, n_boxes['conv10_2'] * 4, kernel_size=3, padding=1)
		self.loc_conv11_2 = nn.Conv2d(256, n_boxes['conv11_2'] * 4, kernel_size=3, padding=1)

		# Class prediction convolutions (predict classes in localization boxes)
		self.cl_conv4_3 = nn.Conv2d(512, n_boxes['conv4_3'] * n_classes, kernel_size=3, padding=1)
		self.cl_conv7 = nn.Conv2d(1024, n_boxes['conv7'] * n_classes, kernel_size=3, padding=1)
		self.cl_conv8_2 = nn.Conv2d(512, n_boxes['conv8_2'] * n_classes, kernel_size=3, padding=1)
		self.cl_conv9_2 = nn.Conv2d(256, n_boxes['conv9_2'] * n_classes, kernel_size=3, padding=1)
		self.cl_conv10_2 = nn.Conv2d(256, n_boxes['conv10_2'] * n_classes, kernel_size=3, padding=1)
		self.cl_conv11_2 = nn.Conv2d(256, n_boxes['conv11_2'] * n_classes, kernel_size=3, padding=1)

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

	def forward(self, conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats):
		"""
		Forward propagation.

		:param conv4_3_feats: conv4_3 feature map, a tensor of dimensions (N, 512, 38, 38)
		:param conv7_feats: conv7 feature map, a tensor of dimensions (N, 1024, 19, 19)
		:param conv8_2_feats: conv8_2 feature map, a tensor of dimensions (N, 512, 10, 10)
		:param conv9_2_feats: conv9_2 feature map, a tensor of dimensions (N, 256, 5, 5)
		:param conv10_2_feats: conv10_2 feature map, a tensor of dimensions (N, 256, 3, 3)
		:param conv11_2_feats: conv11_2 feature map, a tensor of dimensions (N, 256, 1, 1)
		:return: 8732 locations and class scores (i.e. w.r.t each prior box) for each image
		"""
		batch_size = conv4_3_feats.size(0)

		# Predict localization boxes' bounds (as offsets w.r.t prior-boxes)
		l_conv4_3 = self.loc_conv4_3(conv4_3_feats)  # (N, 16, 38, 38)
		l_conv4_3 = l_conv4_3.permute(0, 2, 3,
									  1).contiguous()  # (N, 38, 38, 16), to match prior-box order (after .view())
		# (.contiguous() ensures it is stored in a contiguous chunk of memory, needed for .view() below)
		l_conv4_3 = l_conv4_3.view(batch_size, -1, 4)  # (N, 5776, 4), there are a total 5776 boxes on this feature map

		l_conv7 = self.loc_conv7(conv7_feats)  # (N, 24, 19, 19)
		l_conv7 = l_conv7.permute(0, 2, 3, 1).contiguous()  # (N, 19, 19, 24)
		l_conv7 = l_conv7.view(batch_size, -1, 4)  # (N, 2166, 4), there are a total 2116 boxes on this feature map

		l_conv8_2 = self.loc_conv8_2(conv8_2_feats)  # (N, 24, 10, 10)
		l_conv8_2 = l_conv8_2.permute(0, 2, 3, 1).contiguous()  # (N, 10, 10, 24)
		l_conv8_2 = l_conv8_2.view(batch_size, -1, 4)  # (N, 600, 4)

		l_conv9_2 = self.loc_conv9_2(conv9_2_feats)  # (N, 24, 5, 5)
		l_conv9_2 = l_conv9_2.permute(0, 2, 3, 1).contiguous()  # (N, 5, 5, 24)
		l_conv9_2 = l_conv9_2.view(batch_size, -1, 4)  # (N, 150, 4)

		l_conv10_2 = self.loc_conv10_2(conv10_2_feats)  # (N, 16, 3, 3)
		l_conv10_2 = l_conv10_2.permute(0, 2, 3, 1).contiguous()  # (N, 3, 3, 16)
		l_conv10_2 = l_conv10_2.view(batch_size, -1, 4)  # (N, 36, 4)

		l_conv11_2 = self.loc_conv11_2(conv11_2_feats)  # (N, 16, 1, 1)
		l_conv11_2 = l_conv11_2.permute(0, 2, 3, 1).contiguous()  # (N, 1, 1, 16)
		l_conv11_2 = l_conv11_2.view(batch_size, -1, 4)  # (N, 4, 4)

		# Predict classes in localization boxes
		c_conv4_3 = self.cl_conv4_3(conv4_3_feats)  # (N, 4 * n_classes, 38, 38)
		c_conv4_3 = c_conv4_3.permute(0, 2, 3,
									  1).contiguous()  # (N, 38, 38, 4 * n_classes), to match prior-box order (after .view())
		c_conv4_3 = c_conv4_3.view(batch_size, -1,
								   self.n_classes)  # (N, 5776, n_classes), there are a total 5776 boxes on this feature map

		c_conv7 = self.cl_conv7(conv7_feats)  # (N, 6 * n_classes, 19, 19)
		c_conv7 = c_conv7.permute(0, 2, 3, 1).contiguous()  # (N, 19, 19, 6 * n_classes)
		c_conv7 = c_conv7.view(batch_size, -1,
							   self.n_classes)  # (N, 2166, n_classes), there are a total 2116 boxes on this feature map

		c_conv8_2 = self.cl_conv8_2(conv8_2_feats)  # (N, 6 * n_classes, 10, 10)
		c_conv8_2 = c_conv8_2.permute(0, 2, 3, 1).contiguous()  # (N, 10, 10, 6 * n_classes)
		c_conv8_2 = c_conv8_2.view(batch_size, -1, self.n_classes)  # (N, 600, n_classes)

		c_conv9_2 = self.cl_conv9_2(conv9_2_feats)  # (N, 6 * n_classes, 5, 5)
		c_conv9_2 = c_conv9_2.permute(0, 2, 3, 1).contiguous()  # (N, 5, 5, 6 * n_classes)
		c_conv9_2 = c_conv9_2.view(batch_size, -1, self.n_classes)  # (N, 150, n_classes)

		c_conv10_2 = self.cl_conv10_2(conv10_2_feats)  # (N, 4 * n_classes, 3, 3)
		c_conv10_2 = c_conv10_2.permute(0, 2, 3, 1).contiguous()  # (N, 3, 3, 4 * n_classes)
		c_conv10_2 = c_conv10_2.view(batch_size, -1, self.n_classes)  # (N, 36, n_classes)

		c_conv11_2 = self.cl_conv11_2(conv11_2_feats)  # (N, 4 * n_classes, 1, 1)
		c_conv11_2 = c_conv11_2.permute(0, 2, 3, 1).contiguous()  # (N, 1, 1, 4 * n_classes)
		c_conv11_2 = c_conv11_2.view(batch_size, -1, self.n_classes)  # (N, 4, n_classes)

		# A total of 8732 boxes
		# Concatenate in this specific order (i.e. must match the order of the prior-boxes)
		locs = torch.cat([l_conv4_3, l_conv7, l_conv8_2, l_conv9_2, l_conv10_2, l_conv11_2], dim=1)  # (N, 8732, 4)
		classes_scores = torch.cat([c_conv4_3, c_conv7, c_conv8_2, c_conv9_2, c_conv10_2, c_conv11_2],
								   dim=1)  # (N, 8732, n_classes)

		return locs, classes_scores