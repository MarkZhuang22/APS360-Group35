import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from math import sqrt
from itertools import product as product

from utils import *
from base_model import *
from aux_conv import AuxiliaryConvolutions
from prediction_conv import PredictionConvolutions
from ft_aux_conv import FtAuxiliaryConvolutions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# heavily modified from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection
class SSD300(nn.Module):
    """
    The SSD300 network - encapsulates the base VGG network, auxiliary, and prediction convolutions.
    """
    def __init__(self, config):
        super(SSD300, self).__init__()

        self.n_classes = config.n_classes
        self.mode = config.mode
        self.fmap_dims = config.fmap_dims
        self.obj_scales = config.obj_scales
        self.aspect_ratios = config.aspect_ratios
        if config.base_model == 'vgg':
            self.base = VGGBase()
        elif config.base_model =='resnet':
            self.base = ResNetBase()
        self.aux_convs = AuxiliaryConvolutions()
        self.pred_convs = PredictionConvolutions(config=config)


        if (config.mode == FEATURETRANSFER):
            self.ft_bn = nn.BatchNorm2d(768, affine=True)
            self.ft_aux_convs = FtAuxiliaryConvolutions()

            self.ft_3 = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(512, 128, kernel_size=1, padding=0), #19
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  #
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=1, padding = 0),
                InterpolateModule(size=(38, 38), mode='bilinear')
                )
            
            self.ft_1 = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1, padding = 0),
                nn.ReLU()
                )
            self.ft_2 = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=1, padding = 0),
                nn.ReLU(),
                InterpolateModule(size=(38, 38), mode='bilinear')
                )
        else: 
            # Since lower level features (conv4_3_feats) have considerably larger scales, we take the L2 norm and rescale
            # Rescale factor is initially set at 20, but is learned for each channel during back-prop
            self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))  # there are 512 channels in conv4_3_feats
            nn.init.constant_(self.rescale_factors, 20)

        # Prior boxes
        self.priors_cxcy = self.create_prior_boxes()

    def forward(self, image):
        """
        Forward propagation.

        :param image: images, a tensor of dimensions (N, 3, 300, 300)
        :return: 8732 locations and class scores (i.e. w.r.t each prior box) for each image
        """
        # Run VGG base network convolutions (lower level feature map generators)
        #mode = 'baisc'
        conv4_3_feats, conv7_feats = self.base(image)  # (N, 512, 38, 38), (N, 1024, 19, 19)
        if self.mode  == FEATURETRANSFER:
            conv_ft1 = self.ft_1(conv4_3_feats)
            conv_ft2 = self.ft_2(conv7_feats)
            conv_ft_3 = self.ft_3(conv7_feats)
            FT = torch.cat([conv_ft1,conv_ft2,conv_ft_3], 1)
        
            FT_n = self.ft_bn(FT) # (N, 768, 38, 38)
            convft1_feats, convft2_feats, conv1ft3_feats, convft4_feats,convft5_feats,convft6_feats = \
                                                self.ft_aux_convs(FT_n)
            locs, classes_scores = self.pred_convs(convft1_feats, convft2_feats, conv1ft3_feats, convft4_feats,convft5_feats,convft6_feats)
        

        # Rescale conv4_3 after L2 norm
        ############may be needed for feature transfer
        elif self.mode  == BASIC :
            norm = conv4_3_feats.pow(2).sum(dim=1, keepdim=True).sqrt()  # (N, 1, 38, 38)
            conv4_3_feats = conv4_3_feats / norm  # (N, 512, 38, 38)
            conv4_3_feats = conv4_3_feats * self.rescale_factors  # (N, 512, 38, 38)
            conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats = self.aux_convs(conv7_feats)
              # (N, 512, 10, 10),  (N, 256, 5, 5), (N, 256, 3, 3), (N, 256, 1, 1)
            locs, classes_scores = self.pred_convs(conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats,
                                               conv11_2_feats)  # (N, 8732, 4), (N, 8732, n_classes)

        ############--------------------------
          # (N, 8732, 4), (N, 8732, n_classes)
        # (PyTorch autobroadcasts singleton dimensions during arithmetic)

        # Run auxiliary convolutions (higher level feature map generators)

        #conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats = \
        #	self.aux_convs(conv7_feats)  # (N, 512, 10, 10),  (N, 256, 5, 5), (N, 256, 3, 3), (N, 256, 1, 1)

        # Run prediction convolutions (predict offsets w.r.t prior-boxes and classes in each resulting localization box)
        #locs, classes_scores = self.pred_convs(conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats,
        #									   conv11_2_feats)  # (N, 8732, 4), (N, 8732, n_classes)

        return locs, classes_scores

    def create_prior_boxes(self):
        """
        Create the 8732 prior (default) boxes for the SSD300, as defined in the paper.

        :return: prior boxes in center-size coordinates, a tensor of dimensions (8732, 4)
        """
        fmap_dims = self.fmap_dims

        obj_scales = self.obj_scales
        aspect_ratios = self.aspect_ratios
        fmaps = list(fmap_dims.keys())
        prior_boxes = []

        for k, fmap in enumerate(fmaps):
            for i in range(fmap_dims[fmap]):
                for j in range(fmap_dims[fmap]):
                    cx = (j + 0.5) / fmap_dims[fmap]
                    cy = (i + 0.5) / fmap_dims[fmap]

                    for ratio in aspect_ratios[fmap]:
                        prior_boxes.append([cx, cy, obj_scales[fmap] * sqrt(ratio), obj_scales[fmap] / sqrt(ratio)])

                        # For an aspect ratio of 1, use an additional prior whose scale is the geometric mean of the
                        # scale of the current feature map and the scale of the next feature map
                        if ratio == 1.:
                            try:
                                additional_scale = sqrt(obj_scales[fmap] * obj_scales[fmaps[k + 1]])
                            # For the last feature map, there is no "next" feature map
                            except IndexError:
                                additional_scale = 1.
                            prior_boxes.append([cx, cy, additional_scale, additional_scale])

        prior_boxes = torch.FloatTensor(prior_boxes).to(device)  # (8732, 4)
        prior_boxes.clamp_(0, 1)  # (8732, 4)

        return prior_boxes

    def detect_objects(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k):
        """
        Decipher the 8732 locations and class scores (output of ths SSD300) to detect objects.

        For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.

        :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
        :param min_score: minimum threshold for a box to be considered a match for a certain class
        :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
        :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
        :return: detections (boxes, labels, and scores), lists of length batch_size
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 8732, n_classes)

        # Lists to store final predicted boxes, labels, and scores for all images
        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        for i in range(batch_size):
            # Decode object coordinates from the form we regressed predicted boxes to
            decoded_locs = cxcy_to_xy(
                gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy))  # (8732, 4), these are fractional pt. coordinates

            # Lists to store boxes and scores for this image
            image_boxes = list()
            image_labels = list()
            image_scores = list()

            max_scores, best_label = predicted_scores[i].max(dim=1)  # (8732)

            # Check for each class
            for c in range(1, self.n_classes):
                # Keep only predicted boxes and scores where scores for this class are above the minimum score
                class_scores = predicted_scores[i][:, c]  # (8732)
                score_above_min_score = class_scores > min_score  # torch.uint8 (byte) tensor, for indexing
                n_above_min_score = score_above_min_score.sum().item()
                if n_above_min_score == 0:
                    continue
                class_scores = class_scores[score_above_min_score]  # (n_qualified), n_min_score <= 8732
                class_decoded_locs = decoded_locs[score_above_min_score]  # (n_qualified, 4)

                # Sort predicted boxes and scores by scores
                class_scores, sort_ind = class_scores.sort(dim=0, descending=True)  # (n_qualified), (n_min_score)
                class_decoded_locs = class_decoded_locs[sort_ind]  # (n_min_score, 4)

                # Find the overlap between predicted boxes
                overlap = find_overlap(class_decoded_locs, class_decoded_locs)  # (n_qualified, n_min_score)

                # Non-Maximum Suppression (NMS)

                # A torch.uint8 (byte) tensor to keep track of which predicted boxes to suppress
                # 1 implies suppress, 0 implies don't suppress
                suppress = torch.zeros((n_above_min_score), dtype=torch.uint8).to(device)  # (n_qualified)

                # Consider each box in order of decreasing scores
                for box in range(class_decoded_locs.size(0)):
                    # If this box is already marked for suppression
                    if suppress[box] == 1:
                        continue

                    # Suppress boxes whose overlaps (with this box) are greater than maximum overlap
                    # Find such boxes and update suppress indices
                    ones = overlap[box] > max_overlap
                    ones = ones.type(torch.uint8)  
                    #set_trace()
                    suppress = torch.max(suppress, ones)
                    # The max operation retains previously suppressed boxes, like an 'OR' operation

                    # Don't suppress this box, even though it has an overlap of 1 with itself
                    suppress[box] = 0

                # Store only unsuppressed boxes for this class
                image_boxes.append(class_decoded_locs[1 - suppress])
                image_labels.append(torch.LongTensor((1 - suppress).sum().item() * [c]).to(device))
                image_scores.append(class_scores[1 - suppress])

            # If no object in any class is found, store a placeholder for 'background'
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
                image_labels.append(torch.LongTensor([0]).to(device))
                image_scores.append(torch.FloatTensor([0.]).to(device))

            # Concatenate into single tensors
            image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
            image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
            image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
            n_objects = image_scores.size(0)

            # Keep only the top k objects
            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]  # (top_k)
                image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
                image_labels = image_labels[sort_ind][:top_k]  # (top_k)

            # Append to lists that store predicted boxes and scores for all images
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores  # lists of length batch_size


# Heaviliy modified from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection
# Add several heuristic functions to help training, check highlight part to see how it works
class MultiBoxLoss(nn.Module):
    """
    The MultiBox loss, a loss function for object detection.

    This is a combination of:
    (1) a localization loss for the predicted locations of the boxes, and
    (2) a confidence loss for the predicted class scores.
    """

    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1.):
        super(MultiBoxLoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.smooth_l1 = nn.L1Loss() # want to use l2, but not sure current herusitc can handle that
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)

    def forward(self,config, predicted_locs, predicted_scores, boxes, labels):
        """
        Forward propagation.

        :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
        :param boxes: true  object bounding boxes in boundary coordinates, a list of N tensors
        :param labels: true object labels, a list of N tensors
        :return: multibox loss, a scalar
        """
        #mode = config.mode
        fmap_dims = config.fmap_dims
        aspect_ratios = config.aspect_ratios
        herustic =config.herustic

        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device)  # (N, 8732, 4)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(device)  # (N, 8732)

        # For each image
        for i in range(batch_size):
            n_objects = boxes[i].size(0)

            overlap = find_overlap(boxes[i],
                                           self.priors_xy)  # (n_objects, 8732)
            #print(overlap.shape)
            # For each prior, find the object that has the maximum overlap
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)  # (8732)
        
            # We don't want a situation where an object is not represented in our positive (non-background) priors -
            # 1. An object might not be the best object for all priors, and is therefore not in object_for_each_prior.
            # 2. All priors with the object may be assigned as background based on the threshold (0.5).

            # To remedy this -
            # First, find the prior that has the maximum overlap for each object.
            max_iou, _prior_for_each_object = overlap.max(dim=1)  # (N_o)
            prior_for_each_object =_prior_for_each_object
            #prior_for_each_object
            
            #print((max_indices[0]))
            
            #########################################################
            # Create a mask to filter priors with the maximum IoU
            if(herustic > 0):
                mask = torch.eq(overlap, max_iou.unsqueeze(1))
                # Get the indices of priors with the maximum IoU
                max_indices = []
                for k in range(mask.shape[0]):
                    indices = torch.nonzero(mask[k], as_tuple=False)
                    max_indices.append(indices)

                truth_list = boxes[i] 
                
                if len(_prior_for_each_object) == 1 and len(max_indices[0])==1:
                    # If there is only one index available and it matches the conditions, no need to perform further checks
                    prior_for_each_object =_prior_for_each_object    
                else:
                    prior_size = len(_prior_for_each_object)
                    for j in range(prior_size):
                        truth = truth_list[j]
                        truth_ratio = (truth[2] - truth[0]) / (truth[3] - truth[1])
                        ratio_list = [1, -1000, 2, 0.5, 3, 0.33]
                        gt_truth_ratios = torch.abs(torch.tensor(ratio_list) - truth_ratio).argmin()
                        candiddate_list =[]
                        if herustic != 2:
                            for condidate_idx in max_indices[j]:
                                prior_xy = self.priors_xy[condidate_idx][0]
                                prior_ratio = (prior_xy[2]-prior_xy[0]) / (prior_xy[3]-prior_xy[1])
                                gt_prior_ratios = torch.abs(torch.tensor(ratio_list) - prior_ratio).argmin()
                                if(gt_prior_ratios == gt_truth_ratios): # choose pior with same ratio as the ground truth bbox
                                    candiddate_list.append(condidate_idx)
                                else: # For feature maps that do not have 1/3 or 3, choose 2 instead of 3 and 1/2 instead of 1/3
                                    sum = 0
                                    fmaps = list(fmap_dims.keys())
                                    aspects = list(aspect_ratios.keys())
                                    for l,(fmap,aspects) in enumerate(zip(fmaps,aspect_ratios)):
                                        cur_size = fmap_dims[fmap]*fmap_dims[fmap]*(len(aspect_ratios[aspects])+1)
                                        if(condidate_idx<(sum+cur_size)):
                                            prior_loc = l
                                            break
                                        sum += cur_size
                                    if(len(aspect_ratios[aspects])) <gt_truth_ratios:
                                        if(gt_prior_ratios ==(gt_truth_ratios-2)):
                                            candiddate_list.append(condidate_idx)
                        else:
                            candiddate_list = max_indices[j]

                        # Calculate the Euclidean distance between each candidate bounding box and the ground truth bbox                  
                        min_distance = float('inf')
                        nearest_bbox = None
                        for bbox in candiddate_list:
                                distance = torch.sqrt(torch.sum((bbox - truth) ** 2))
                                if distance < min_distance:
                                    min_distance = distance
                                    nearest_bbox = bbox
            
                        if( nearest_bbox and prior_for_each_object[j] != nearest_bbox):
                            prior_for_each_object[j] = nearest_bbox

            #########################################################

        
            # Then, assign each object to the corresponding maximum-overlap-prior. (This fixes 1.)
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(device)

            # To ensure these priors qualify, artificially give them an overlap of greater than 0.5. (This fixes 2.)
            overlap_for_each_prior[prior_for_each_object] = 1.

            # Labels for each prior
            label_for_each_prior = labels[i][object_for_each_prior]  # (8732)
            # Set priors whose overlaps with objects are less than the threshold to be background (no object)
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0  # (8732)

            # Store
            true_classes[i] = label_for_each_prior

            # Encode center-size object coordinates into the form we regressed predicted boxes to
            true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)  # (8732, 4)

        # Identify priors that are positive (object/non-background)
        positive_priors = true_classes != 0  # (N, 8732)

        # LOCALIZATION LOSS

        # Localization loss is computed only over positive (non-background) priors
        loc_loss = self.smooth_l1(predicted_locs[positive_priors], true_locs[positive_priors])  # (), scalar

        # Note: indexing with a torch.uint8 (byte) tensor flattens the tensor when indexing is across multiple dimensions (N & 8732)
        # So, if predicted_locs has the shape (N, 8732, 4), predicted_locs[positive_priors] will have (total positives, 4)

        # CONFIDENCE LOSS

        # Confidence loss is computed over positive priors and the most difficult (hardest) negative priors in each image
        # That is, FOR EACH IMAGE,
        # we will take the hardest (neg_pos_ratio * n_positives) negative priors, i.e where there is maximum loss
        # This is called Hard Negative Mining - it concentrates on hardest negatives in each image, and also minimizes pos/neg imbalance

        # Number of positive and hard-negative priors per image
        n_positives = positive_priors.sum(dim=1)  # (N)
        n_hard_negatives = self.neg_pos_ratio * n_positives  # (N)

        # First, find the loss for all priors
        conf_loss_all = self.cross_entropy(predicted_scores.view(-1, n_classes), true_classes.view(-1))  # (N * 8732)
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)  # (N, 8732)

        # We already know which priors are positive
        conf_loss_pos = conf_loss_all[positive_priors]  # (sum(n_positives))

        # Next, find which priors are hard-negative
        # To do this, sort ONLY negative priors in each image in order of decreasing loss and take top n_hard_negatives
        conf_loss_neg = conf_loss_all.clone()  # (N, 8732)
        conf_loss_neg[positive_priors] = 0.  # (N, 8732), positive priors are ignored (never in top n_hard_negatives)
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)  # (N, 8732), sorted by decreasing hardness
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(device)  # (N, 8732)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, 8732)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]  # (sum(n_hard_negatives))

        # As in the paper, averaged over positive priors only, although computed over both positive and hard-negative priors
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()  # (), scalar

        # TOTAL LOSS

        return conf_loss + self.alpha * loc_loss