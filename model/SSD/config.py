from utils import *
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class Config():

    def __init__(self):
        
        self.n_classes = len(label_map)  # number of different types of objects
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Learning parameters
        path = "./checkpoint_ssd300.pth.tar"
        if os.path.exists(path):
          self.checkpoint = path  # path to model checkpoint, None if none
        else:
          self.checkpoint = None
        self.batch_size = 16  # batch size
        self.workers = 0  # number of workers for loading data in the DataLoader
        self.print_freq = 50  # print training status every __ batches
        self.lr = 1e-3  # learning rate
        self.decay_lr_at = [80000, 100000]  # decay learning rate after these many iterations
        self.decay_lr_to = 0.1  # decay learning rate to this fraction of the existing learning rate
        self.momentum = 0.9  # momentum
        self.weight_decay = 5e-4  # weight decay
        self.epochs = 40
        self.heuristic = 0
        self.mode = BASIC # BASIC or FEATURETRANSFER or FEATURETRANSFER_2
        #self.mode = FEATURETRANSFER_2
        self.net_size = 'small'
        self.base_model = 'vgg'
        self.chanel_attention = 'SE' #SE or CBAM or NONE
          
        self.fmap_dims = {'conv4_3': 38, 'conv7': 19, 'conv8_2': 10, 'conv9_2': 5, 'conv10_2': 3, 'conv11_2': 1}
        self.obj_scales = {'conv4_3': 0.1, 'conv7': 0.2, 'conv8_2': 0.375, 'conv9_2': 0.55, 'conv10_2': 0.725, 'conv11_2': 0.9}

        n_boxes_basic = {'conv4_3': 4, 'conv7': 6, 'conv8_2': 6, 'conv9_2': 6, 'conv10_2': 4, 'conv11_2': 4}
        n_boxes_other = {'conv4_3': 6, 'conv7': 6, 'conv8_2': 6, 'conv9_2': 6, 'conv10_2': 4, 'conv11_2': 4}

        aspect_ratios_basic = {'conv4_3': [1., 2., 0.5], 'conv7': [1., 2., 3., 0.5, .333],
                              'conv8_2': [1., 2., 3., 0.5, .333], 'conv9_2': [1., 2., 3., 0.5, .333],
                              'conv10_2': [1., 2., 0.5], 'conv11_2': [1., 2., 0.5]}
        aspect_ratios_other = {'conv4_3': [1., 2., 3., 0.5, .333], 'conv7': [1., 2., 3., 0.5, .333],
                              'conv8_2': [1., 2., 3., 0.5, .333], 'conv9_2': [1., 2., 3., 0.5, .333],
                              'conv10_2': [1., 2., 0.5], 'conv11_2': [1., 2., 0.5]}


        self.n_boxes = n_boxes_basic if self.net_size == 'small' else n_boxes_other
        self.aspect_ratios = aspect_ratios_basic if self.net_size == 'small'  else aspect_ratios_other

