
#from eval import evaluate

from utils import *


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class Config():

    def __init__(self):
        
        
        # Model parameters
        # Not too many here since the SSD300 has a very specific structure
        self.n_classes = len(label_map)  # number of different types of objects
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Learning parameters
        path = "./checkpoint_ssd300.pth.tar"
        if os.path.exists(path):
          self.checkpoint = path  # path to model checkpoint, None if none
        else:
          self.checkpoint = None
        self.batch_size = 16  # batch size
        # self.iterations = 120 # number of iterations to train
        self.iterations = 120000 # number of iterations to train
        self.workers = 0  # number of workers for loading data in the DataLoader
        self.print_freq = 50  # print training status every __ batches
        self.lr = 1e-3  # learning rate
        self.decay_lr_at = [80000, 100000]  # decay learning rate after these many iterations
        self.decay_lr_to = 0.1  # decay learning rate to this fraction of the existing learning rate
        self.momentum = 0.9  # momentum
        self.weight_decay = 5e-4  # weight decay
        self.grad_clip = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation
        self.epochs = 20

