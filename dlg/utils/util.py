import torch
import random
import numpy as np
import datetime
import socket
import sys
from collections import namedtuple


def system_startup(args=None, defs=None):
    """Print useful system information."""
    # Choose GPU device and print status information:
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    setup = dict(device=device, dtype=torch.float)  # non_blocking=NON_BLOCKING
    print('Currently evaluating -------------------------------:')
    # print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    print(f'CPUs: {torch.get_num_threads()}, GPUs: {torch.cuda.device_count()} on {socket.gethostname()}.')
    if args is not None:
        print(args)
    if defs is not None:
        print(repr(defs))
    if torch.cuda.is_available():
        print(f'GPU : {torch.cuda.get_device_name(device=device)}')
    return device, setup

def set_random_seed(seed=233):
    torch.manual_seed(seed + 1)
    torch.cuda.manual_seed(seed + 2)
    torch.cuda.manual_seed_all(seed + 3)
    np.random.seed(seed + 4)
    # torch.cuda.manual_seed_all(seed + 5)
    random.seed(seed + 6)

def set_deterministic():
    """Switch pytorch into a deterministic computation mode."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Logger(object):
    def __init__(self, filename='default.csv', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
	    pass

class mnist_data_cfg_default:
    classes = 10
    shape = (1, 28, 28)
    mean = (0.13066047430038452, )
    std = (0.30810782313346863,)

class cifar10_data_cfg_default:
    classes = 10
    shape = (3, 32, 32)
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.247, 0.243, 0.261)

class imagenet_data_cfg_default:
    classes = 1000
    shape = (3, 224, 224)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

class skin_data_cfg_default:
    classes = 7
    shape = (3, 224, 224)
    mean = (0.763, 0.546, 0.570)
    std = (0.141, 0.153, 0.170)

class celeba32_data_cfg_default:
    classes = 2
    shape = (3, 32, 32)
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

class tinyimagenet_data_cfg_default:
    classes = 200
    shape = (3, 224, 224)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

class attack_cfg_default:
    type = "analytic"
    attack_type = "imprint-readout"
    label_strategy = "random"  # Labels are not actually required for this attack
    normalize_gradients = False
    impl = namedtuple("impl", ["dtype", "mixed_precision", "JIT"])("float", False, "")
    # impl = namedtuple("impl", ["dtype", "mixed_precision", "JIT"])("float", False, "trace")
    # init = "randn"

