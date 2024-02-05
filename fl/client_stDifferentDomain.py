import torch
import flwr as fl

import sys
import time
import datetime
import argparse
from pathlib import Path
from collections import OrderedDict

from utils.util import (
    system_startup, set_random_seed,
    set_deterministic, Logger
)
from fl.main_stDifferentDomain import train, test
from data.load_data import load_data
from models.load_model import load_model


def parse_args():

    parser = argparse.ArgumentParser(description='Test under federated learning framework')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--output_dir', default='./logs', type=str)
    parser.add_argument('--demo', default=False, action='store_true')
    parser.add_argument('--pretrained', default=False, action='store_true')
    parser.add_argument('--train_lr', default=0.01, type=float)
    # Dataset
    parser.add_argument('--root', default='/data/dataset', type=str)
    parser.add_argument('--dataset', default='MNIST', type=str)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_sen', default=64, type=int)
    # Defenses
    parser.add_argument('--defense', default='none', type=str)
    # parameter for soteria and compression defense
    parser.add_argument('--attack', default='gs', type=str)
    parser.add_argument('--percent_num', default=70, type=int, help='1-40 for soteria and 1-80 for compresion.')
    parser.add_argument('--layer_num', default=6, type=int, help='(8) for cifar10, mnist 10 (12 cifar10) for imprintattack with perturb_imprint is False, 1 for True.')
    # parameter for dp defense
    parser.add_argument('--noise_name', default='Gaussian', type=str)
    parser.add_argument('--loc', default=0., type=float)
    parser.add_argument('--scale', default=1e-2, type=float, help='from 1e-4 to 1e-1.')
    # parameter for dcs
    parser.add_argument('--per_adv', default=1, type=int, help='>= num_sen.')
    parser.add_argument('--dcs_iter', default=1000, type=int)
    parser.add_argument('--dcs_lr', default=0.1, type=float)
    parser.add_argument('--lambda_xsim', default=0.01, type=float,
                        help='control the contribution from x_sim')
    parser.add_argument('--lambda_zsim', default=0.01, type=float,
                        help='control the contribution from fx_sim, 0.01 on MNIST and 0.1 on CIFAR10')
    parser.add_argument('--epsilon', default=0.01, type=float)
    parser.add_argument('--early_stop' , default=True, action='store_false')
    parser.add_argument('--xsim_thr', default=150., type=float)
    parser.add_argument('--lambda_y', default=0.7, type=float)
    parser.add_argument('--project' , default=True, action='store_false', help='Project <adv_g, ori_g>')
    parser.add_argument('--startpoint', default='none', type=str)
    parser.add_argument('--mixup' , default=True, action='store_false', help='startpoint using mixup')
    # parameter for precode and ats
    parser.add_argument('--precode_size', default=256, type=int)
    parser.add_argument('--beta', default=0.001, type=float)
    parser.add_argument('--aug_list', default='21-13-3+7-4-15', type=str)
    # Clients
    parser.add_argument('--method', default='iid', type=str)
    parser.add_argument('--TotalDevNum', dest='TotalDevNum', default=10, type=int)
    parser.add_argument('--DevNum', dest='DevNum', default=5, type=int)
    parser.add_argument('--n_data', dest='n_data', default=200, type=int, help='Number of data per client.')
    parser.add_argument('--Pro_dev', default=5, type=int)
    args = parser.parse_args()
    return args


class MNISTClient(fl.client.NumPyClient):
    def __init__(self, net, args, device):
        super(MNISTClient, self).__init__()
        self.device = device
        self.net = net
        self.args = args
        self.trainloader, self.testloader, self.num_examples, _ = load_data(args)
        _, _, _, self.proxyloader = load_data(args, dataset='CIFAR10')
        self.dm = torch.as_tensor(self.num_examples['dm'], **setup)[None, :, None, None]  # 1xCx1x1
        self.ds = torch.as_tensor(self.num_examples['ds'], **setup)[None, :, None, None]
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=args.train_lr, momentum=0.9, weight_decay=5e-4)

    def get_parameters(self):
        '''return the model weight as a list of NumPy ndarrays'''
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        '''update the local model weights with the parameters received from the server'''
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        '''set the local model weights, train the local model,
           receive the updated local model weights'''
        self.set_parameters(parameters)
        self.optimizer.param_groups[0]['lr'] = config['lr'] # update learning rate for each round following the config from server
        loss = train(self.loss_fn, self.optimizer, self.net, self.trainloader,
                     config['local_epochs'], self.dm, self.ds, self.args, self.device, self.proxyloader)
        print('Device {:2d} \t Train loss {:.4f} \t LR {:.4f} \t Train images {:} \t Round {:3d}'.format(
              self.args.DevNum, loss, self.optimizer.param_groups[0]['lr'], self.num_examples["trainset"], config['current_round']))
        return self.get_parameters(), self.num_examples["trainset"], {"loss": float(loss), "cid": self.args.DevNum}

    def evaluate(self, parameters, config):
        '''test the local model'''
        self.set_parameters(parameters)
        test_loss, test_acc = test(self.loss_fn, self.net, self.testloader, self.device)
        train_loss, train_acc = test(self.loss_fn, self.net, self.trainloader, self.device)
        return float(test_loss), self.num_examples["testset"], {
            "test_acc": float(test_acc), "train_acc": float(train_acc),
            "train_loss": float(train_loss), "cid": self.args.DevNum}


if __name__ == '__main__':

    args = parse_args()
    save_path_root = args.output_dir + '/' + args.dataset + '/' + args.method + '/' + args.defense
    save_path = save_path_root + '/client_logs'
    Path(save_path).mkdir(parents=True, exist_ok=True)
    sys.stdout = Logger(save_path + f'/client_{args.DevNum}.csv' , sys.stdout)
    print(args)

    # Choose GPU device and print status information
    device, setup = system_startup()
    set_random_seed(args.seed)
    set_deterministic()

    # Load Model
    net = load_model(args, setup)

    start_time = time.time()
    fl.client.start_numpy_client("localhost:8080", client=MNISTClient(net, args, device))

    # Print final timestamp
    print(f'Defense {args.defense} on Dataset {args.dataset}.')
    print(f"Finished computations with time: {str(datetime.timedelta(seconds=time.time() - start_time))}")
    print()
