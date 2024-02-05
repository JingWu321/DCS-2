import torch
import flwr as fl
from flwr.common import (
    EvaluateRes, FitRes, Parameters,
    Scalar, parameters_to_weights,
)
from flwr.server.client_proxy import ClientProxy

import sys
import time
import datetime
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import OrderedDict
from typing import List, Optional, Tuple, Dict

from utils.util import (
    system_startup, set_random_seed,
    set_deterministic, Logger
)
from models.load_model import load_model


def parse_args():
    parser = argparse.ArgumentParser(description='Test under federated learning framework')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--output_dir', default='./logs', type=str)
    parser.add_argument('--cnt', default=1, type=int)
    parser.add_argument('--method', default='iid', type=str)
    parser.add_argument('--TotalDevNum', dest='TotalDevNum', default=10, type=int)
    parser.add_argument('--pretrained', default=False, action='store_true')
    # Dataset
    parser.add_argument('--dataset', default='MNIST', type=str)
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--ckp_path', default='./weights/model_round_100.pth', type=str)
    # Defenses
    parser.add_argument('--defense', default='none', type=str)
    parser.add_argument('--precode_size', default=256, type=int)
    parser.add_argument('--beta', default=0.001, type=float)
    parser.add_argument('--aug_list', default='21-13-3+7-4-15', type=str)
    # server
    parser.add_argument('--minfit', default=10, type=int)
    parser.add_argument('--mineval', default=10, type=int)
    parser.add_argument('--minavl', default=10, type=int)
    parser.add_argument('--num_rounds', default=100, type=int)
    parser.add_argument('--save_rnd', default=100, type=int)
    parser.add_argument('--train_lr', default=0.01, type=float)  # 0.001 for cifar10 non-iid, 0.01 for mnist.
    parser.add_argument('--local_epochs', default=1, type=int)
    args = parser.parse_args()
    return args


class SaveModelAndMetricsStrategy_random(fl.server.strategy.FedAvg):

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]], # FitRes is like EvaluateRes and has a metrics key
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        """Aggregate model weights using weighted average and store checkpoint"""
        aggregated_parameters_tuple = super().aggregate_fit(rnd, results, failures)
        aggregated_parameters, _ = aggregated_parameters_tuple

        if (rnd % save_rnd == 0) and aggregated_parameters is not None:
            print(f"Saving round {rnd} aggregated_parameters...")
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_weights: List[np.ndarray] = parameters_to_weights(aggregated_parameters)

            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(net.state_dict().keys(), aggregated_weights)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)
            save_dir = save_path_root + '/weights'
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), save_dir+f'/ckp_{rnd}_{cnt}.pth')

        return aggregated_parameters_tuple


    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        test_num = list(map(lambda r: r[1].num_examples, results))
        test_loss = list(map(lambda r: r[1].loss, results))
        test_acc = list(map(lambda r: r[1].metrics['test_acc'], results))
        train_loss = list(map(lambda r: r[1].metrics['train_loss'], results))
        train_acc = list(map(lambda r: r[1].metrics['train_acc'], results))
        cid_list = list(map(lambda r: r[1].metrics['cid'], results))
        print('Round {:03d} \t FL_loss {:.4f} \t FL_acc {:.4f} \t Test images {}'.format(
              rnd, test_loss[0], test_acc[0], test_num[0]))

        # record
        FLloss_list.append(round(test_loss[0], 4))
        FLacc_list.append(round(test_acc[0], 4))
        for i in range(Total_dev):
            trainAcc_list[cid_list[i]-1].append(round(train_acc[i], 4))
            trainLoss_list[cid_list[i]-1].append(round(train_loss[i], 4))

        if rnd == 1:
            self.best_acc = 0.
            self.best_rnd = 0
        if self.best_acc < test_acc[0]:
            self.best_acc = test_acc[0]
            self.best_rnd = rnd
        if rnd == Total_rnds:
            print(f'Best FL accuracy: {self.best_acc} on round {self.best_rnd}')

        # Call aggregate_evaluate from base class (FedAvg)
        return super().aggregate_evaluate(rnd, results, failures)


def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config = {
        "current_round": server_round,
        "local_epochs": local_epochs,
        "lr": train_lr*(0.99**server_round),
    }
    return config


if __name__ == '__main__':

    args = parse_args()
    save_path_root = args.output_dir + '/' + args.dataset + '/' + args.method + '/' + args.defense
    save_path = save_path_root + '/server_logs'
    Path(save_path).mkdir(parents=True, exist_ok=True)
    sys.stdout = Logger(save_path + f'/server_{args.cnt}.csv' , sys.stdout)
    print(args)

    # Choose GPU device and print status information
    device, setup = system_startup()
    set_random_seed(args.seed)
    set_deterministic()

    # Load Model
    net = load_model(args, setup)

    # Whether to load Pre-trained Model
    if args.resume:
        # load Pre-trained Model
        ckp = torch.load(args.ckp_path)
        init_params = [val.cpu().numpy() for _, val in ckp.items()]
    else:
        init_params = None

    # Global Variables
    cnt = args.cnt
    Total_dev = args.TotalDevNum
    Total_rnds = args.num_rounds
    train_lr = args.train_lr
    local_epochs = args.local_epochs
    save_rnd = args.save_rnd
    FLloss_list = []
    FLacc_list = []
    trainAcc_list = [[] for i in range(Total_dev)]
    trainLoss_list = [[] for i in range(Total_dev)]
    mydict = {}

    # Create strategy and run server
    strategy = SaveModelAndMetricsStrategy_random(
        min_fit_clients=args.minfit, # Minimum number of clients to be sampled for the next round
        min_eval_clients=args.mineval, # Minimum number of clients used during validation
        min_available_clients=args.minavl, # Minimum number of clients that need to be connected to the server before a training round can start
        on_fit_config_fn=fit_config, # Function that returns the training configuration for each round
        initial_parameters=init_params, # Initial model parameters
    )
    start_time = time.time()
    fl.server.start_server("[::]:8080", config={"num_rounds": args.num_rounds}, strategy=strategy)

    # save the results
    mydict.update({'S_acc': FLacc_list})
    mydict.update({'S_loss': FLloss_list})
    for i in range(Total_dev):
        mydict.update({f'C{i+1}_acc': trainAcc_list[i]})
        mydict.update({f'C{i+1}_loss': trainLoss_list[i]})
    dataframe = pd.DataFrame(mydict)
    dataframe.to_csv(save_path_root + f'/results_{args.cnt}.csv', index=False, sep=',')

    # Print final timestamp
    print('Defense Method: ', args.defense)
    print(f"Finished computations with time: {str(datetime.timedelta(seconds=time.time() - start_time))}")
    print()
