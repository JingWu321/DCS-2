import torch
import sys
import time
import datetime
import argparse
from pathlib import Path

from utils.util import (
    system_startup, set_random_seed,
    set_deterministic, Logger
)
from data.load_data import load_data
from models.load_model import load_model
from dlg.process_stDifferentDomain import load_process


def get_args_parser():
    parser = argparse.ArgumentParser(description='Test attacks and defenses.')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--output_dir', default='./logs', type=str)
    # Dataset
    parser.add_argument('--root', default='/data/dataset', type=str)
    parser.add_argument('--dataset', default='MNIST', type=str)
    parser.add_argument('--num_workers', default=2, type=int)
    # Setting
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_sen', default=1, type=int)
    parser.add_argument('--batch_idx', default=3, type=int)
    parser.add_argument('--pretrained', default=False, action='store_true')
    parser.add_argument('--demo', default=False, action='store_true', help='Run one batch for demo.')
    # Attacks
    parser.add_argument('--attack', default='gs', type=str, help='dlg, gs, ggl, imprint')
    # dlg and gs attack
    parser.add_argument('--max_iter', default=10000, type=int, help='Max iteration for attack.')
    parser.add_argument('--lr', default=0.1, type=float, help='Learning rate for attack.')
    parser.add_argument('--lr_decay', default=True, action='store_false')
    parser.add_argument('--tv', default=1e-4, type=float, help='Weight of TV penalty.')
    parser.add_argument('--boxed' , default=False, action='store_true',
                        help='Project into image space')
    # imprint attack
    parser.add_argument('--imprint', default='no_sparse', type=str,
                        help='Sparse or others for imprint attack')
    parser.add_argument('--bins', default=10, type=int,
                        help='Number of bins for imprint attack')
    # advance attack
    parser.add_argument('--prior', default=-1, type=int,
                        help='Prior knowledge for advance attacks, -1 for no prior, >=0 to use avg imgs.')
    # Defenses
    parser.add_argument('--defense', default='none', type=str,
                        help='none, soteria, cp, dp, precode, ats, dcs, noise')
    parser.add_argument('--percent_num', default=70, type=float, help='1-40 for soteria and 1-80 for compresion.')
    parser.add_argument('--layer_num', default=6, type=int,
                        help='32 for cifar10, mnist 10 (36 cifar10) for imprintattack with perturb_imprint is False, 1 for True.')
    parser.add_argument('--perturb_imprint', default=False, action='store_true')
    # defense dp
    parser.add_argument('--noise_name', default='Gaussian', type=str)
    parser.add_argument('--loc', default=0., type=float)
    parser.add_argument('--scale', default=1e-4, type=float, help='from 1e-4 to 1e-1.')
    # defense dcs
    parser.add_argument('--per_adv', default=1, type=int, help='>= num_sen.')
    parser.add_argument('--dcs_iter', default=500, type=int)
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
    # defense precode, notice that to compare with precode, shuffle in dataloader need to be False
    parser.add_argument('--precode_size', default=256, type=int)
    parser.add_argument('--beta', default=0.001, type=float)
    # defense ats
    parser.add_argument('--aug_list', default='21-13-3+7-4-15', type=str)
    # Clients
    parser.add_argument('--method', dest='method', default='iid', type=str)
    parser.add_argument('--TotalDevNum', dest='TotalDevNum', default=10, type=int)
    parser.add_argument('--DevNum', dest='DevNum', default=5, type=int)
    parser.add_argument('--n_data', dest='n_data', default=64, type=int, help='Number of data per client.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = get_args_parser()
    save_path = args.output_dir  + '/' + args.dataset + '/' + args.attack
    Path(save_path).mkdir(parents=True, exist_ok=True)
    sys.stdout = Logger(save_path + f'/{args.defense}.csv' , sys.stdout)
    print(args)

    # Choose GPU device and print status information
    device, setup = system_startup()
    set_random_seed(args.seed)
    set_deterministic()

    # Load Dataset
    trainloader, testloader, num_examples, _ = load_data(args)
    mean, std = num_examples["dm"], num_examples["ds"]
    dm = torch.as_tensor(mean, **setup)[None, :, None, None]  # 1xCx1x1
    ds = torch.as_tensor(std, **setup)[None, :, None, None]
    print('Total images {:d} on {}'.format(num_examples['trainset'], args.dataset))

    # Load model
    loss_fn, model, attacker, server_payload, secrets, generator = load_model(args, setup)
    model.eval()

    # Start testing
    print(f'Defense {args.defense} against Attack {args.attack} on Dataset {args.dataset}.')
    start_time = time.time()
    if args.demo:
        for i, (gt_imgs, gt_labels) in enumerate(trainloader):
            if i < args.batch_idx:
                continue
            gt_imgs, gt_labels = gt_imgs.to(device), gt_labels.to(device)
            print(f'Sensitive_labels: {gt_labels[-args.num_sen:].cpu()}')

            _, _, _, proxyloader = load_data(args, dataset='CIFAR10')
            test_psnr, test_ssim, test_lpips, defense_time, reconstructed_data = load_process(
               args, gt_imgs, gt_labels, model, loss_fn, attacker, server_payload, secrets, generator,
               dm, ds, device, save_path, proxyloader
            )

            break
    else:
        mpsnr = []
        mssim = []
        mlpips = []
        for i, (gt_imgs, gt_labels) in enumerate(trainloader):
            gt_imgs, gt_labels = gt_imgs.to(device), gt_labels.to(device)
            if args.prior > -1:
                if gt_labels != args.prior:
                    continue
            _, _, _, proxyloader = load_data(args, dataset='CIFAR10')
            test_psnr, test_ssim, test_lpips, defense_time, reconstructed_data = load_process(
               args, gt_imgs, gt_labels, model, loss_fn, attacker, server_payload, secrets, generator,
               dm, ds, device, save_path, proxyloader
            )
            mpsnr.append(test_psnr)
            mssim.append(test_ssim[0])
            mlpips.append(test_lpips)

        print(torch.mean(torch.stack(mpsnr), dim=0).item(),
              torch.mean(torch.stack(mssim), dim=0).item(),
              torch.mean(torch.stack(mlpips), dim=0).item())



    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    print(f"Finished computations with time: {str(datetime.timedelta(seconds=time.time() - start_time))}")
    # print()

