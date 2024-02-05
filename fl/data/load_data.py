import torch
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10, ImageFolder
from torch.utils.data import DataLoader, Subset, random_split
import os
import numpy as np
from glob import glob

from .load_skin import get_data, CustomDataset
from .load_celeba import CelebA
from .load_qmnist import QMNIST

from defenses.ats import comm


def load_data(args, dataset=None, batch_size=None):

    if dataset is None:
        dataset = args.dataset
    if batch_size is None:
        batch_size = args.batch_size

    if dataset == 'MNIST':
        data_mean = (0.13066047430038452, )
        data_std = (0.30810782313346863,)
        data_transform = transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Normalize(data_mean, data_std)])
        trainset = MNIST(root=args.root, train=True, download=False,
                         transform=data_transform)
        testset = MNIST(root=args.root, train=False, download=False,
                        transform=data_transform)
        total_data = len(trainset.data)
    elif dataset == 'CIFAR10':
        data_mean = (0.4914, 0.4822, 0.4465)
        data_std = (0.247, 0.243, 0.261)
        data_transform = transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Normalize(data_mean, data_std)])
        trainset = CIFAR10(root=args.root, train=True, download=False,
                           transform=data_transform)
        testset = CIFAR10(root=args.root, train=False, download=False,
                          transform=data_transform)
        total_data = len(trainset.data)
        if args.defense == 'ats':
            trainset.transform = comm.build_transform(
                data_mean, data_std, comm.split(args.aug_list))
    elif dataset == 'ImageNet':
        data_mean = (0.485, 0.456, 0.406)
        data_std = (0.229, 0.224, 0.225)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        trainset = ImageFolder(root=args.root + '/' + 'ImageNet/val',
                               transform=transforms.Compose([
                                          transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          normalize,]))
        testset = ImageFolder(root=args.root + '/' + 'ImageNet/val',
                              transform=transforms.Compose([
                                          transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          normalize,]))
        total_data = len(trainset.samples)
    elif dataset == 'HAM10000':
        base_dir = args.root + '/HAM10000/'
        # all_image_path = glob(os.path.join(base_dir, 'train/train_input/*.jpg'))
        all_image_path = glob(os.path.join(base_dir, 'input/*.jpg'))
        imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in all_image_path}
        df_train, df_val = get_data(base_dir, imageid_path_dict)

        data_mean = (0.763, 0.546, 0.570)
        data_std = (0.141, 0.153, 0.170)
        normalize = transforms.Normalize(mean=[0.763, 0.546, 0.570],
                                         std=[0.141, 0.153, 0.170])
        train_transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,])
        test_transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,])
        trainset = CustomDataset(df_train, transform=train_transform)
        testset = CustomDataset(df_val, transform=test_transform)
        total_data = len(df_train)
    elif dataset == 'QMNIST':
        data_mean = (0.13066047430038452, )
        data_std = (0.30810782313346863,)
        data_transform = transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Normalize(data_mean, data_std)])
        trainset = QMNIST(root=args.root, what='train', compat=True, download=False,
                          transform=data_transform)
        testset = QMNIST(root=args.root, what='test10k', compat=True, download=False,
                         transform=data_transform)
        total_data = len(trainset.data)
    elif args.dataset == 'CelebA':
        data_mean = (0.5, 0.5, 0.5) # for fair comparison with GGL
        data_std = (0.5, 0.5, 0.5)
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])
        trainset = CelebA(root=args.root + '/CelebA', train=False,
                          transform=transforms.Compose([
                            transforms.Resize(32),
                            transforms.CenterCrop(32),
                            transforms.ToTensor(),
                            normalize,
                          ]))
        testset = CelebA(root=args.root + '/CelebA', train=False,
                         transform=transforms.Compose([
                            transforms.Resize(32),
                            transforms.CenterCrop(32),
                           transforms.ToTensor(),
                           normalize,
                          ]))
        total_data = len(trainset.images)
    elif args.dataset == 'TinyImageNet':
        data_mean = (0.485, 0.456, 0.406)
        data_std = (0.229, 0.224, 0.225)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        trainset = ImageFolder(root=args.root + '/TinyImageNet/tiny-imagenet-200/train',
                               transform=transforms.Compose([
                                          transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          normalize,]))
        testset = ImageFolder(root=args.root + '/TinyImageNet/tiny-imagenet-200/val',
                              transform=transforms.Compose([
                                          transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          normalize,]))
        total_data = len(trainset.samples)
    else:
        assert False, 'not support the dataset yet.'

    # split trainset into two parts, one for senstive data, the other for proxy data
    len_sen = int(total_data * 0.25)
    len_proxy = total_data - len_sen
    trainset_sens, trainset_proxy = random_split(trainset, [len_sen, len_proxy])
    assert (args.n_data * args.per_adv) * args.TotalDevNum < len_proxy, 'not enough proxy data'

    # split trainset_sens for each client
    data_idx = [[] for _ in range(args.TotalDevNum)]
    if args.method == 'iid':
        idxs = np.random.permutation(len_sen)
        data_idx = np.array_split(idxs[:args.n_data * args.TotalDevNum], args.TotalDevNum)
    elif (args.method == 'non-iid') and (args.dataset == 'MNIST' or args.dataset == 'CIFAR10'):
        train_y = np.array(trainset_sens.targets)
        class_idx = [np.where(train_y==i)[0] for i in range(10)]
        for i in range(args.TotalDevNum):
            idxs = np.random.choice(range(10), 2, replace=False)
            len0 = len(class_idx[idxs[0]])
            len1 = len(class_idx[idxs[1]])
            idxx0 = torch.randint(0, len0, (args.n_data,))
            idxx1 = torch.randint(0, len1, (args.n_data,))
            data_idx[i] = class_idx[idxs[0]][idxx0].tolist() + class_idx[idxs[1]][idxx1].tolist()
    else:
        assert False, 'not support the data split yet.'
    train_subset = Subset(trainset_sens, data_idx[args.DevNum - 1])

    # split trainset_proxy for each client
    proxy_idx = [[] for _ in range(args.TotalDevNum)]
    idxs_proxy = np.random.permutation(len_proxy)
    proxy_idx = np.array_split(idxs_proxy[:(args.n_data * args.per_adv) * args.TotalDevNum], args.TotalDevNum)
    proxy_subset = Subset(trainset_proxy, proxy_idx[args.DevNum - 1])

    # dataloaders
    shuffle_flag = False if args.defense == 'PRECODE' else True # for comparison with defence PRECODE
    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=shuffle_flag,
                             drop_last=True, num_workers=args.num_workers)
    proxyloader = DataLoader(proxy_subset, batch_size=int(batch_size*args.per_adv), shuffle=shuffle_flag,
                             drop_last=True, num_workers=args.num_workers)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, drop_last=True,
                            num_workers=args.num_workers)
    num_examples = {"trainset" : len(train_subset), "testset" : len(testset), "dm": data_mean, "ds": data_std}
    return trainloader, testloader, num_examples, proxyloader
