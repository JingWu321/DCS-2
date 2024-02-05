import torch
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10, ImageFolder
from torch.utils.data import DataLoader, Subset
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
    else:
        assert False, 'not support the dataset yet.'

    # split dataset for each client
    data_idx = [[] for _ in range(args.TotalDevNum)]
    if args.method == 'iid':
        idxs = np.random.permutation(total_data)
        data_idx = np.array_split(idxs[:args.n_data * args.TotalDevNum], args.TotalDevNum)
    elif (args.method == 'non-iid') and (args.dataset == 'MNIST' or args.dataset == 'CIFAR10'):
        train_y = np.array(trainset.targets)
        class_idx = [np.where(train_y==i)[0] for i in range(10)]
        for i in range(args.TotalDevNum):
            idxs = np.random.choice(range(10), 2, replace=False)
            len0 = len(class_idx[idxs[0]])
            len1 = len(class_idx[idxs[1]])
            num = args.n_data
            idxx0 = torch.randint(0, len0, (num,))
            idxx1 = torch.randint(0, len1, (num,))
            data_idx[i] = class_idx[idxs[0]][idxx0].tolist() + class_idx[idxs[1]][idxx1].tolist()
    else:
        assert False, 'not support the data split yet.'

    train_subset = Subset(trainset, data_idx[args.DevNum - 1])
    shuffle_flag = False if args.defense == 'PRECODE' else True # for comparison with defence PRECODE
    trainloader = DataLoader(train_subset, batch_size=batch_size,
                             shuffle=shuffle_flag, drop_last=True,
                             num_workers=args.num_workers)
    testloader = DataLoader(testset, batch_size=batch_size,
                            shuffle=False, drop_last=True,
                            num_workers=args.num_workers)
    num_examples = {"trainset" : len(train_subset), "testset" : len(testset), "dm": data_mean, "ds": data_std}
    return trainloader, testloader, num_examples
