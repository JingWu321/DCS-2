import torch
import torch.nn as nn

from .net import (
    LeNet_MNIST, LeNet_MNIST_imp, LeNet_PRECODE, LeNet_PRECODE_imp,
    ConvNet, ConvNet_imp, ConvNet_PRECODE, ConvNet_PRECODE_imp
)
from .resnet import resnet18, resnet18_imp
from .ggl_net import Generator
from attacks import attacks


def load_model(args, setup):
    attacker = None
    server_payload = None
    secrets = None
    generator = None
    loss_fn = nn.CrossEntropyLoss()

    if args.dataset == 'MNIST':
        if args.attack == 'imprint':
            model = LeNet_MNIST_imp()
            if args.defense == 'precode':
                model = LeNet_PRECODE_imp(args.precode_size, beta=args.beta)
        else:
            model = LeNet_MNIST()
            if args.defense == 'precode':
                model = LeNet_PRECODE(args.precode_size, beta=args.beta)
    elif args.dataset == 'CIFAR10':
        if args.attack == 'imprint':
            if args.defense == 'precode':
                model = ConvNet_PRECODE_imp(args.precode_size, beta=args.beta, width=32, num_classes=10, num_channels=3)
            else:
                model = ConvNet_imp(width=32, num_classes=10, num_channels=3)
        else:
            if args.defense == 'precode':
                model = ConvNet_PRECODE(args.precode_size, beta=args.beta, width=32, num_classes=10, num_channels=3)
            else:
                model = ConvNet(width=32, num_classes=10, num_channels=3)
    elif args.dataset == 'CelebA':
        if args.attack == 'imprint':
            model = ConvNet_imp(width=32, num_classes=2, num_channels=3)
        else:
            model = ConvNet(width=32, num_classes=2, num_channels=3)
    elif args.dataset == 'ImageNet':
        if args.attack == 'imprint':
            model = resnet18_imp(pretrained=args.pretrained)
        else:
            model = resnet18(pretrained=args.pretrained)
    elif args.dataset == 'HAM10000':
        if args.attack == 'imprint':
            model = resnet18_imp(pretrained=args.pretrained)
        else:
            model = resnet18(pretrained=args.pretrained)
        fc = getattr(model, 'fc')
        feature_dim = fc.in_features
        setattr(model,'fc', torch.nn.Linear(feature_dim, 7))
    elif args.dataset == 'TinyImageNet':
        if args.attack == 'imprint':
            model = resnet18_imp(pretrained=args.pretrained)
        else:
            model = resnet18(pretrained=args.pretrained)
        fc = getattr(model, 'fc')
        feature_dim = fc.in_features
        setattr(model,'fc', torch.nn.Linear(feature_dim, 200))

    if args.attack == 'imprint':
        model, attacker, server_payload, secrets = attacks.Imprint_setting(
            args, model, loss_fn, setup)
    elif args.attack == 'ggl':
        generator = Generator()
        checkpoint = torch.load('./models/celeba_wgan-gp_generator_32.pth.tar')
        generator.load_state_dict(checkpoint['state_dict'])
        generator.eval()
        generator.to(**setup)

    model.to(**setup)
    return loss_fn, model, attacker, server_payload, secrets, generator

