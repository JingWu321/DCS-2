import torch
import torch.nn as nn
from collections import OrderedDict

from .variational_bottleneck import VariationalBottleneck


class LeNet_MNIST(nn.Module):
    """LeNet variant from https://github.com/mit-han-lab/dlg/blob/master/models/vision.py."""
    def __init__(self):
        """3-Layer sigmoid Conv with large linear layer."""
        super(LeNet_MNIST, self).__init__()
        act = nn.Sigmoid
        # act = nn.ReLU
        self.features = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            act(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(588, 10)
        )

        self.apply(self.weights_init)

    def forward(self, x):
        out = self.features(x)
        feature = out.view(out.size(0), 588)
        out = self.classifier(feature)
        return out, feature, x

    @staticmethod
    def weights_init(m):
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)


# for imprintattack
class LeNet_MNIST_imp(nn.Module):
    def __init__(self):
        """3-Layer sigmoid Conv with large linear layer."""
        super(LeNet_MNIST_imp, self).__init__()
        act = nn.Sigmoid
        # act = nn.ReLU
        self.features = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            act(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(588, 10)
        )

        self.apply(self.weights_init)

    def forward(self, x):
        x_in = x[0].view(x[0].size(0), 1, 28, 28)
        out = self.features(x_in)
        feature = out.view(out.size(0), 588)
        out = self.classifier(feature)
        return out, feature, x[1]

    @staticmethod
    def weights_init(m):
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)


class LeNet_PRECODE(nn.Module):
    """LeNet variant from https://github.com/mit-han-lab/dlg/blob/master/models/vision.py."""
    def __init__(self, hidden_size, beta=1e-3):
        """3-Layer sigmoid Conv with large linear layer."""
        super(LeNet_PRECODE, self).__init__()
        act = nn.Sigmoid
        self.features = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            act(),
            nn.Flatten(),
        )
        # self.hidden2mu = nn.Linear(588, hidden_size)
        # self.hidden2log_var = nn.Linear(588, hidden_size)
        # self.classifier = nn.Sequential(
        #     nn.Linear(hidden_size, 588),
        #     nn.Linear(588, 10),
        # )
        # self.hidden_size = hidden_size
        # self.setup = setup

        self.VB = VariationalBottleneck((588,), K=hidden_size, beta=beta)
        self.classifier = nn.Sequential(
            nn.Linear(588, 10)
        )

        self.apply(self.weights_init)

    # def set_z(self, z_value):
    #     self.z = z_value

    # def reparametrize(self, mu, log_var, z):
    #     sigma = torch.exp(0.5*log_var)
    #     self.sigma = sigma
    #     return mu + sigma*z

    def forward(self, x):
        hidden = self.features(x)
        # self.hidden = hidden
        # mu = self.hidden2mu(hidden)
        # log_var = self.hidden2log_var(hidden)
        # self.mu = mu
        # self.log_var = log_var
        # z = torch.randn([x.size()[0],self.hidden_size]).to(**self.setup)
        # feature = self.reparametrize(mu, log_var, z)
        feature = self.VB(hidden)
        out = self.classifier(feature)
        return out, hidden, x

    # calculates the VBLoss
    def loss(self):
        return self.VB.loss()

    @staticmethod
    def weights_init(m):
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)


# for imprintattack
class LeNet_PRECODE_imp(nn.Module):
    def __init__(self, hidden_size, beta=1e-3):
        """3-Layer sigmoid Conv with large linear layer."""
        super(LeNet_PRECODE_imp, self).__init__()
        act = nn.Sigmoid
        self.features = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            act(),
            nn.Flatten(),
        )
        # self.hidden2mu = nn.Linear(588, hidden_size)
        # self.hidden2log_var = nn.Linear(588, hidden_size)
        # self.classifier = nn.Sequential(
        #     nn.Linear(hidden_size, 588),
        #     nn.Linear(588, 10),
        # )
        # self.hidden_size = hidden_size
        # self.setup = setup

        self.VB = VariationalBottleneck((588,), K=hidden_size, beta=beta)
        self.classifier = nn.Sequential(
            nn.Linear(588, 10)
        )

        self.apply(self.weights_init)

    # def set_z(self, z_value):
    #     self.z = z_value

    # def reparametrize(self, mu, log_var, z):
    #     sigma = torch.exp(0.5*log_var)
    #     self.sigma = sigma
    #     return mu + sigma*z

    def forward(self, x):
        x_in = x[0].view(x[0].size(0), 1, 28, 28)
        hidden = self.features(x_in)
        # self.hidden = hidden
        # mu = self.hidden2mu(hidden)
        # log_var = self.hidden2log_var(hidden)
        # self.mu = mu
        # self.log_var = log_var
        # z = torch.randn([x_in.size()[0],self.hidden_size]).to(**self.setup)
        # feature = self.reparametrize(mu, log_var, z)
        feature = self.VB(hidden)
        out = self.classifier(feature)
        return out, hidden, x[1]

    # calculates the VBLoss
    def loss(self):
        return self.VB.loss()

    @staticmethod
    def weights_init(m):
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)


class ConvNet(nn.Module):
    """ConvNetBN."""

    def __init__(self, width=32, num_classes=10, num_channels=3):
        """Init with width and num classes."""
        super(ConvNet, self).__init__()
        self.model = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(num_channels, 1 * width, kernel_size=3, padding=1)),
            ('bn0', nn.BatchNorm2d(1 * width)),
            ('relu0', nn.ReLU()),

            ('conv1', nn.Conv2d(1 * width, 2 * width, kernel_size=3, padding=1)),
            ('bn1', nn.BatchNorm2d(2 * width)),
            ('relu1', nn.ReLU()),

            ('conv2', nn.Conv2d(2 * width, 2 * width, kernel_size=3, padding=1)),
            ('bn2', nn.BatchNorm2d(2 * width)),
            ('relu2', nn.ReLU()),

            ('conv3', nn.Conv2d(2 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn3', nn.BatchNorm2d(4 * width)),
            ('relu3', nn.ReLU()),

            ('conv4', nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn4', nn.BatchNorm2d(4 * width)),
            ('relu4', nn.ReLU()),

            ('conv5', nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn5', nn.BatchNorm2d(4 * width)),
            ('relu5', nn.ReLU()),

            ('pool0', nn.MaxPool2d(3)),

            ('conv6', nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn6', nn.BatchNorm2d(4 * width)),
            ('relu6', nn.ReLU()),

            ('conv6', nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn6', nn.BatchNorm2d(4 * width)),
            ('relu6', nn.ReLU()),

            ('conv7', nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn7', nn.BatchNorm2d(4 * width)),
            ('relu7', nn.ReLU()),

            ('pool1', nn.MaxPool2d(3))
        ]))
        self.linear = nn.Linear(36 * width, num_classes)

    def forward(self, input):
        out = self.model(input)
        feature = out.view(out.size(0), -1)
        out = self.linear(feature)
        return out, feature, input


class ConvNet_imp(nn.Module):
    """ConvNetBN."""

    def __init__(self, width=32, num_classes=10, num_channels=3):
        """Init with width and num classes."""
        super(ConvNet_imp, self).__init__()
        self.model = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(num_channels, 1 * width, kernel_size=3, padding=1)),
            ('bn0', nn.BatchNorm2d(1 * width)),
            ('relu0', nn.ReLU()),

            ('conv1', nn.Conv2d(1 * width, 2 * width, kernel_size=3, padding=1)),
            ('bn1', nn.BatchNorm2d(2 * width)),
            ('relu1', nn.ReLU()),

            ('conv2', nn.Conv2d(2 * width, 2 * width, kernel_size=3, padding=1)),
            ('bn2', nn.BatchNorm2d(2 * width)),
            ('relu2', nn.ReLU()),

            ('conv3', nn.Conv2d(2 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn3', nn.BatchNorm2d(4 * width)),
            ('relu3', nn.ReLU()),

            ('conv4', nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn4', nn.BatchNorm2d(4 * width)),
            ('relu4', nn.ReLU()),

            ('conv5', nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn5', nn.BatchNorm2d(4 * width)),
            ('relu5', nn.ReLU()),

            ('pool0', nn.MaxPool2d(3)),

            ('conv6', nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn6', nn.BatchNorm2d(4 * width)),
            ('relu6', nn.ReLU()),

            ('conv6', nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn6', nn.BatchNorm2d(4 * width)),
            ('relu6', nn.ReLU()),

            ('conv7', nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn7', nn.BatchNorm2d(4 * width)),
            ('relu7', nn.ReLU()),

            ('pool1', nn.MaxPool2d(3))
        ]))
        self.linear = nn.Linear(36 * width, num_classes)

    def forward(self, x):
        x_in = x[0].view(x[0].size(0), 3, 32, 32)
        out = self.model(x_in)
        feature = out.view(out.size(0), -1)
        out = self.linear(feature)
        return out, feature, x[1]


class ConvNet_PRECODE(nn.Module):
    """ConvNetBN."""

    def __init__(self, hidden_size, beta=1e-3, width=32, num_classes=10, num_channels=3):
        """Init with width and num classes."""
        super(ConvNet_PRECODE, self).__init__()
        self.model = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(num_channels, 1 * width, kernel_size=3, padding=1)),
            ('bn0', nn.BatchNorm2d(1 * width)),
            ('relu0', nn.ReLU()),

            ('conv1', nn.Conv2d(1 * width, 2 * width, kernel_size=3, padding=1)),
            ('bn1', nn.BatchNorm2d(2 * width)),
            ('relu1', nn.ReLU()),

            ('conv2', nn.Conv2d(2 * width, 2 * width, kernel_size=3, padding=1)),
            ('bn2', nn.BatchNorm2d(2 * width)),
            ('relu2', nn.ReLU()),

            ('conv3', nn.Conv2d(2 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn3', nn.BatchNorm2d(4 * width)),
            ('relu3', nn.ReLU()),

            ('conv4', nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn4', nn.BatchNorm2d(4 * width)),
            ('relu4', nn.ReLU()),

            ('conv5', nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn5', nn.BatchNorm2d(4 * width)),
            ('relu5', nn.ReLU()),

            ('pool0', nn.MaxPool2d(3)),

            ('conv6', nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn6', nn.BatchNorm2d(4 * width)),
            ('relu6', nn.ReLU()),

            ('conv6', nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn6', nn.BatchNorm2d(4 * width)),
            ('relu6', nn.ReLU()),

            ('conv7', nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn7', nn.BatchNorm2d(4 * width)),
            ('relu7', nn.ReLU()),

            ('pool1', nn.MaxPool2d(3))
        ]))

        # self.hidden2mu = nn.Linear(36 * width, hidden_size)
        # self.hidden2log_var = nn.Linear(36 * width, hidden_size)
        # self.hidden_size = hidden_size
        # self.setup = setup
        # self.linear = nn.Sequential(
        #     nn.Linear(hidden_size, 36 * width),
        #     nn.Linear(36 * width, num_classes),
        # )

        self.VB = VariationalBottleneck((36 * width,), K=hidden_size, beta=beta)
        self.linear = nn.Linear(36 * width, num_classes)

    # def reparametrize(self, mu, log_var, z):
    #     sigma = torch.exp(0.5*log_var)
    #     self.sigma = sigma
    #     return mu + sigma*z

    def forward(self, input):
        out = self.model(input)
        hidden = out.view(out.size(0), -1)
        # precode
        # self.hidden = hidden
        # mu = self.hidden2mu(hidden)
        # log_var = self.hidden2log_var(hidden)
        # self.mu = mu
        # self.log_var = log_var
        # z = torch.randn([input.size()[0], self.hidden_size]).to(**self.setup)
        # feature = self.reparametrize(mu, log_var, z)
        feature = self.VB(hidden)
        # classify
        out = self.linear(feature)
        return out, feature, input

    # calculates the VBLoss
    def loss(self):
        return self.VB.loss()


class ConvNet_PRECODE_imp(nn.Module):
    """ConvNetBN."""

    def __init__(self, hidden_size, beta=1e-3, width=32, num_classes=10, num_channels=3):
        """Init with width and num classes."""
        super(ConvNet_PRECODE_imp, self).__init__()
        self.model = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(num_channels, 1 * width, kernel_size=3, padding=1)),
            ('bn0', nn.BatchNorm2d(1 * width)),
            ('relu0', nn.ReLU()),

            ('conv1', nn.Conv2d(1 * width, 2 * width, kernel_size=3, padding=1)),
            ('bn1', nn.BatchNorm2d(2 * width)),
            ('relu1', nn.ReLU()),

            ('conv2', nn.Conv2d(2 * width, 2 * width, kernel_size=3, padding=1)),
            ('bn2', nn.BatchNorm2d(2 * width)),
            ('relu2', nn.ReLU()),

            ('conv3', nn.Conv2d(2 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn3', nn.BatchNorm2d(4 * width)),
            ('relu3', nn.ReLU()),

            ('conv4', nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn4', nn.BatchNorm2d(4 * width)),
            ('relu4', nn.ReLU()),

            ('conv5', nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn5', nn.BatchNorm2d(4 * width)),
            ('relu5', nn.ReLU()),

            ('pool0', nn.MaxPool2d(3)),

            ('conv6', nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn6', nn.BatchNorm2d(4 * width)),
            ('relu6', nn.ReLU()),

            ('conv6', nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn6', nn.BatchNorm2d(4 * width)),
            ('relu6', nn.ReLU()),

            ('conv7', nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn7', nn.BatchNorm2d(4 * width)),
            ('relu7', nn.ReLU()),

            ('pool1', nn.MaxPool2d(3))
        ]))

        # self.hidden2mu = nn.Linear(36 * width, hidden_size)
        # self.hidden2log_var = nn.Linear(36 * width, hidden_size)
        # self.hidden_size = hidden_size
        # self.setup = setup
        # self.linear = nn.Sequential(
        #     nn.Linear(hidden_size, 36 * width),
        #     nn.Linear(36 * width, num_classes),
        # )

        self.VB = VariationalBottleneck((36 * width,), K=hidden_size, beta=beta)
        self.linear = nn.Linear(36 * width, num_classes)

    # def reparametrize(self, mu, log_var, z):
    #     sigma = torch.exp(0.5*log_var)
    #     self.sigma = sigma
    #     return mu + sigma*z

    def forward(self, x):
        x_in = x[0].view(x[0].size(0), 3, 32, 32)
        out = self.model(x_in)
        hidden = out.view(out.size(0), -1)
        # precode
        # self.hidden = hidden
        # mu = self.hidden2mu(hidden)
        # log_var = self.hidden2log_var(hidden)
        # self.mu = mu
        # self.log_var = log_var
        # z = torch.randn([x_in.size()[0], self.hidden_size]).to(**self.setup)
        # feature = self.reparametrize(mu, log_var, z)
        feature = self.VB(hidden)
        # classify
        out = self.linear(feature)
        return out, feature, x[1]

    # calculates the VBLoss
    def loss(self):
        return self.VB.loss()

