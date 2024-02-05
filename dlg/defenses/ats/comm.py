import torchvision.transforms as transforms
import numpy as np

from .policy import policies


class sub_transform:
    def __init__(self, policy_list):
        self.policy_list = policy_list

    def __call__(self, img):
        idx = np.random.randint(0, len(self.policy_list))
        select_policy = self.policy_list[idx]
        for policy_id in select_policy:
            img = policies[policy_id](img)
        return img


def construct_policy(policy_list):
    if isinstance(policy_list[0], list):
        return sub_transform(policy_list)
    elif isinstance(policy_list[0], int):
        return sub_transform([policy_list])
    else:
        raise NotImplementedError


def build_transform(data_mean, data_std, policy_list=list()):
    transform_list = list()
    if len(policy_list) > 0:
        transform_list.append(construct_policy(policy_list))
    print(transform_list)

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std),
    ])
    transform = transforms.Compose(transform_list)
    return transform


def split(aug_list):
    if '+' not in aug_list:
        return [int(idx) for idx in aug_list.split('-')]
    else:
        ret_list = list()
        for aug in aug_list.split('+'):
            ret_list.append([int(idx) for idx in aug.split('-')])
        return ret_list

