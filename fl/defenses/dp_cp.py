import torch
import gc
import numpy as np


# model compression
def defense_cp(gt_gradients, device, percent_num=10):

    gt_gradient = [grad.detach().clone() for grad in gt_gradients]
    for i in range(len(gt_gradient)):
        grad_tensor = gt_gradient[i].cpu().numpy()
        flattened_weights = np.abs(grad_tensor.flatten())
        # Generate the pruning threshold according to 'prune by percentage'.
        thresh = np.percentile(flattened_weights, percent_num)
        grad_tensor = np.where(abs(grad_tensor) < thresh, 0, grad_tensor)
        gt_gradient[i] = torch.Tensor(grad_tensor).to(device)

    return gt_gradient


def defense_cppartial(loss_fn, model, gt_images, gt_labels, device, percent_num, num_sen):
    pred_y, _, _ = model(gt_images)
    losses = loss_fn(pred_y, gt_labels)
    accumulated_gradient = []
    for idx in range(len(losses)):
        gt_gradients = torch.autograd.grad(losses[idx], model.parameters(), retain_graph=True)
        if idx > (gt_images.size(0) - num_sen) - 1: # prune the gradient of the sensitive samples
            gt_gradients = defense_cp(gt_gradients, device, percent_num)

        gt_gradient = [grad.detach().clone() for grad in gt_gradients]
        if idx == 0:
            accumulated_gradient = gt_gradient
        else:
            for i in range(len(accumulated_gradient)):
                accumulated_gradient[i] += gt_gradient[i]
        torch.cuda.empty_cache()
        gc.collect()
    gt_gradient = [grad.detach().clone() / len(losses) for grad in accumulated_gradient]
    return gt_gradient, losses.mean()


# adding noise into gradients
def defense_dp(gt_gradients, device, loc, scale, noise_name):

    gt_gradient = [grad.detach().clone() for grad in gt_gradients]
    for i in range(len(gt_gradient)):
        grad_tensor = gt_gradient[i].cpu().numpy()
        # grad_tensor = gt_gradient[i]
        if noise_name == 'Laplace':
            noise = np.random.laplace(loc, scale, size=grad_tensor.shape)
        else:
            noise = np.random.normal(loc, scale, size=grad_tensor.shape)
            # noise = torch.normal(loc, scale, size=grad_tensor.shape).to(device)
        # print(f'mu:{loc - torch.mean(noise)}')
        # print(f'std:{scale - torch.std(noise)}')
        grad_tensor = grad_tensor + noise
        gt_gradient[i] = torch.Tensor(grad_tensor).to(device)
        # gt_gradient[i] = grad_tensor + noise

    return gt_gradient


def defense_dppartial(loss_fn, model, gt_images, gt_labels, device, loc, scale, noise_name, num_sen):
    pred_y, _, _ = model(gt_images)
    losses = loss_fn(pred_y, gt_labels)
    accumulated_gradient = []
    for idx in range(len(losses)):
        gt_gradients = torch.autograd.grad(losses[idx], model.parameters(), retain_graph=True)
        if idx > (gt_images.size(0) - num_sen) - 1: # add noise to the gradient of the sensitive samples
            gt_gradients = defense_dp(gt_gradients, device, loc, scale, noise_name)

        gt_gradient = [grad.detach().clone() for grad in gt_gradients]
        if idx == 0:
            accumulated_gradient = gt_gradient
        else:
            for i in range(len(accumulated_gradient)):
                accumulated_gradient[i] += gt_gradient[i]
        torch.cuda.empty_cache()
        gc.collect()
    gt_gradient = [grad.detach().clone() / len(losses) for grad in accumulated_gradient]
    return gt_gradient, losses.mean()



# differential privacy with clipping
def defense_dpsgd(loss_fn, model, gt_images, gt_labels, device, loc, scale, noise_name):
    pred_y, _, _ = model(gt_images)
    losses = loss_fn(pred_y, gt_labels)
    max_norm = 15.0
    accumulated_gradient = []
    for idx in range(len(losses)):
        gt_gradients = torch.autograd.grad(losses[idx], model.parameters(), retain_graph=True)
        total_norm = torch.norm(torch.stack([torch.norm(grad.detach(), 2).to(device) for grad in gt_gradients]), 2)
        print(f'previous norm:{total_norm}')
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        per_gradient = [grad.detach().clone().mul_(clip_coef_clamped) for grad in gt_gradients]
        print(f'current norm:{torch.norm(torch.stack([torch.norm(grad.detach(), 2).to(device) for grad in per_gradient]), 2)}')
        if idx == 0:
            accumulated_gradient = per_gradient
        else:
            for i in range(len(accumulated_gradient)):
                accumulated_gradient[i] += per_gradient[i]
        torch.cuda.empty_cache()
        gc.collect()
        # print(f'per_gradient[0]:{per_gradient[0].size()}')
    # print(f'accu_gradient:{len(accumulated_gradient)}')

    gt_gradient = [grad.detach().clone() / len(losses) for grad in accumulated_gradient]
    for i in range(len(gt_gradient)):
        grad_tensor = gt_gradient[i].cpu().numpy()
        # grad_tensor = gt_gradient[i]
        if noise_name == 'Laplace':
            noise = np.random.laplace(loc, scale*scale*max_norm*max_norm, size=grad_tensor.shape)
        else:
            noise = np.random.normal(loc, scale*scale*max_norm*max_norm, size=grad_tensor.shape)
        grad_tensor = grad_tensor + noise
        gt_gradient[i] = torch.Tensor(grad_tensor).to(device)

    return gt_gradient

