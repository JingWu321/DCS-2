import torch
import gc
import numpy as np


def defense_soteria(args, gt_images, gt_labels, model, loss_fn, device, layer_num, percent_num=1, perturb_imprint=False):
    ## compute ||d(f(r))/dX||/||r||
    ## use ||r||/||d(f(r))/dX|| to approximate ||r(d(f(r))/dX)^-1||
    model.eval()
    model.zero_grad()
    gt_images.requires_grad = True
    if perturb_imprint:
        out, _, feature_fc1_graph = model(gt_images)  # perturb the imprint module
    else:
        out, feature_fc1_graph, _ = model(gt_images)
    deviation_f1_target = torch.zeros_like(feature_fc1_graph)
    deviation_f1_x_norm = torch.zeros_like(feature_fc1_graph)

    for f in range(deviation_f1_x_norm.size(1)):
        deviation_f1_target[:,f] = 1
        feature_fc1_graph.backward(deviation_f1_target, retain_graph=True)
        deviation_f1_x = gt_images.grad.data  # df(x)/dx
        if args.attack == 'dlg':
            deviation_f1_x_norm[:,f] = torch.norm(
                deviation_f1_x.view(deviation_f1_x.size(0), -1), dim=1)/(feature_fc1_graph.data[:,f])
        else:
            deviation_f1_x_norm[:,f] = torch.norm(
                deviation_f1_x.view(deviation_f1_x.size(0), -1), dim=1)/(feature_fc1_graph.data[:,f] + 0.1)
        model.zero_grad()
        gt_images.grad.data.zero_()
        deviation_f1_target[:,f] = 0
        del deviation_f1_x
        torch.cuda.empty_cache()
        gc.collect()

    # prune r_i corresponding to smallest ||d(f(r_i))/dX||/||r_i||
    deviation_f1_x_norm_sum = deviation_f1_x_norm.sum(axis=0)
    thresh = np.percentile(deviation_f1_x_norm_sum.flatten().cpu().numpy(), percent_num)
    mask = np.where(abs(deviation_f1_x_norm_sum.cpu()) < thresh, 0, 1).astype(np.float32)
    # print(sum(mask))

    gt_loss = loss_fn(out, gt_labels)
    gt_gradients = torch.autograd.grad(gt_loss, model.parameters())
    # for grad in gt_gradients:
    #     print(grad.size())
    gt_gradient = [grad.detach().clone() for grad in gt_gradients]
    # perturb gradtients
    # print(gt_gradient[layer_num].size(), torch.Tensor(mask).size())
    gt_gradient[layer_num] = gt_gradient[layer_num] * torch.Tensor(mask).to(device)
    del deviation_f1_target, deviation_f1_x_norm
    del deviation_f1_x_norm_sum, feature_fc1_graph
    torch.cuda.empty_cache()
    gc.collect()

    return gt_gradient, gt_loss


def defense_soteriapartial(args, loss_fn, model, gt_images, gt_labels, device, layer_num, percent_num, num_sen):
    pred_y, _, _ = model(gt_images[:-num_sen])
    loss = loss_fn(pred_y, gt_labels[:-num_sen])
    gt_gradients = torch.autograd.grad(loss, model.parameters())
    sen_gradient, sen_loss = defense_soteria(args, gt_images[-num_sen:], gt_labels[-num_sen:], model, loss_fn, device, layer_num, percent_num)

    accumulated_gradient = []
    gt_gradient = [grad.detach().clone() for grad in gt_gradients]
    accumulated_gradient = gt_gradient
    for i in range(len(accumulated_gradient)):
        accumulated_gradient[i] += sen_gradient[i]
    torch.cuda.empty_cache()
    gc.collect()

    gt_gradient = [grad.detach().clone() / 2 for grad in accumulated_gradient]
    return gt_gradient, (loss+sen_loss)/2


