import torch
import torch.nn as nn
from torchvision.utils import save_image
import os
import gc
import numpy as np
import quadprog


# projection
def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    memories_np = memories.cpu().t().contiguous().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    # print("memories_np shape:{}".format(memories_np.shape))
    # print("gradient_np shape:{}".format(gradient_np.shape))
    t = memories_np.shape[0]  # task mums
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0] # get the optimal solution of v~
    x = np.dot(v, memories_np) + gradient_np  # g~ = v*GT +g
    # gradient.copy_(torch.Tensor(x).view(-1))
    new_grad = torch.Tensor(x).view(-1)
    return new_grad


# optimize-based defense for all attacks
def defense_closure(args, model, optimizer, loss_fn, sen_img, sen_out, sen_g, proxy_imgs, proxy_labels):
    def closure():
        optimizer.zero_grad()
        model.zero_grad()

        # proxy data's gradient
        proxy_out, _, _ = model(proxy_imgs)
        proxy_losses = loss_fn(proxy_out, proxy_labels)
        proxy_g = torch.autograd.grad(proxy_losses, model.parameters(), create_graph=True)

        # compute the similarity
        total_loss = 0.
        pnorm = [0, 0]
        rec_loss = 0.
        for gx, gy in zip(sen_g, proxy_g):
            rec_loss += (gx * gy).sum()
            pnorm[0] += gx.pow(2).sum()
            pnorm[1] += gy.pow(2).sum()
        g_sim = 1 - rec_loss / (torch.sqrt(pnorm[0]) * torch.sqrt(pnorm[1]) + 1e-12)

        x_sim = torch.norm(proxy_imgs.reshape(proxy_imgs.size(0), -1) - sen_img.reshape(sen_img.size(0), -1), dim=1).mean()
        fx_sim = (torch.norm(proxy_out - sen_out, dim=1) / torch.norm(sen_out, dim=1)).mean()

        x_tmp = -x_sim*args.lambda_xsim
        total_loss = g_sim + torch.exp(x_tmp) + args.lambda_zsim * (fx_sim-args.epsilon)
        # print(f"loss:{total_loss:.2f}, g_sim:{g_sim:.2f}, x_sim:{x_sim:.2f}, fx_sim:{fx_sim:.2f}, exp(x):{torch.exp(x_tmp):.2f}, z:{args.lambda_zsim * (fx_sim-args.epsilon):.2f}")
        total_loss.backward(retain_graph=True)

        return total_loss, g_sim.item(), x_sim.item(), fx_sim.item()

    return closure


def defense_optim(args, model, loss_fn, gt_imgs, gt_labels, dm, ds, device, proxy_imgs, proxy_labels, save_path):

    model.eval()
    my_criterion = nn.CrossEntropyLoss(reduction='mean')

    # original gradient ori_g
    out, _, _ = model(gt_imgs)
    gt_losses = my_criterion(out, gt_labels)
    gt_gradients = torch.autograd.grad(gt_losses, model.parameters(), retain_graph=True)
    ori_g = torch.cat(list(map(lambda grad: grad.detach().view(-1), gt_gradients)))
    torch.cuda.empty_cache()
    gc.collect()

    # sen_g
    if gt_imgs.size(0) == args.num_sen:
        sen_g = gt_gradients
    else:
        sen_out, _, _ = model(gt_imgs[-args.num_sen:])
        sen_loss = my_criterion(sen_out, gt_labels[-args.num_sen:])
        sen_g = torch.autograd.grad(sen_loss, model.parameters(), retain_graph=True)
    torch.cuda.empty_cache()
    gc.collect()

    sen_img = gt_imgs[-args.num_sen:]
    sen_label = gt_labels[-args.num_sen:]
    # initial concealing data using mixup
    if args.mixup:
        for sk in range(args.num_sen):
            proxy_imgs[sk*args.per_adv: sk*args.per_adv + args.per_adv] = args.lambda_y * proxy_imgs[sk*args.per_adv: sk*args.per_adv + args.per_adv] \
                                                                        + (1 - args.lambda_y) * sen_img[sk].repeat(args.per_adv, 1, 1, 1)

    # craft proxy data
    my_sen_img = torch.repeat_interleave(sen_img, repeats=args.per_adv, dim=0)
    my_senout = torch.repeat_interleave(out[-args.num_sen:], repeats=args.per_adv, dim=0)
    proxy_imgs = proxy_imgs.to(device).requires_grad_(True)
    optimizer = torch.optim.Adam([proxy_imgs], lr=args.dcs_lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[args.dcs_iter // 2.667,
                                                                 args.dcs_iter // 1.6,
                                                                 args.dcs_iter // 1.142],
                                                     gamma=0.1)   # 3/8 5/8 7/8
    for j in range(args.dcs_iter):
        closure = defense_closure(args, model, optimizer, my_criterion,
                                  my_sen_img, my_senout, sen_g, proxy_imgs, proxy_labels)
        rec_loss, _, x_sim, _ = optimizer.step(closure)
        if args.early_stop and (x_sim > args.xsim_thr):
            break
        scheduler.step()
        optimizer.zero_grad()

    # concat tensors (proxy_img and ori_img)
    adv_img = torch.cat([proxy_imgs, gt_imgs], dim=0).to(device)
    adv_label = torch.cat([proxy_labels, gt_labels], dim=0).to(device)
    if args.demo:
        adv_denormalized = torch.clamp(adv_img * ds + dm, 0, 1)
        save_image(adv_denormalized, os.path.join(save_path, f'{args.batch_idx}_dcs.png'))

    # new gradient \ddot{g} after modifying the data
    adv_out, _, _ = model(adv_img)
    # if args.lamday_y is 0 or 1, then divide the loss by 2, otherwise divide by 3
    divd = 2. if args.lambda_y == 0 or args.lambda_y == 1 else 3.
    loss = ((args.lambda_y * loss_fn(adv_out[:args.num_sen * args.per_adv], proxy_labels) \
            + (1 - args.lambda_y) * loss_fn(adv_out[:args.num_sen * args.per_adv], sen_label.repeat(args.per_adv))) \
            + loss_fn(adv_out[args.num_sen * args.per_adv:], gt_labels)) / divd
    adv_dydw = torch.autograd.grad(loss, model.parameters())
    adv_g = torch.cat(list(map(lambda grad: grad.detach().view(-1), adv_dydw)))

    # check if gradient violates constrains
    dotg = torch.mm(adv_g.unsqueeze(0), ori_g.unsqueeze(1))  # view adv_g/ori_g as a vector (1, N) x (N, 1)
    # print(f'dotg: {dotg}, adv_g: {adv_g.unsqueeze(0).size()} and ori_g: {ori_g.unsqueeze(1).size()}')
    # if args.demo:
    print(f'dotg: {(dotg < 0).sum()}')
    if args.project and ((dotg < 0).sum() != 0):
        new_grad = project2cone2(adv_g.unsqueeze(0), ori_g.unsqueeze(1))
        # overwrite current param
        pointer = 0
        dy_dx = []
        for n, p in model.named_parameters():
            num_param = p.numel()
            # p.grad.copy_(new_grad[pointer: pointer + num_param].view_as(p))
            # dy_dx.append(p.grad)
            dy_dx.append(new_grad[pointer: pointer + num_param].view_as(p).to(device))
            pointer += num_param
        gt_gradient = dy_dx
    else:
        # gt_gradient = adv_dydw
        gt_gradient = list(map(lambda grad: grad.detach().clone(), adv_dydw))

    return gt_gradient, adv_img, adv_label, loss
