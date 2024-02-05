import torch
from timm.utils import accuracy, AverageMeter
import gc

from defenses import dp_cp, soteria, dcs


def train(loss_fn, optimizer, model, trainloader, epochs, dm, ds, args, device, proxyloader):
    model.train()
    train_losses = AverageMeter()

    for epoch_idx in range(epochs):
        model.train()
        for batch_idx, (gt_imgs, gt_labels) in enumerate(trainloader):
            gt_imgs, gt_labels = gt_imgs.to(device), gt_labels.to(device)

            # apply defense
            if (args.DevNum == args.Pro_dev) and args.defense != 'none':
                # print('Apply defenses')

                # Load starpoint if needed
                for j, (proxy_imgs, proxy_labels) in enumerate(proxyloader):
                    if j < batch_idx:
                        continue
                    proxy_imgs = proxy_imgs.to(device)
                    # randomly generate labels has same size as proxy_labels
                    proxy_labels = torch.randint(0, 2, (proxy_labels.size(0),)).to(device)
                    break
                if args.startpoint == 'noise':
                    # print('Initial with noise')
                    proxy_imgs = torch.randn_like(proxy_imgs).to(device)
                else:
                    # print(f'Initial with {args.dataset}')
                    pass

                model.eval()
                if args.defense == 'dcs':
                    protect_gradient, _, _, gt_loss = dcs.defense_optim(
                        args, model, loss_fn, gt_imgs, gt_labels, dm, ds, device,
                        proxy_imgs, proxy_labels, save_path=None)
                elif args.defense == 'precode':
                    out, _, _ = model(gt_imgs)
                    # mu, log_var = model.mu, model.log_var
                    # benign_loss = loss_fn(out, gt_labels)
                    # kl_loss = (-0.5*(1+log_var - mu**2 - torch.exp(log_var)).sum(dim=1)).mean(dim=0)
                    # gt_loss = benign_loss + 0.003*kl_loss
                    gt_loss = loss_fn(out, gt_labels) + model.loss() # add the VB loss to the overall loss
                    gt_gradients = torch.autograd.grad(gt_loss, model.parameters())
                    protect_gradient = [grad.detach().clone() for grad in gt_gradients]
                elif args.defense == 'ats':
                    out, _, _ = model(gt_imgs)
                    gt_loss = loss_fn(out, gt_labels)
                    gt_gradients = torch.autograd.grad(gt_loss, model.parameters())
                    protect_gradient = [grad.detach().clone() for grad in gt_gradients]
                elif args.defense == 'dp':
                    out, _, _ = model(gt_imgs)
                    gt_loss = loss_fn(out, gt_labels)
                    gt_gradients = torch.autograd.grad(gt_loss, model.parameters())
                    protect_gradient = dp_cp.defense_dp(gt_gradients, device, args.loc, args.scale, args.noise_name)
                elif args.defense == 'cp':
                    out, _, _ = model(gt_imgs)
                    gt_loss = loss_fn(out, gt_labels)
                    gt_gradients = torch.autograd.grad(gt_loss, model.parameters())
                    protect_gradient = dp_cp.defense_cp(gt_gradients, device, args.percent_num)
                elif args.defense == 'soteria':
                    protect_gradient, gt_loss = soteria.defense_soteria(
                        args, gt_imgs, gt_labels, model, loss_fn, device, args.layer_num, args.percent_num
                    )
                elif args.defense == 'dcs_cp':
                    protect_gradient, _, _, gt_loss = dcs.defense_optim(
                        args, model, loss_fn, gt_imgs, gt_labels, dm, ds, device,
                        proxy_imgs, proxy_labels, save_path=None)
                    protect_gradient = dp_cp.defense_cp(protect_gradient, device, args.percent_num)
                elif args.defense == 'dcs_dp':
                    protect_gradient, _, _, gt_loss = dcs.defense_optim(
                        args, model, loss_fn, gt_imgs, gt_labels, dm, ds, device,
                        proxy_imgs, proxy_labels, save_path=None)
                    protect_gradient = dp_cp.defense_dp(protect_gradient, device, args.loc, args.scale, args.noise_name)
                else:
                    assert False, 'Not support other defenses yet.'
                torch.cuda.empty_cache()
                gc.collect()
                # model.train()
                # out, _, _ = model(gt_imgs)
                # loss = loss_fn(out, gt_labels)
                # loss.backward()
                # torch.cuda.empty_cache()
                # gc.collect()

                # Overwrite current gradient with protected gradient
                loss = gt_loss
                model.train()
                pointer = 0
                for n, p in model.named_parameters():
                    if p.grad is not None:
                        p.grad.copy_(protect_gradient[pointer].view_as(p))
                    pointer += 1
                torch.cuda.empty_cache()
                gc.collect()
                optimizer.step()
                optimizer.zero_grad()

            else:
                # print('No defenses!')
                model.train()
                out, _, _ = model(gt_imgs)
                loss = loss_fn(out, gt_labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            train_losses.update(loss, gt_imgs.size(0))
            torch.cuda.empty_cache()
            gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
    return train_losses.avg


def test(criterion, net, testloader, device):
    """Validate the network on the entire test set."""
    net.eval()
    top1 = AverageMeter()
    test_losses = AverageMeter()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs, _, _ = net(images)
            acc1 = accuracy(outputs, labels, topk=(1,))
            top1.update(acc1[0], images.size(0))
            loss = criterion(outputs, labels)
            test_losses.update(loss, images.size(0))
    test_loss = test_losses.avg.item()
    test_acc = top1.avg.item()
    return test_loss, test_acc

