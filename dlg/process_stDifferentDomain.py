import torch
from torchvision.utils import save_image
import os
import time
import datetime

from attacks import attacks
from defenses import dcs, dp_cp, soteria
from utils.metrics import psnr, ssim_batch, lpips_loss


def load_process(args, gt_imgs, gt_labels, model, loss_fn,
                 attacker, server_payload, secrets, generator,
                 dm, ds, device, save_path, proxyloader):

    # Load starpoint if needed
    for j, (proxy_imgs, proxy_labels) in enumerate(proxyloader):
        # print(f'Batch {j}, proxy_labels: {proxy_labels.size()}')
        if j < args.batch_idx:
            continue
        proxy_imgs = proxy_imgs.to(device)
        # randomly generate labels has same size as proxy_labels
        proxy_labels = torch.randint(0, 2, (proxy_labels.size(0),)).to(device)
        print(f'Proxy_labels: {proxy_labels.cpu()}')
        break
    if args.startpoint == 'noise':
        print('Initial with noise')
        proxy_imgs = torch.randn_like(proxy_imgs).to(device)
    else:
        print(f'Initial with {args.dataset}')


    # defense
    model.eval()
    st = time.time()
    if args.defense == 'none':
        out, _, _ = model(gt_imgs)
        gt_loss = loss_fn(out, gt_labels)
        gt_gradients = torch.autograd.grad(gt_loss, model.parameters())
        protect_gradient = [grad.detach().clone() for grad in gt_gradients]
    elif args.defense == 'dcs':
        protect_gradient, adv_imgs, adv_labels, _ = dcs.defense_optim(
            args, model, loss_fn, gt_imgs, gt_labels, dm, ds, device,
            proxy_imgs, proxy_labels, save_path)
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
        protect_gradient = soteria.defense_soteria(
            args, gt_imgs, gt_labels, model, loss_fn, device, args.layer_num, args.percent_num
        )
    elif args.defense == 'dcs_cp':
        protect_gradient, _, _, _ = dcs.defense_optim(
            args, model, loss_fn, gt_imgs, gt_labels, dm, ds, device,
            proxy_imgs, proxy_labels, save_path)
        protect_gradient = dp_cp.defense_cp(protect_gradient, device, args.percent_num)
    elif args.defense == 'dcs_dp':
        protect_gradient, _, _, _ = dcs.defense_optim(
            args, model, loss_fn, gt_imgs, gt_labels, dm, ds, device,
            proxy_imgs, proxy_labels, save_path)
        protect_gradient = dp_cp.defense_dp(protect_gradient, device, args.loc, args.scale, args.noise_name)
    else:
        assert False, 'Not support other defenses yet.'
    defense_time = time.time() - st
    print(f"Finished defence with time: {str(datetime.timedelta(seconds=time.time() - st))}")

    # attack
    if args.attack == 'dlg' or args.attack == 'gs':
        reconstructed_data = attacks.DLG_attack(
            args, protect_gradient, gt_imgs, gt_labels, model, loss_fn, dm, ds, device)
    elif args.attack == 'imprint':
        if args.defense == 'dcs':
            gt_labels = adv_labels # fed all ground truth labels
        reconstructed_data = attacks.Robbing_attack(
            protect_gradient, gt_labels, attacker, server_payload, secrets)
    elif args.attack == 'ggl' and args.dataset == 'CelebA':
        reconstructed_z = attacks.GGl_attack(
            args, generator, protect_gradient, gt_imgs, gt_labels, model, loss_fn, dm, ds, device)
        reconstructed_data = generator(reconstructed_z.float())
    else:
        assert False, 'Not support other attacks yet.'


    # metrics
    if reconstructed_data is not None:
        # if args.attack == 'imprint' and args.defense == 'dcs':
        #     idx = 0
        #     output_denormalized = torch.clamp(reconstructed_data * ds + dm, 0, 1)
        #     gt_denormalized = torch.clamp(gt_imgs * ds + dm, 0, 1)
        #     test_psnr = psnr(output_denormalized[idx].unsqueeze(0), gt_denormalized[idx].unsqueeze(0), batched=False, factor=1.)
        #     test_ssim = ssim_batch(output_denormalized[idx].unsqueeze(0), gt_denormalized[idx].unsqueeze(0))
        #     if args.dataset == 'ImageNet' or args.dataset == 'TinyImageNet':
        #         test_lpips = lpips_loss(output_denormalized[idx].unsqueeze(0).cpu(), gt_denormalized[idx].unsqueeze(0).cpu())
        #     else:
        #         test_lpips = torch.tensor(-0.).to(device)
        # else:
        idx = 0 - args.num_sen
        output_denormalized = torch.clamp(reconstructed_data * ds + dm, 0, 1)
        gt_denormalized = torch.clamp(gt_imgs * ds + dm, 0, 1)
        test_psnr = psnr(output_denormalized[idx:], gt_denormalized[idx:], batched=False, factor=1.)
        test_ssim = ssim_batch(output_denormalized[idx:], gt_denormalized[idx:])
        if args.dataset == 'ImageNet' or args.dataset == 'TinyImageNet':
            test_lpips = lpips_loss(output_denormalized[idx:].cpu(), gt_denormalized[idx:].cpu())
        else:
            test_lpips = torch.tensor(-0.).to(device)
        print('PSNR {:.4f} SSIM {:.4f} LPIPS {:.4f}'.format(test_psnr.item(), test_ssim[0].item(), test_lpips.item()))
        # save
        if args.demo:
            rc_filename = str(args.batch_idx) + '_psnr' + str(round(test_psnr.item(), 4)) + '.png'
            save_image(output_denormalized, os.path.join(save_path, rc_filename))
            gt_filename = str(args.batch_idx) + '_gt.png'
            save_image(gt_denormalized, os.path.join(save_path, gt_filename))
    else:
        print('Attack failed.')
        return None, None, None, defense_time, reconstructed_data

    return test_psnr, test_ssim, test_lpips, defense_time, reconstructed_data
