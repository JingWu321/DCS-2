import csv
import statistics


root = './logs/MNIST/noniid/gs/'
file_list = [
    'none.csv',
    'dp.csv',
    'cp.csv',
    'soteria.csv',
    'dcs.csv',
    # 'dcs_mixup.csv',
    # 'dcs_noise.csv',
    # 'dcs_noise_mixup.csv',
    # 'dcs_cp.csv',
    # 'none_prior.csv',
    # 'dcs_prior.csv',
    # 'lambda_y/0/dcs.csv',
    # 'lambda_y/0.1/dcs.csv',
    # 'lambda_y/0.3/dcs.csv',
    # 'lambda_y/0.5/dcs.csv',
    # 'lambda_y/0.7/dcs.csv',
    # 'lambda_y/0.9/dcs.csv',
    # 'lambda_y/1/dcs.csv',
]


# root = './logs/CIFAR10/gs/'
# file_list = [
#     'none.csv',
#     'dp.csv',
#     'cp.csv',
#     'soteria.csv',
#     'ats.csv',
#     'dcs.csv',
#     'dcs(lambda_y=1).csv',
# ]


# root = './logs/CelebA/ggl/'
# file_list = [
#     'none.csv',
#     'dp.csv',
#     'cp.csv',
#     'soteria.csv',
#     'dcs.csv',
#     'dcs(lambda_y=0).csv',
#     'dcs(lambda_y=1).csv',
# ]

# root = './logs/TinyImageNet/'
# file_list = [
#     # 'none.csv',
#     # 'dp(G0.5).csv',
#     # 'cp(p50).csv',
#     # 'soteria(p90l64)_no0.1.csv',
#     # 'dcs.csv',
#     'lambda_y/0/TinyImageNet/imprint/dcs.csv',
#     'lambda_y/0.1/TinyImageNet/imprint/dcs.csv',
#     'lambda_y/0.3/TinyImageNet/imprint/dcs.csv',
#     'lambda_y/0.5/TinyImageNet/imprint/dcs.csv',
#     'lambda_y/0.7/TinyImageNet/imprint/dcs.csv',
#     'lambda_y/0.9/TinyImageNet/imprint/dcs.csv',
#     'lambda_y/1/TinyImageNet/imprint/dcs.csv',
# ]

num = 64
for idx in range(len(file_list)):
    print(f'================== {file_list[idx]} ==================')
    file_name = root + file_list[idx]
    csvreader = csv.reader(open(file_name), delimiter=' ')
    psnr_avg = []
    ssim_avg = []
    lpips_avg = []
    cnt = 1
    for line in csvreader:
        if line[0] == 'PSNR':
            # print(line[1], line[3], line[5])
            psnr_avg.append(float(line[1]))
            ssim_avg.append(float(line[3]))
            lpips_avg.append(float(line[5]))
            cnt += 1

    print(f'psnr: mean {statistics.mean(psnr_avg[:num])} with std {statistics.stdev(psnr_avg[:num])}')
    print(f'ssim: mean {statistics.mean(ssim_avg[:num])} with std {statistics.stdev(ssim_avg[:num])}')
    print(f'lpips: mean {statistics.mean(lpips_avg[:num])} with std {statistics.stdev(lpips_avg[:num])}')
    print()
