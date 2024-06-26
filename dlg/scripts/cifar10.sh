# CUDA_VISIBLE_DEVICES=0 python main.py --n_data=64 --dataset='CIFAR10' --defense='none'
# CUDA_VISIBLE_DEVICES=0 python main.py --n_data=64 --dataset='CIFAR10' --defense='dp' --scale=1e-3
# CUDA_VISIBLE_DEVICES=0 python main.py --n_data=64 --dataset='CIFAR10' --defense='cp' --percent_num=90
# CUDA_VISIBLE_DEVICES=0 python main.py --n_data=64 --dataset='CIFAR10' --defense='soteria' --percent_num=50 --layer_num=32
# CUDA_VISIBLE_DEVICES=0 python main.py --n_data=64 --dataset='CIFAR10' --defense='ats' --aug_list='21-13-3+7-4-15+1-2-5-8-10'
# CUDA_VISIBLE_DEVICES=0 python main.py --n_data=64 --dataset='CIFAR10' --defense='precode' --precode_size=256 --beta=1e-3
# CUDA_VISIBLE_DEVICES=0 python main.py --n_data=64 --dataset='CIFAR10' --defense='dcs' --mixup --dcs_iter=1000 --dcs_lr=0.1 --lambda_xsim=0.01 --lambda_zsim=0.01 --lambda_y=0.7 --xsim_thr=500. --epsilon=0.01
# CUDA_VISIBLE_DEVICES=0 python main.py --n_data=64 --dataset='CIFAR10' --defense='dcs' --mixup --dcs_iter=1000 --dcs_lr=0.1 --lambda_xsim=0.01 --lambda_zsim=0.01 --lambda_y=1.0 --xsim_thr=500. --epsilon=0.01

