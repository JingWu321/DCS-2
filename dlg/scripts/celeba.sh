# CUDA_VISIBLE_DEVICES=1 python main_gp.py --output_dir='./logs/dcs_nopro/stDifferentDomain' --n_data=64 --dataset='CelebA' --mixup --attack='ggl' --defense='dcs' --dcs_iter=1000 --dcs_lr=0.1 --lambda_xsim=0.01 --lambda_zsim=0.1 --lambda_y=1.0 --xsim_thr=500. --epsilon=0.01

# CUDA_VISIBLE_DEVICES=1 python main.py --n_data=64 --dataset='CelebA' --attack='ggl' --defense='none'
# CUDA_VISIBLE_DEVICES=1 python main.py --n_data=64 --dataset='CelebA' --attack='ggl' --defense='dp' --scale=1e-3
# CUDA_VISIBLE_DEVICES=1 python main.py --n_data=64 --dataset='CelebA' --attack='ggl' --defense='cp' --percent_num=70
# CUDA_VISIBLE_DEVICES=1 python main.py --n_data=64 --dataset='CelebA' --attack='ggl' --defense='soteria' --percent_num=70 --layer_num=32
# CUDA_VISIBLE_DEVICES=1 python main.py --n_data=64 --dataset='CelebA' --attack='ggl' --defense='dcs' --mixup --dcs_iter=1000 --dcs_lr=0.1 --lambda_xsim=0.01 --lambda_zsim=0.1 --lambda_y=0.7 --xsim_thr=500. --epsilon=0.01


# CUDA_VISIBLE_DEVICES=1 python main.py --output_dir='./logs/lambda_y/1' --n_data=64 --dataset='CelebA' --attack='ggl' --defense='dcs' --mixup --dcs_iter=1000 --dcs_lr=0.1 --lambda_xsim=0.01 --lambda_zsim=0.1 --lambda_y=1.0 --xsim_thr=500. --epsilon=0.01
# CUDA_VISIBLE_DEVICES=1 python main.py --output_dir='./logs/lambda_y/0' --n_data=64 --dataset='CelebA' --attack='ggl' --defense='dcs' --mixup --dcs_iter=1000 --dcs_lr=0.1 --lambda_xsim=0.01 --lambda_zsim=0.1 --lambda_y=0. --xsim_thr=500. --epsilon=0.01


