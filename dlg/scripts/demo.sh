# [MNIST]
# CUDA_VISIBLE_DEVICES=0 python main.py --demo --batch_idx=3 --output_dir='./logs/demo' --n_data=64 --dataset='MNIST' --defense='none'
# CUDA_VISIBLE_DEVICES=0 python main.py --demo --batch_idx=3 --output_dir='./logs/demo' --n_data=64 --dataset='MNIST' --defense='dp' --scale=1e-2
# CUDA_VISIBLE_DEVICES=0 python main.py --demo --batch_idx=3 --output_dir='./logs/demo' --n_data=64 --dataset='MNIST' --defense='cp' --percent_num=70
# CUDA_VISIBLE_DEVICES=0 python main.py --demo --batch_idx=3 --output_dir='./logs/demo' --n_data=64 --dataset='MNIST' --defense='soteria' --percent_num=20 --layer_num=6
# CUDA_VISIBLE_DEVICES=0 python main.py --demo --batch_idx=3 --output_dir='./logs/demo' --n_data=64 --dataset='MNIST' --defense='dcs' --mixup --dcs_iter=1000 --dcs_lr=0.1 --lambda_xsim=0.01 --lambda_zsim=0.01 --lambda_y=0.7 --xsim_thr=300. --epsilon=0.01

# CUDA_VISIBLE_DEVICES=0 python main.py --demo --batch_idx=3 --output_dir='./logs/demo' --n_data=64 --dataset='MNIST' --attack='gs' --defense='none' --batch_size=1 --num_sen=1 --prior=6
# CUDA_VISIBLE_DEVICES=0 python main.py --demo --batch_idx=3 --output_dir='./logs/demo' --n_data=64 --dataset='MNIST' --attack='gs' --defense='dcs' --mixup --dcs_iter=1000 --dcs_lr=0.1 --lambda_xsim=0.01 --lambda_zsim=0.01 --lambda_y=0.7 --xsim_thr=300. --epsilon=0.01 --batch_size=1 --num_sen=1 --prior=6



# [CIFAR10]
# CUDA_VISIBLE_DEVICES=0 python main.py --demo --batch_idx=3 --output_dir='./logs/demo' --n_data=64 --dataset='CIFAR10' --defense='none'
# CUDA_VISIBLE_DEVICES=0 python main.py --demo --batch_idx=3 --output_dir='./logs/demo' --n_data=64 --dataset='CIFAR10' --defense='dp' --scale=1e-3
# CUDA_VISIBLE_DEVICES=0 python main.py --demo --batch_idx=3 --output_dir='./logs/demo' --n_data=64 --dataset='CIFAR10' --defense='cp' --percent_num=70
# CUDA_VISIBLE_DEVICES=0 python main.py --demo --batch_idx=3 --output_dir='./logs/demo' --n_data=64 --dataset='CIFAR10' --defense='soteria' --percent_num=50 --layer_num=32
# CUDA_VISIBLE_DEVICES=0 python main.py --demo --batch_idx=3 --output_dir='./logs/demo' --n_data=64 --dataset='CIFAR10' --defense='dcs' --mixup --dcs_iter=1000 --dcs_lr=0.1 --lambda_xsim=0.01 --lambda_zsim=0.01 --lambda_y=0.7 --xsim_thr=300. --epsilon=0.01

# CUDA_VISIBLE_DEVICES=0 python main.py --demo --batch_idx=5 --output_dir='./logs/demo' --n_data=64 --dataset='CIFAR10' --defense='none'
# CUDA_VISIBLE_DEVICES=0 python main.py --demo --batch_idx=5 --output_dir='./logs/demo' --n_data=64 --dataset='CIFAR10' --defense='dp' --scale=1e-3
# CUDA_VISIBLE_DEVICES=0 python main.py --demo --batch_idx=5 --output_dir='./logs/demo' --n_data=64 --dataset='CIFAR10' --defense='cp' --percent_num=90
# CUDA_VISIBLE_DEVICES=0 python main.py --demo --batch_idx=5 --output_dir='./logs/demo' --n_data=64 --dataset='CIFAR10' --defense='soteria' --percent_num=50 --layer_num=32
# CUDA_VISIBLE_DEVICES=0 python main.py --demo --batch_idx=5 --output_dir='./logs/demo' --n_data=64 --dataset='CIFAR10' --defense='dcs' --mixup --dcs_iter=1000 --dcs_lr=0.1 --lambda_xsim=0.01 --lambda_zsim=0.01 --lambda_y=0.7 --xsim_thr=500. --epsilon=0.01



# [CelebA]
# CUDA_VISIBLE_DEVICES=0 python main.py --demo --batch_idx=3 --output_dir='./logs/demo' --n_data=64 --dataset='CelebA' --attack='ggl' --defense='none'
# CUDA_VISIBLE_DEVICES=0 python main.py --demo --batch_idx=3 --output_dir='./logs/demo' --n_data=64 --dataset='CelebA' --attack='ggl' --defense='dp' --scale=1e-3
# CUDA_VISIBLE_DEVICES=0 python main.py --demo --batch_idx=3 --output_dir='./logs/demo' --n_data=64 --dataset='CelebA' --attack='ggl' --defense='cp' --percent_num=70
# CUDA_VISIBLE_DEVICES=0 python main.py --demo --batch_idx=3 --output_dir='./logs/demo' --n_data=64 --dataset='CelebA' --attack='ggl' --defense='soteria' --percent_num=70 --layer_num=32
# CUDA_VISIBLE_DEVICES=0 python main.py --demo --batch_idx=3 --output_dir='./logs/demo' --n_data=64 --dataset='CelebA' --attack='ggl' --defense='dcs' --mixup --dcs_iter=1000 --dcs_lr=0.1 --lambda_xsim=0.01 --lambda_zsim=0.1 --lambda_y=0.7 --xsim_thr=300. --epsilon=0.01

# CUDA_VISIBLE_DEVICES=0 python main.py --demo --batch_idx=0 --output_dir='./logs/demo' --n_data=64 --dataset='CelebA' --attack='ggl' --defense='none'
# CUDA_VISIBLE_DEVICES=0 python main.py --demo --batch_idx=0 --output_dir='./logs/demo' --n_data=64 --dataset='CelebA' --attack='ggl' --defense='dp' --scale=1e-3
# CUDA_VISIBLE_DEVICES=0 python main.py --demo --batch_idx=0 --output_dir='./logs/demo' --n_data=64 --dataset='CelebA' --attack='ggl' --defense='cp' --percent_num=70
# CUDA_VISIBLE_DEVICES=0 python main.py --demo --batch_idx=0 --output_dir='./logs/demo' --n_data=64 --dataset='CelebA' --attack='ggl' --defense='soteria' --percent_num=70 --layer_num=32
# CUDA_VISIBLE_DEVICES=0 python main.py --demo --batch_idx=0 --output_dir='./logs/demo' --n_data=64 --dataset='CelebA' --attack='ggl' --defense='dcs' --mixup --dcs_iter=1000 --dcs_lr=0.1 --lambda_xsim=0.01 --lambda_zsim=0.1 --lambda_y=0.7 --xsim_thr=500. --epsilon=0.01

# CUDA_VISIBLE_DEVICES=0 python main_stDifferentDomain.py --demo --batch_idx=0 --output_dir='./logs/demo/stDifferentDomain' --n_data=64 --dataset='CelebA' --mixup --attack='ggl' --defense='dcs' --dcs_iter=1000 --dcs_lr=0.1 --lambda_xsim=0.01 --lambda_zsim=0.1 --lambda_y=0.7 --xsim_thr=500. --epsilon=0.01
# CUDA_VISIBLE_DEVICES=0 python main_stDifferentDomain.py --output_dir='./logs/demo/stDifferentDomain' --n_data=64 --dataset='CelebA' --mixup --attack='ggl' --defense='dcs' --dcs_iter=1000 --dcs_lr=0.1 --lambda_xsim=0.01 --lambda_zsim=0.1 --lambda_y=0.7 --xsim_thr=500. --epsilon=0.01




