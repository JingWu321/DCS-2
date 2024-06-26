# CUDA_VISIBLE_DEVICES=0 python main.py --n_data=64 --dataset='MNIST' --defense='dcs' --mixup --dcs_iter=1000 --dcs_lr=0.1 --lambda_xsim=0.01 --lambda_zsim=0.01 --lambda_y=0.7 --xsim_thr=300. --epsilon=0.01
# CUDA_VISIBLE_DEVICES=0 python main.py --n_data=64 --dataset='MNIST' --defense='soteria' --percent_num=20 --layer_num=6
# CUDA_VISIBLE_DEVICES=0 python main.py --n_data=64 --dataset='MNIST' --defense='cp' --percent_num=70
# CUDA_VISIBLE_DEVICES=0 python main.py --n_data=64 --dataset='MNIST' --defense='dp' --scale=1e-2
# CUDA_VISIBLE_DEVICES=0 python main.py --n_data=64 --dataset='MNIST' --defense='none'


# CUDA_VISIBLE_DEVICES=0 python main.py --output_dir='./logs/MNIST/comb' --n_data=64 --dataset='MNIST' --defense='dcs_cp' --mixup --percent_num=70 --dcs_iter=1000 --dcs_lr=0.1 --lambda_xsim=0.01 --lambda_zsim=0.01 --lambda_y=0.7 --xsim_thr=300. --epsilon=0.01
# CUDA_VISIBLE_DEVICES=0 python main.py --output_dir='./logs/MNIST/comb' --n_data=64 --dataset='MNIST' --defense='dcs_dp' --mixup --scale=1e-2 --dcs_iter=1000 --dcs_lr=0.1 --lambda_xsim=0.01 --lambda_zsim=0.01 --lambda_y=0.7 --xsim_thr=300. --epsilon=0.01


# for i in 0 1 2 3 4 5 6 7 8 9
# do
#     CUDA_VISIBLE_DEVICES=0 python main.py --output_dir='./logs/MNIST/prior' --n_data=64 --dataset='MNIST' --attack='gs' --defense='none' --batch_size=1 --num_sen=1 --prior=$i --batch_idx=1
# done

# for i in 0 1 2 3 4 5 6 7 8 9
# do
#     CUDA_VISIBLE_DEVICES=0 python main.py --output_dir='./logs/MNIST/prior' --n_data=64 --dataset='MNIST' --attack='gs' --defense='dcs' --mixup --dcs_iter=1000 --dcs_lr=0.1 --lambda_xsim=0.01 --lambda_zsim=0.01 --lambda_y=0.7 --xsim_thr=300. --epsilon=0.01 --batch_size=1 --num_sen=1 --prior=$i --batch_idx=1
# done


# CUDA_VISIBLE_DEVICES=0 python main.py --output_dir='./logs/MNIST/st_mixup' --n_data=64 --dataset='MNIST' --defense='dcs' --dcs_iter=1000 --dcs_lr=0.1 --lambda_xsim=0.01 --lambda_zsim=0.01 --lambda_y=0.7 --xsim_thr=300. --epsilon=0.01
# CUDA_VISIBLE_DEVICES=0 python main.py --output_dir='./logs/MNIST/st_noise' --startpoint='noise' --n_data=64 --dataset='MNIST' --defense='dcs' --mixup --dcs_iter=1000 --dcs_lr=0.1 --lambda_xsim=0.01 --lambda_zsim=0.01 --lambda_y=0.7 --xsim_thr=300. --epsilon=0.01
# CUDA_VISIBLE_DEVICES=0 python main.py --output_dir='./logs/MNIST/st_noise_mixup' --startpoint='noise' --n_data=64 --dataset='MNIST' --defense='dcs' --dcs_iter=1000 --dcs_lr=0.1 --lambda_xsim=0.01 --lambda_zsim=0.01 --lambda_y=0.7 --xsim_thr=300. --epsilon=0.01


# CUDA_VISIBLE_DEVICES=1 python main.py --output_dir='./logs/MNIST/lambda_y/1' --n_data=64 --dataset='MNIST' --mixup --defense='dcs' --dcs_iter=1000 --dcs_lr=0.1 --lambda_xsim=0.01 --lambda_zsim=0.01 --lambda_y=1.0 --xsim_thr=300. --epsilon=0.01
# CUDA_VISIBLE_DEVICES=1 python main.py --output_dir='./logs/lambda_y/0' --n_data=64 --dataset='MNIST' --defense='dcs' --mixup --dcs_iter=1000 --dcs_lr=0.1 --lambda_xsim=0.01 --lambda_zsim=0.01 --lambda_y=0. --xsim_thr=300. --epsilon=0.01
# CUDA_VISIBLE_DEVICES=1 python main.py --output_dir='./logs/lambda_y/0.1' --n_data=64 --dataset='MNIST' --defense='dcs' --mixup --dcs_iter=1000 --dcs_lr=0.1 --lambda_xsim=0.01 --lambda_zsim=0.01 --lambda_y=0.1 --xsim_thr=300. --epsilon=0.01
# CUDA_VISIBLE_DEVICES=1 python main.py --output_dir='./logs/lambda_y/0.3' --n_data=64 --dataset='MNIST' --defense='dcs' --mixup --dcs_iter=1000 --dcs_lr=0.1 --lambda_xsim=0.01 --lambda_zsim=0.01 --lambda_y=0.3 --xsim_thr=300. --epsilon=0.01
# CUDA_VISIBLE_DEVICES=1 python main.py --output_dir='./logs/lambda_y/0.5' --n_data=64 --dataset='MNIST' --defense='dcs' --mixup --dcs_iter=1000 --dcs_lr=0.1 --lambda_xsim=0.01 --lambda_zsim=0.01 --lambda_y=0.5 --xsim_thr=300. --epsilon=0.01
# CUDA_VISIBLE_DEVICES=1 python main.py --output_dir='./logs/lambda_y/0.9' --n_data=64 --dataset='MNIST' --defense='dcs' --mixup --dcs_iter=1000 --dcs_lr=0.1 --lambda_xsim=0.01 --lambda_zsim=0.01 --lambda_y=0.9 --xsim_thr=300. --epsilon=0.01

