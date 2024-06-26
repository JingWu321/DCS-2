# for ((i = 0; i < 64; ++i))
# do
#     CUDA_VISIBLE_DEVICES=0 python main.py --n_data=64 --dataset='TinyImageNet' --attack='imprint' --output_dir='./logs/TinyImageNet/none' --defense='none' --demo --batch_idx=$i
# done

# for ((i = 0; i < 64; ++i))
# do
#     CUDA_VISIBLE_DEVICES=0 python main.py --n_data=64 --dataset='TinyImageNet' --attack='imprint' --output_dir='./logs/TinyImageNet/dp' --defense='dp' --scale=0.5 --demo --batch_idx=$i
# done

# for ((i = 0; i < 64; ++i))
# do
#     CUDA_VISIBLE_DEVICES=1 python main.py --n_data=64 --dataset='TinyImageNet' --attack='imprint' --output_dir='./logs/TinyImageNet/cp' --defense='cp' --percent_num=50 --demo --batch_idx=$i
# done

# for ((i = 0; i < 64; ++i))
# do
#     CUDA_VISIBLE_DEVICES=0 python main.py --n_data=64 --dataset='TinyImageNet' --attack='imprint' --output_dir='./logs/TinyImageNet/soteria' --defense='soteria' --percent_num=90 --layer_num=64 --demo --batch_idx=$i
# done


# # for ((i = 0; i < 64; ++i))
# for i in 0 1 3 4 5 6 7 13 14 18 21 23 24 25 26 27 28 29 34 35 36 39 40 46 48 53 56 57 59 60
# # for i in 2 8 9 10 11 12 15 16 17 19 20 22 30 31 32 33 37 38 41 42 43 44 45 47 49 50 51 52 53 54 55 58 61 62 63
# do
#     CUDA_VISIBLE_DEVICES=0 python main.py --n_data=64 --dataset='TinyImageNet' --attack='imprint' --output_dir='./logs/TinyImageNet/dcs' --defense='dcs' --mixup --dcs_iter=1000 --dcs_lr=0.1 --lambda_xsim=0.001 --lambda_zsim=0.01 --lambda_y=0.7 --xsim_thr=400. --epsilon=0.01 --demo --batch_idx=$i
# done



# # for ((i = 0; i < 64; ++i))
# for i in 0 1 2 3 4 5 6 7 8 13 14 18 21 23 24 25 28 29 35 36 39 40 46 48 53 56 57 59 60
# # for i in 9 10 11 12 15 16 17 19 20 22 26 27 30 31 32 33 34 37 38 41 42 43 44 45 47 49 50 51 52 54 55 58 61 62 63
# do
#     CUDA_VISIBLE_DEVICES=1 python main.py --n_data=64 --dataset='TinyImageNet' --attack='imprint' --output_dir='./logs/lambda_y/1' --defense='dcs' --mixup --dcs_iter=1000 --dcs_lr=0.1 --lambda_xsim=0.001 --lambda_zsim=0.01 --lambda_y=1.0 --xsim_thr=400. --epsilon=0.01 --demo --batch_idx=$i
# done

# # # for ((i = 0; i < 64; ++i))
# # 30
# for i in 0 1 2 3 4 5 6 7 9 13 14 18 21 23 24 25 27 28 31 34 35 36 40 43 45 46 48 53 57 59 60 61
# # for i in 8 10 11 12 15 16 17 19 20 22 26 29 32 33 37 38 39 41 42 44 47 49 50 51 52 54 55 56 58 62 63
# do
#     CUDA_VISIBLE_DEVICES=1 python main.py --n_data=64 --dataset='TinyImageNet' --attack='imprint' --output_dir='./logs/lambda_y/0' --defense='dcs' --mixup --dcs_iter=1000 --dcs_lr=0.1 --lambda_xsim=0.001 --lambda_zsim=0.01 --lambda_y=0. --xsim_thr=400. --epsilon=0.01 --demo --batch_idx=$i
# done


# # for ((i = 0; i < 64; ++i))
# # 30
# for i in 0 1 2 3 4 5 6 7 13 14 18 21 23 24 25 26 27 28 31 34 35 36 40 43 44 45 46 48 49 53 57 59 60 61
# # for i in 8 9 10 11 12 15 16 17 19 20 22 29 32 33 37 38 39 41 42 47 50 51 52 54 55 56 58 62 63
# do
#     CUDA_VISIBLE_DEVICES=1 python main.py --n_data=64 --dataset='TinyImageNet' --attack='imprint' --output_dir='./logs/lambda_y/0.1' --lambda_y=0.1 --defense='dcs' --mixup --dcs_iter=1000 --dcs_lr=0.1 --lambda_xsim=0.001 --lambda_zsim=0.01 --xsim_thr=400. --epsilon=0.01 --demo --batch_idx=$i
# done

# # for ((i = 0; i < 64; ++i))
# for i in 0 1 2 3 4 5 6 7 13 14 18 21 23 24 25 26 27 28 31 34 35 36 40 43 44 45 46 48 49 53 57 59 60 61
# # for i in 8 9 10 11 12 15 16 17 19 20 22 29 32 33 37 38 39 41 42 47 50 51 52 54 55 56 58 62 63
# do
#     CUDA_VISIBLE_DEVICES=1 python main.py --n_data=64 --dataset='TinyImageNet' --attack='imprint' --output_dir='./logs/lambda_y/0.3' --lambda_y=0.3 --defense='dcs' --mixup --dcs_iter=1000 --dcs_lr=0.1 --lambda_xsim=0.001 --lambda_zsim=0.01 --xsim_thr=400. --epsilon=0.01 --demo --batch_idx=$i
# done

# # for ((i = 0; i < 64; ++i))
# for i in 0 1 2 3 4 5 6 7 13 14 18 21 23 24 25 26 27 28 31 34 35 36 40 43 44 45 46 48 49 53 57 59 60 61
# # for i in 8 9 10 11 12 15 16 17 19 20 22 29 32 33 37 38 39 41 42 47 50 51 52 54 55 56 58 62 63
# do
#     CUDA_VISIBLE_DEVICES=1 python main.py --n_data=64 --dataset='TinyImageNet' --attack='imprint' --output_dir='./logs/lambda_y/0.5' --lambda_y=0.5 --defense='dcs' --mixup --dcs_iter=1000 --dcs_lr=0.1 --lambda_xsim=0.001 --lambda_zsim=0.01 --xsim_thr=400. --epsilon=0.01 --demo --batch_idx=$i
# done

# # for ((i = 0; i < 64; ++i))
# for i in 0 1 2 3 4 5 6 7 8 10 13 14 18 21 23 24 25 26 28 29 31 35 36 39 40 41 42 44 45 46 48 49 53 56 57 59 60 61
# # for i in 9 11 12 15 16 17 19 20 22 27 30 32 33 34 37 38 43 47 50 51 52 54 55 58 62 63
# do
#     CUDA_VISIBLE_DEVICES=1 python main.py --n_data=64 --dataset='TinyImageNet' --attack='imprint' --output_dir='./logs/lambda_y/0.9' --lambda_y=0.9 --defense='dcs' --mixup --dcs_iter=1000 --dcs_lr=0.1 --lambda_xsim=0.001 --lambda_zsim=0.01 --xsim_thr=400. --epsilon=0.01 --demo --batch_idx=$i
# done



