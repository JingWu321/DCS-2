# for j in 1 2 # 3 4 5
# do
#     echo "Job $j is running"
#     # server
#     CUDA_VISIBLE_DEVICES=3 python server.py --minfit=10 --mineval=10 --minavl=10 --num_rounds=50 --dataset='TinyImageNet' --pretrained --train_lr=0.01 --method='iid' --seed=42 --cnt=$j --defense='none' &
#     sleep 5
#     # clients part 1
#     i=1
#     while((i<=5))
#     do
#         CUDA_VISIBLE_DEVICES=3 python client.py --DevNum=$i --dataset='TinyImageNet' --pretrained --train_lr=0.01 --n_data=2000 --batch_size=64 --num_sen=64 --num_workers=2 --method='iid' --TotalDevNum=10 --seed=42 --defense='none' --attack='gs' &
#         echo "Client $i is running"
#         sleep 5
#         let "i+=1"
#     done
#     echo "clients part 1 start"
#     # clients part 2
#     i=6
#     while((i<=10))
#     do
#         CUDA_VISIBLE_DEVICES=3 python client.py --DevNum=$i --dataset='TinyImageNet' --pretrained --train_lr=0.01 --n_data=2000 --batch_size=64 --num_sen=64 --num_workers=2 --method='iid' --TotalDevNum=10 --seed=42 --defense='none' --attack='gs' &
#         echo "Client $i is running"
#         sleep 5
#         let "i+=1"
#     done
#     echo "clients part 2 start"
#     # wait for current cnt to finish
#     wait
# done
# echo "all workers done"

# for j in 1 2 # 3 4 5
# do
#     echo "Job $j is running"
#     # server
#     CUDA_VISIBLE_DEVICES=3 python server.py --minfit=10 --mineval=10 --minavl=10 --num_rounds=50 --dataset='TinyImageNet' --pretrained --train_lr=0.01 --method='iid' --seed=42 --cnt=$j --defense='dp' &
#     sleep 5
#     # clients part 1
#     i=1
#     while((i<=5))
#     do
#         CUDA_VISIBLE_DEVICES=3 python client.py --DevNum=$i --dataset='TinyImageNet' --pretrained --train_lr=0.01 --n_data=2000 --batch_size=64 --num_sen=64 --num_workers=2 --method='iid' --TotalDevNum=10 --seed=42 --defense='dp' --scale=0.5 --attack='gs' &
#         echo "Client $i is running"
#         sleep 5
#         let "i+=1"
#     done
#     echo "clients part 1 start"
#     # clients part 2
#     i=6
#     while((i<=10))
#     do
#         CUDA_VISIBLE_DEVICES=3 python client.py --DevNum=$i --dataset='TinyImageNet' --pretrained --train_lr=0.01 --n_data=2000 --batch_size=64 --num_sen=64 --num_workers=2 --method='iid' --TotalDevNum=10 --seed=42 --defense='dp' --scale=0.5 --attack='gs' &
#         echo "Client $i is running"
#         sleep 5
#         let "i+=1"
#     done
#     echo "clients part 2 start"
#     # wait for current cnt to finish
#     wait
# done
# echo "all workers done"


# for j in 1 2 # 3 4 5
# do
#     echo "Job $j is running"
#     # server
#     CUDA_VISIBLE_DEVICES=3 python server.py --minfit=10 --mineval=10 --minavl=10 --num_rounds=50 --dataset='TinyImageNet' --pretrained --train_lr=0.01 --method='iid' --seed=42 --cnt=$j --defense='cp' &
#     sleep 5
#     # clients part 1
#     i=1
#     while((i<=5))
#     do
#         CUDA_VISIBLE_DEVICES=3 python client.py --DevNum=$i --dataset='TinyImageNet' --pretrained --train_lr=0.01 --n_data=2000 --batch_size=64 --num_sen=64 --num_workers=2 --method='iid' --TotalDevNum=10 --seed=42 --defense='cp' --percent_num=50 --attack='gs' &
#         echo "Client $i is running"
#         sleep 5
#         let "i+=1"
#     done
#     echo "clients part 1 start"
#     # clients part 2
#     i=6
#     while((i<=10))
#     do
#         CUDA_VISIBLE_DEVICES=3 python client.py --DevNum=$i --dataset='TinyImageNet' --pretrained --train_lr=0.01 --n_data=2000 --batch_size=64 --num_sen=64 --num_workers=2 --method='iid' --TotalDevNum=10 --seed=42 --defense='cp' --percent_num=50 --attack='gs' &
#         echo "Client $i is running"
#         sleep 5
#         let "i+=1"
#     done
#     echo "clients part 2 start"
#     # wait for current cnt to finish
#     wait
# done
# echo "all workers done"


# for j in 1 2 # 3 4 5
# do
#     echo "Job $j is running"
#     # server
#     CUDA_VISIBLE_DEVICES=3 python server.py --minfit=10 --mineval=10 --minavl=10 --num_rounds=50 --dataset='TinyImageNet' --pretrained --train_lr=0.01 --method='iid' --seed=42 --cnt=$j --defense='dcs' &
#     sleep 5
#     # clients part 1
#     i=1
#     while((i<=5))
#     do
#         CUDA_VISIBLE_DEVICES=3 python client.py --mixup --DevNum=$i --dataset='TinyImageNet' --pretrained --train_lr=0.01 --n_data=2000 --batch_size=64 --num_sen=64 --num_workers=2 --method='iid' --TotalDevNum=10 --seed=42 --defense='dcs' --lambda_xsim=0.001 --dcs_iter=1000 --epsilon=0.01 --xsim_thr=400. --attack='gs' &
#         echo "Client $i is running"
#         sleep 5
#         let "i+=1"
#     done
#     echo "clients part 1 start"
#     # clients part 2
#     i=6
#     while((i<=10))
#     do
#         CUDA_VISIBLE_DEVICES=3 python client.py --mixup --DevNum=$i --dataset='TinyImageNet' --pretrained --train_lr=0.01 --n_data=2000 --batch_size=64 --num_sen=64 --num_workers=2 --method='iid' --TotalDevNum=10 --seed=42 --defense='dcs' --lambda_xsim=0.001 --dcs_iter=1000 --epsilon=0.01 --xsim_thr=400. --attack='gs' &
#         echo "Client $i is running"
#         sleep 5
#         let "i+=1"
#     done
#     echo "clients part 2 start"
#     # wait for current cnt to finish
#     wait
# done
# echo "all workers done"



# for j in 1 2 # 3 4 5
# do
#     echo "Job $j is running"
#     # server
#     CUDA_VISIBLE_DEVICES=3 python server.py --minfit=10 --mineval=10 --minavl=10 --num_rounds=50 --dataset='TinyImageNet' --pretrained --train_lr=0.01 --method='iid' --seed=42 --cnt=$j --defense='soteria' &
#     sleep 5
#     # clients part 1
#     i=1
#     while((i<=5))
#     do
#         CUDA_VISIBLE_DEVICES=3 python client.py --DevNum=$i --dataset='TinyImageNet' --pretrained --train_lr=0.01 --n_data=2000 --batch_size=64 --num_sen=64 --num_workers=2 --method='iid' --TotalDevNum=10 --seed=42 --defense='soteria' --percent_num=90 --layer_num=60 --attack='gs' &
#         echo "Client $i is running"
#         sleep 5
#         let "i+=1"
#     done
#     echo "clients part 1 start"
#     # clients part 2
#     i=6
#     while((i<=10))
#     do
#         CUDA_VISIBLE_DEVICES=3 python client.py --DevNum=$i --dataset='TinyImageNet' --pretrained --train_lr=0.01 --n_data=2000 --batch_size=64 --num_sen=64 --num_workers=2 --method='iid' --TotalDevNum=10 --seed=42 --defense='soteria' --percent_num=90 --layer_num=60 --attack='gs' &
#         echo "Client $i is running"
#         sleep 5
#         let "i+=1"
#     done
#     echo "clients part 2 start"
#     # wait for current cnt to finish
#     wait
# done
# echo "all workers done"


# for j in 1 2 # 3 4 5
# do
#     echo "Job $j is running"
#     # server
#     CUDA_VISIBLE_DEVICES=3 python server.py --minfit=10 --mineval=10 --minavl=10 --num_rounds=50 --dataset='TinyImageNet' --pretrained --train_lr=0.01 --method='iid' --seed=42 --cnt=$j --defense='dcs' &
#     sleep 5
#     # clients part 1
#     i=1
#     while((i<=5))
#     do
#         CUDA_VISIBLE_DEVICES=3 python client.py --mixup --lambda_y=1. --DevNum=$i --dataset='TinyImageNet' --pretrained --train_lr=0.01 --n_data=2000 --batch_size=64 --num_sen=64 --num_workers=2 --method='iid' --TotalDevNum=10 --seed=42 --defense='dcs' --lambda_xsim=0.001 --dcs_iter=1000 --epsilon=0.01 --xsim_thr=400. --attack='gs' &
#         echo "Client $i is running"
#         sleep 5
#         let "i+=1"
#     done
#     echo "clients part 1 start"
#     # clients part 2
#     i=6
#     while((i<=10))
#     do
#         CUDA_VISIBLE_DEVICES=3 python client.py --mixup --lambda_y=1. --DevNum=$i --dataset='TinyImageNet' --pretrained --train_lr=0.01 --n_data=2000 --batch_size=64 --num_sen=64 --num_workers=2 --method='iid' --TotalDevNum=10 --seed=42 --defense='dcs' --lambda_xsim=0.001 --dcs_iter=1000 --epsilon=0.01 --xsim_thr=400. --attack='gs' &
#         echo "Client $i is running"
#         sleep 5
#         let "i+=1"
#     done
#     echo "clients part 2 start"
#     # wait for current cnt to finish
#     wait
# done
# echo "all workers done"




# for j in 1
# do
#     echo "Job $j is running"
#     # server
#     CUDA_VISIBLE_DEVICES=3 python server.py --output_dir='./logs/time' --minfit=10 --mineval=10 --minavl=10 --num_rounds=1 --dataset='TinyImageNet' --pretrained --train_lr=0.01 --method='iid' --seed=42 --cnt=$j --defense='none' &
#     sleep 5
#     # clients part 1
#     i=1
#     while((i<=5))
#     do
#         CUDA_VISIBLE_DEVICES=3 python client.py --output_dir='./logs/time' --DevNum=$i --dataset='TinyImageNet' --pretrained --train_lr=0.01 --n_data=2000 --batch_size=64 --num_sen=64 --num_workers=2 --method='iid' --TotalDevNum=10 --seed=42 --defense='none' --attack='gs' &
#         echo "Client $i is running"
#         sleep 5
#         let "i+=1"
#     done
#     echo "clients part 1 start"
#     # clients part 2
#     i=6
#     while((i<=10))
#     do
#         CUDA_VISIBLE_DEVICES=3 python client.py --output_dir='./logs/time' --DevNum=$i --dataset='TinyImageNet' --pretrained --train_lr=0.01 --n_data=2000 --batch_size=64 --num_sen=64 --num_workers=2 --method='iid' --TotalDevNum=10 --seed=42 --defense='none' --attack='gs' &
#         echo "Client $i is running"
#         sleep 5
#         let "i+=1"
#     done
#     echo "clients part 2 start"
#     # wait for current cnt to finish
#     wait
# done
# echo "all workers done"


# for j in 1
# do
#     echo "Job $j is running"
#     # server
#     CUDA_VISIBLE_DEVICES=3 python server.py --output_dir='./logs/time' --minfit=10 --mineval=10 --minavl=10 --num_rounds=1 --dataset='TinyImageNet' --pretrained --train_lr=0.01 --method='iid' --seed=42 --cnt=$j --defense='dcs' &
#     sleep 5
#     # clients part 1
#     i=1
#     while((i<=5))
#     do
#         CUDA_VISIBLE_DEVICES=3 python client.py --output_dir='./logs/time' --mixup --lambda_y=0.7 --DevNum=$i --dataset='TinyImageNet' --pretrained --train_lr=0.01 --n_data=2000 --batch_size=64 --num_sen=64 --num_workers=2 --method='iid' --TotalDevNum=10 --seed=42 --defense='dcs' --lambda_xsim=0.001 --dcs_iter=1000 --epsilon=0.01 --xsim_thr=400. --attack='gs' &
#         echo "Client $i is running"
#         sleep 5
#         let "i+=1"
#     done
#     echo "clients part 1 start"
#     # clients part 2
#     i=6
#     while((i<=10))
#     do
#         CUDA_VISIBLE_DEVICES=3 python client.py --output_dir='./logs/time' --mixup --lambda_y=0.7 --DevNum=$i --dataset='TinyImageNet' --pretrained --train_lr=0.01 --n_data=2000 --batch_size=64 --num_sen=64 --num_workers=2 --method='iid' --TotalDevNum=10 --seed=42 --defense='dcs' --lambda_xsim=0.001 --dcs_iter=1000 --epsilon=0.01 --xsim_thr=400. --attack='gs' &
#         echo "Client $i is running"
#         sleep 5
#         let "i+=1"
#     done
#     echo "clients part 2 start"
#     # wait for current cnt to finish
#     wait
# done
# echo "all workers done"
