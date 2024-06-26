# for j in 1 2 # 3 4 5
# do
#     echo "Job $j is running"
#     # server
#     CUDA_VISIBLE_DEVICES=3 python server_cifar10.py --minfit=10 --mineval=10 --minavl=10 --num_rounds=50 --train_lr=0.01 --method='iid' --dataset='CIFAR10' --seed=42 --cnt=$j --defense='none' &
#     sleep 5
#     # clients part 1
#     i=1
#     while((i<=5))
#     do
#         CUDA_VISIBLE_DEVICES=3 python client_cifar10.py --DevNum=$i --n_data=2000 --method='iid' --dataset='CIFAR10' --TotalDevNum=10 --seed=42 --defense='none' --attack='gs' &
#         echo "Client $i is running"
#         sleep 5
#         let "i+=1"
#     done
#     echo "clients part 1 start"
#     # clients part 2
#     i=6
#     while((i<=10))
#     do
#         CUDA_VISIBLE_DEVICES=3 python client_cifar10.py --DevNum=$i --n_data=2000 --method='iid' --dataset='CIFAR10' --TotalDevNum=10 --seed=42 --defense='none' --attack='gs' &
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
#     CUDA_VISIBLE_DEVICES=3 python server_cifar10.py --minfit=10 --mineval=10 --minavl=10 --num_rounds=50 --train_lr=0.01 --method='iid' --dataset='CIFAR10' --seed=42 --cnt=$j --defense='dp' &
#     sleep 5
#     # clients part 1
#     i=1
#     while((i<=5))
#     do
#         CUDA_VISIBLE_DEVICES=3 python client_cifar10.py --DevNum=$i --n_data=2000 --method='iid' --dataset='CIFAR10' --TotalDevNum=10 --seed=42 --defense='dp' --scale=1e-3 --attack='gs' &
#         echo "Client $i is running"
#         sleep 5
#         let "i+=1"
#     done
#     echo "clients part 1 start"
#     # clients part 2
#     i=6
#     while((i<=10))
#     do
#         CUDA_VISIBLE_DEVICES=3 python client_cifar10.py --DevNum=$i --n_data=2000 --method='iid' --dataset='CIFAR10' --TotalDevNum=10 --seed=42 --defense='dp' --scale=1e-3 --attack='gs' &
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
#     CUDA_VISIBLE_DEVICES=3 python server_cifar10.py --minfit=10 --mineval=10 --minavl=10 --num_rounds=50 --train_lr=0.01 --method='iid' --dataset='CIFAR10' --seed=42 --cnt=$j --defense='cp' &
#     sleep 5
#     # clients part 1
#     i=1
#     while((i<=5))
#     do
#         CUDA_VISIBLE_DEVICES=3 python client_cifar10.py --DevNum=$i --n_data=2000 --method='iid' --dataset='CIFAR10' --TotalDevNum=10 --seed=42 --defense='cp' --percent_num=90 --attack='gs' &
#         echo "Client $i is running"
#         sleep 5
#         let "i+=1"
#     done
#     echo "clients part 1 start"
#     # clients part 2
#     i=6
#     while((i<=10))
#     do
#         CUDA_VISIBLE_DEVICES=3 python client_cifar10.py --DevNum=$i --n_data=2000 --method='iid' --dataset='CIFAR10' --TotalDevNum=10 --seed=42 --defense='cp' --percent_num=90 --attack='gs' &
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
#     CUDA_VISIBLE_DEVICES=3 python server_cifar10.py --minfit=10 --mineval=10 --minavl=10 --num_rounds=50 --train_lr=0.01 --method='iid' --dataset='CIFAR10' --seed=42 --cnt=$j --defense='dcs' &
#     sleep 5
#     # clients part 1
#     i=1
#     while((i<=5))
#     do
#         CUDA_VISIBLE_DEVICES=3 python client_cifar10.py --mixup --DevNum=$i --n_data=2000 --method='iid' --dataset='CIFAR10' --TotalDevNum=10 --seed=42 --defense='dcs' --dcs_iter=1000 --dcs_lr=0.1 --lambda_xsim=0.01 --lambda_zsim=0.01 --lambda_y=0.7 --xsim_thr=500. --epsilon=0.01 --attack='gs' &
#         echo "Client $i is running"
#         sleep 5
#         let "i+=1"
#     done
#     echo "clients part 1 start"
#     # clients part 2
#     i=6
#     while((i<=10))
#     do
#         CUDA_VISIBLE_DEVICES=3 python client_cifar10.py --mixup --DevNum=$i --n_data=2000 --method='iid' --dataset='CIFAR10' --TotalDevNum=10 --seed=42 --defense='dcs' --dcs_iter=1000 --dcs_lr=0.1 --lambda_xsim=0.01 --lambda_zsim=0.01 --lambda_y=0.7 --xsim_thr=500. --epsilon=0.01 --attack='gs' &
#         echo "Client $i is running"
#         sleep 5
#         let "i+=1"
#     done
#     echo "clients part 2 start"
#     # wait for current cnt to finish
#     wait
# done
# echo "all workers done"

for j in 1 2 # 3 4 5
do
    echo "Job $j is running"
    # server
    CUDA_VISIBLE_DEVICES=3 python server_cifar10.py --minfit=10 --mineval=10 --minavl=10 --num_rounds=50 --train_lr=0.01 --method='iid' --dataset='CIFAR10' --seed=42 --cnt=$j --defense='soteria' &
    sleep 5
    # clients part 1
    i=1
    while((i<=5))
    do
        CUDA_VISIBLE_DEVICES=3 python client_cifar10.py --DevNum=$i --n_data=2000 --method='iid' --dataset='CIFAR10' --TotalDevNum=10 --seed=42 --defense='soteria' --percent_num=50 --layer_num=32 --attack='gs' &
        echo "Client $i is running"
        sleep 5
        let "i+=1"
    done
    echo "clients part 1 start"
    # clients part 2
    i=6
    while((i<=10))
    do
        CUDA_VISIBLE_DEVICES=3 python client_cifar10.py --DevNum=$i --n_data=2000 --method='iid' --dataset='CIFAR10' --TotalDevNum=10 --seed=42 --defense='soteria' --percent_num=50 --layer_num=32 --attack='gs' &
        echo "Client $i is running"
        sleep 5
        let "i+=1"
    done
    echo "clients part 2 start"
    # wait for current cnt to finish
    wait
done
echo "all workers done"


# for j in 1 2 # 3 4 5
# do
#     echo "Job $j is running"
#     # server
#     CUDA_VISIBLE_DEVICES=3 python server_cifar10.py --minfit=10 --mineval=10 --minavl=10 --num_rounds=50 --train_lr=0.01 --method='iid' --dataset='CIFAR10' --seed=42 --cnt=$j --defense='ats' &
#     sleep 5
#     # clients part 1
#     i=1
#     while((i<=5))
#     do
#         CUDA_VISIBLE_DEVICES=3 python client_cifar10.py --DevNum=$i --n_data=2000 --method='iid' --dataset='CIFAR10' --TotalDevNum=10 --seed=42 --defense='ats' --aug_list='21-13-3+7-4-15+1-2-5-8-10' --attack='gs' &
#         echo "Client $i is running"
#         sleep 5
#         let "i+=1"
#     done
#     echo "clients part 1 start"
#     # clients part 2
#     i=6
#     while((i<=10))
#     do
#         CUDA_VISIBLE_DEVICES=3 python client_cifar10.py --DevNum=$i --n_data=2000 --method='iid' --dataset='CIFAR10' --TotalDevNum=10 --seed=42 --defense='ats' --aug_list='21-13-3+7-4-15+1-2-5-8-10' --attack='gs' &
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
#     CUDA_VISIBLE_DEVICES=3 python server_cifar10.py --minfit=10 --mineval=10 --minavl=10 --num_rounds=50 --train_lr=0.01 --method='iid' --dataset='CIFAR10' --seed=42 --cnt=$j --defense='precode' --precode_size=32 --beta=1e-4 &
#     sleep 5
#     # clients part 1
#     i=1
#     while((i<=5))
#     do
#         CUDA_VISIBLE_DEVICES=3 python client_cifar10.py --DevNum=$i --n_data=2000 --method='iid' --dataset='CIFAR10' --TotalDevNum=10 --seed=42 --defense='precode' --precode_size=32 --beta=1e-4 --attack='gs' &
#         echo "Client $i is running"
#         sleep 5
#         let "i+=1"
#     done
#     echo "clients part 1 start"
#     # clients part 2
#     i=6
#     while((i<=10))
#     do
#         CUDA_VISIBLE_DEVICES=3 python client_cifar10.py --DevNum=$i --n_data=2000 --method='iid' --dataset='CIFAR10' --TotalDevNum=10 --seed=42 --defense='precode' --precode_size=32 --beta=1e-4 --attack='gs' &
#         echo "Client $i is running"
#         sleep 5
#         let "i+=1"
#     done
#     echo "clients part 2 start"
#     # wait for current cnt to finish
#     wait
# done
# echo "all workers done"



