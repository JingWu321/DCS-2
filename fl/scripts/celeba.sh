for j in 1 2 # 3 4 5
do
    echo "Job $j is running"
    # server
    CUDA_VISIBLE_DEVICES=1 python server.py --minfit=10 --mineval=10 --minavl=10 --num_rounds=100 --train_lr=0.001 --method='iid' --dataset='CelebA' --seed=42 --cnt=$j --defense='none' &
    sleep 5
    # clients part 1
    i=1
    while((i<=5))
    do
        CUDA_VISIBLE_DEVICES=1 python client.py --n_data=500 --DevNum=$i --method='iid' --dataset='CelebA' --TotalDevNum=10 --seed=42 --defense='none' --attack='ggl' &
        echo "Client $i is running"
        sleep 5
        let "i+=1"
    done
    echo "clients part 1 start"
    # clients part 2
    i=6
    while((i<=10))
    do
        CUDA_VISIBLE_DEVICES=0 python client.py --n_data=500 --DevNum=$i --method='iid' --dataset='CelebA' --TotalDevNum=10 --seed=42 --defense='none' --attack='ggl' &
        echo "Client $i is running"
        sleep 5
        let "i+=1"
    done
    echo "clients part 2 start"
    # wait for current cnt to finish
    wait
done
echo "all workers done"


for j in 1 2 # 3 4 5
do
    echo "Job $j is running"
    # server
    CUDA_VISIBLE_DEVICES=1 python server.py --minfit=10 --mineval=10 --minavl=10 --num_rounds=100 --train_lr=0.001 --method='iid' --dataset='CelebA' --seed=42 --cnt=$j --defense='dp' &
    sleep 5
    # clients part 1
    i=1
    while((i<=5))
    do
        CUDA_VISIBLE_DEVICES=1 python client.py --n_data=500 --DevNum=$i --method='iid' --dataset='CelebA' --TotalDevNum=10 --seed=42 --defense='dp' --scale=1e-3 --attack='ggl' &
        echo "Client $i is running"
        sleep 5
        let "i+=1"
    done
    echo "clients part 1 start"
    # clients part 2
    i=6
    while((i<=10))
    do
        CUDA_VISIBLE_DEVICES=0 python client.py --n_data=500 --DevNum=$i --method='iid' --dataset='CelebA' --TotalDevNum=10 --seed=42 --defense='dp' --scale=1e-3 --attack='ggl' &
        echo "Client $i is running"
        sleep 5
        let "i+=1"
    done
    echo "clients part 2 start"
    # wait for current cnt to finish
    wait
done
echo "all workers done"

for j in 1 2 # 3 4 5
do
    echo "Job $j is running"
    # server
    CUDA_VISIBLE_DEVICES=1 python server.py --minfit=10 --mineval=10 --minavl=10 --num_rounds=100 --train_lr=0.001 --method='iid' --dataset='CelebA' --seed=42 --cnt=$j --defense='soteria' &
    sleep 5
    # clients part 1
    i=1
    while((i<=5))
    do
        CUDA_VISIBLE_DEVICES=1 python client.py --n_data=500 --DevNum=$i --method='iid' --dataset='CelebA' --TotalDevNum=10 --seed=42 --defense='soteria' --percent_num=70 --layer_num=32 --attack='ggl' &
        echo "Client $i is running"
        sleep 5
        let "i+=1"
    done
    echo "clients part 1 start"
    # clients part 2
    i=6
    while((i<=10))
    do
        CUDA_VISIBLE_DEVICES=0 python client.py --n_data=500 --DevNum=$i --method='iid' --dataset='CelebA' --TotalDevNum=10 --seed=42 --defense='soteria' --percent_num=70 --layer_num=32 --attack='ggl' &
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
#     CUDA_VISIBLE_DEVICES=1 python server.py --minfit=10 --mineval=10 --minavl=10 --num_rounds=100 --train_lr=0.001 --method='iid' --dataset='CelebA' --seed=42 --cnt=$j --defense='cp' &
#     sleep 5
#     # clients part 1
#     i=1
#     while((i<=5))
#     do
#         CUDA_VISIBLE_DEVICES=1 python client.py --n_data=500 --DevNum=$i --method='iid' --dataset='CelebA' --TotalDevNum=10 --seed=42 --defense='cp' --percent_num=70 --attack='ggl' &
#         echo "Client $i is running"
#         sleep 5
#         let "i+=1"
#     done
#     echo "clients part 1 start"
#     # clients part 2
#     i=6
#     while((i<=10))
#     do
#         CUDA_VISIBLE_DEVICES=0 python client.py --n_data=500 --DevNum=$i --method='iid' --dataset='CelebA' --TotalDevNum=10 --seed=42 --defense='cp' --percent_num=70 --attack='ggl' &
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
#     CUDA_VISIBLE_DEVICES=1 python server.py --minfit=10 --mineval=10 --minavl=10 --num_rounds=100 --train_lr=0.001 --method='iid' --dataset='CelebA' --seed=42 --cnt=$j --defense='dcs' &
#     sleep 5
#     # clients part 1
#     i=1
#     while((i<=5))
#     do
#         CUDA_VISIBLE_DEVICES=1 python client.py --n_data=500 --DevNum=$i --method='iid' --dataset='CelebA' --TotalDevNum=10 --seed=42 --defense='dcs' --dcs_iter=1000 --dcs_lr=0.1 --lambda_xsim=0.01 --lambda_zsim=0.01 --lambda_y=0.7 --xsim_thr=300. --epsilon=0.01 --attack='ggl' &
#         echo "Client $i is running"
#         sleep 5
#         let "i+=1"
#     done
#     echo "clients part 1 start"
#     # clients part 2
#     i=6
#     while((i<=10))
#     do
#         CUDA_VISIBLE_DEVICES=0 python client.py --n_data=500 --DevNum=$i --method='iid' --dataset='CelebA' --TotalDevNum=10 --seed=42 --defense='dcs' --dcs_iter=1000 --dcs_lr=0.1 --lambda_xsim=0.01 --lambda_zsim=0.01 --lambda_y=0.7 --xsim_thr=300. --epsilon=0.01 --attack='ggl' &
#         echo "Client $i is running"
#         sleep 5
#         let "i+=1"
#     done
#     echo "clients part 2 start"
#     # wait for current cnt to finish
#     wait
# done
# echo "all workers done"

