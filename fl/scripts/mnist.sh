# for j in 1 2 3 4 5
# do
#     echo "Job $j is running"
#     # server
#     CUDA_VISIBLE_DEVICES=1 python server.py --minfit=10 --mineval=10 --minavl=10 --num_rounds=200 --method='iid' --seed=42 --cnt=$j --defense='none' --local_epochs=1 &
#     sleep 5
#     # clients part 1
#     i=1
#     while((i<=5))
#     do
#         CUDA_VISIBLE_DEVICES=1 python client.py --DevNum=$i --method='iid' --TotalDevNum=10 --seed=42 --defense='none' --attack='gs' &
#         echo "Client $i is running"
#         sleep 5
#         let "i+=1"
#     done
#     echo "clients part 1 start"
#     # clients part 2
#     i=6
#     while((i<=10))
#     do
#         CUDA_VISIBLE_DEVICES=1 python client.py --DevNum=$i --method='iid' --TotalDevNum=10 --seed=42 --defense='none' --attack='gs' &
#         echo "Client $i is running"
#         sleep 5
#         let "i+=1"
#     done
#     echo "clients part 2 start"
#     # wait for current cnt to finish
#     wait
# done
# echo "all workers done"


# for j in 1 2 3 4 5
# do
#     echo "Job $j is running"
#     # server
#     CUDA_VISIBLE_DEVICES=1 python server.py --minfit=10 --mineval=10 --minavl=10 --num_rounds=200 --method='iid' --seed=42 --cnt=$j --defense='dp'  --local_epochs=1 &
#     sleep 5
#     # clients part 1
#     i=1
#     while((i<=5))
#     do
#         CUDA_VISIBLE_DEVICES=1 python client.py --DevNum=$i --method='iid' --TotalDevNum=10 --seed=42 --defense='dp' --scale=1e-2 --attack='gs' &
#         echo "Client $i is running"
#         sleep 5
#         let "i+=1"
#     done
#     echo "clients part 1 start"
#     # clients part 2
#     i=6
#     while((i<=10))
#     do
#         CUDA_VISIBLE_DEVICES=1 python client.py --DevNum=$i --method='iid' --TotalDevNum=10 --seed=42 --defense='dp' --scale=1e-2 --attack='gs' &
#         echo "Client $i is running"
#         sleep 5
#         let "i+=1"
#     done
#     echo "clients part 2 start"
#     # wait for current cnt to finish
#     wait
# done
# echo "all workers done"


# for j in 1 2 3 4 5
# do
#     echo "Job $j is running"
#     # server
#     CUDA_VISIBLE_DEVICES=1 python server.py --minfit=10 --mineval=10 --minavl=10 --num_rounds=200 --method='iid' --seed=42 --cnt=$j --defense='cp' --local_epochs=1 &
#     sleep 5
#     # clients part 1
#     i=1
#     while((i<=5))
#     do
#         CUDA_VISIBLE_DEVICES=1 python client.py --DevNum=$i --method='iid' --TotalDevNum=10 --seed=42 --defense='cp' --percent_num=70 --attack='gs' &
#         echo "Client $i is running"
#         sleep 5
#         let "i+=1"
#     done
#     echo "clients part 1 start"
#     # clients part 2
#     i=6
#     while((i<=10))
#     do
#         CUDA_VISIBLE_DEVICES=1 python client.py --DevNum=$i --method='iid' --TotalDevNum=10 --seed=42 --defense='cp' --percent_num=70 --attack='gs' &
#         echo "Client $i is running"
#         sleep 5
#         let "i+=1"
#     done
#     echo "clients part 2 start"
#     # wait for current cnt to finish
#     wait
# done
# echo "all workers done"


# for j in 1 2 3 4 5
# do
#     echo "Job $j is running"
#     # server
#     CUDA_VISIBLE_DEVICES=1 python server.py --minfit=10 --mineval=10 --minavl=10 --num_rounds=200 --method='iid' --seed=42 --cnt=$j --defense='dcs' --local_epochs=1 &
#     sleep 5
#     # clients part 1
#     i=1
#     while((i<=5))
#     do
#         CUDA_VISIBLE_DEVICES=1 python client.py --mixup --DevNum=$i --startpoint='none' --method='iid' --TotalDevNum=10 --seed=42 --defense='dcs' --percent_num=70 --dcs_iter=1000 --dcs_lr=0.1 --lambda_xsim=0.01 --lambda_zsim=0.01 --lambda_y=0.7 --xsim_thr=300. --epsilon=0.01 --attack='gs' &
#         echo "Client $i is running"
#         sleep 5
#         let "i+=1"
#     done
#     echo "clients part 1 start"
#     # clients part 2
#     i=6
#     while((i<=10))
#     do
#         CUDA_VISIBLE_DEVICES=1 python client.py --mixup --DevNum=$i --startpoint='none' --method='iid' --TotalDevNum=10 --seed=42 --defense='dcs' --percent_num=70 --dcs_iter=1000 --dcs_lr=0.1 --lambda_xsim=0.01 --lambda_zsim=0.01 --lambda_y=0.7 --xsim_thr=300. --epsilon=0.01 --attack='gs' &
#         echo "Client $i is running"
#         sleep 5
#         let "i+=1"
#     done
#     echo "clients part 2 start"
#     # wait for current cnt to finish
#     wait
# done
# echo "all workers done"


# for j in 1 2 3 4 5
# do
#     echo "Job $j is running"
#     # server
#     CUDA_VISIBLE_DEVICES=1 python server.py --minfit=10 --mineval=10 --minavl=10 --num_rounds=200 --method='iid' --seed=42 --cnt=$j --defense='soteria' --local_epochs=2 &
#     sleep 5
#     # clients part 1
#     i=1
#     while((i<=5))
#     do
#         CUDA_VISIBLE_DEVICES=1 python client.py --DevNum=$i --method='iid' --TotalDevNum=10 --seed=42 --defense='soteria' --percent_num=20 --layer_num=6 --attack='gs' &
#         echo "Client $i is running"
#         sleep 5
#         let "i+=1"
#     done
#     echo "clients part 1 start"
#     # clients part 2
#     i=6
#     while((i<=10))
#     do
#         CUDA_VISIBLE_DEVICES=1 python client.py --DevNum=$i --method='iid' --TotalDevNum=10 --seed=42 --defense='soteria' --percent_num=20 --layer_num=6 --attack='gs' &
#         echo "Client $i is running"
#         sleep 5
#         let "i+=1"
#     done
#     echo "clients part 2 start"
#     # wait for current cnt to finish
#     wait
# done
# echo "all workers done"

# for j in 1 2 3 4 5
# do
#     echo "Job $j is running"
#     # server
#     CUDA_VISIBLE_DEVICES=1 python server.py --minfit=10 --mineval=10 --minavl=10 --num_rounds=200 --method='iid' --seed=42 --cnt=$j --defense='dcs_cp' --local_epochs=1 &
#     sleep 5
#     # clients part 1
#     i=1
#     while((i<=5))
#     do
#         CUDA_VISIBLE_DEVICES=1 python client.py --mixup --DevNum=$i --startpoint='none' --method='iid' --TotalDevNum=10 --seed=42 --defense='dcs_cp' --percent_num=70 --dcs_iter=1000 --dcs_lr=0.1 --lambda_xsim=0.01 --lambda_zsim=0.01 --lambda_y=0.7 --xsim_thr=300. --epsilon=0.01 --attack='gs' &
#         echo "Client $i is running"
#         sleep 5
#         let "i+=1"
#     done
#     echo "clients part 1 start"
#     # clients part 2
#     i=6
#     while((i<=10))
#     do
#         CUDA_VISIBLE_DEVICES=1 python client.py --mixup --DevNum=$i --startpoint='none' --method='iid' --TotalDevNum=10 --seed=42 --defense='dcs_cp' --percent_num=70 --dcs_iter=1000 --dcs_lr=0.1 --lambda_xsim=0.01 --lambda_zsim=0.01 --lambda_y=0.7 --xsim_thr=300. --epsilon=0.01 --attack='gs' &
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
#     CUDA_VISIBLE_DEVICES=0 python server.py --output_dir='./logs/lambda_y/1.0' --minfit=10 --mineval=10 --minavl=10 --num_rounds=200 --method='iid' --seed=42 --cnt=$j --defense='dcs' --local_epochs=1 &
#     sleep 5
#     # clients part 1
#     i=1
#     while((i<=5))
#     do
#         CUDA_VISIBLE_DEVICES=0 python client.py --output_dir='./logs/lambda_y/1.0' --lambda_y=1.0 --mixup --DevNum=$i --startpoint='none' --method='iid' --TotalDevNum=10 --seed=42 --defense='dcs' --percent_num=70 --dcs_iter=1000 --dcs_lr=0.1 --lambda_xsim=0.01 --lambda_zsim=0.01 --xsim_thr=300. --epsilon=0.01 --attack='gs' &
#         echo "Client $i is running"
#         sleep 5
#         let "i+=1"
#     done
#     echo "clients part 1 start"
#     # clients part 2
#     i=6
#     while((i<=10))
#     do
#         CUDA_VISIBLE_DEVICES=0 python client.py --output_dir='./logs/lambda_y/1.0' --lambda_y=1.0 --mixup --DevNum=$i --startpoint='none' --method='iid' --TotalDevNum=10 --seed=42 --defense='dcs' --percent_num=70 --dcs_iter=1000 --dcs_lr=0.1 --lambda_xsim=0.01 --lambda_zsim=0.01 --xsim_thr=300. --epsilon=0.01 --attack='gs' &
#         echo "Client $i is running"
#         sleep 5
#         let "i+=1"
#     done
#     echo "clients part 2 start"
#     # wait for current cnt to finish
#     wait
# done
# echo "all workers done"




