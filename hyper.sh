# for ep in 10 15 20 25 30
#     do
#         python train.py --BATCH_SIZE $bs
#     done

for lr in 0.0001 0.0005 0.001 0.005 0.01
    do
        python train.py --LEARNING_RATE $lr
    done


for bs in 64 128 256 512 1024
    do
        python train.py --BATCH_SIZE $bs
    done

