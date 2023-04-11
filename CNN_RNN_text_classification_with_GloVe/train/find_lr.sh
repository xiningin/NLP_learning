for model in 'CNN' 'RNN' 'LSTM' 'Bi-LSTM'
do
    for lr in 1e-7 5e-7 1e-6 5e-6 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2 5e-2 1e-1
    do
        python train.py \
            --model $model \
            --learning_rate $lr \
            --epoch_num 20 \
            --batch_size 64
    done
done