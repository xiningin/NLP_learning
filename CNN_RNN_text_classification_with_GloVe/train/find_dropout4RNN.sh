for model in 'RNN' 'LSTM' 'Bi-LSTM'
do
    for dropout in 0.1 0.2 0.3 0.4 0.5
    do
        python train.py \
            --model $model \
            --learning_rate 0.0001 \
            --epoch_num 70 \
            --batch_size 256 \
            --dropout $dropout
    done
done