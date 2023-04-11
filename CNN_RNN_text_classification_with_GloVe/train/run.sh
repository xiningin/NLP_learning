for model in 'CNN' 'RNN' 'LSTM' 'Bi-LSTM'
do
    if (($model == 'CNN'))
    then 
        for lr in 3e-4 4e-4 5e-4 6e-4 7e-4 8e-4
        do
            python train.py \
                --model $model \
                --learning_rate $lr \
                --epoch_num 20 \
                --batch_size 64
        done
    elif (($model == 'RNN'))
    then
        for lr in 3e-4 4e-4 5e-4 6e-4 7e-4 8e-4
        do  
            python train.py \
                --model $model \
                --learning_rate $lr \
                --epoch_num 50 \
                --batch_size 64
        done
    elif (($model == 'LSTM'))
    then
        for lr in 3e-4 4e-4 5e-4 6e-4 7e-4 8e-4
        do  
            python train.py \
                --model $model \
                --learning_rate $lr \
                --epoch_num 50 \
                --batch_size 64
        done 
    elif (($model == 'Bi-LSTM'))
    then
        for lr in 3e-4 4e-4 5e-4 6e-4 7e-4 8e-4
        do  
            python train.py \
                --model $model \
                --learning_rate $lr \
                --epoch_num 50 \
                --batch_size 64
        done 
    fi
done