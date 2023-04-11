for do_use in 0 1
do
    python train.py \
    --model 'CNN' \
    --learning_rate 0.0005 \
    --epoch_num 20 \
    --batch_size 128 \
    --random_embedding $do_use
done