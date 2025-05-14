#!/bin/bash

gpu=0
model_name=LightGCNpp
model_type=graph

item_ranking=10,20,40
embedding_size=64
epoch=200
batch_size=2048
learning_rate=0.001
reg_lambda=0.0001
n_layer=2

# Take dataset as argument
dataset=$1

# Set alpha, beta, gamma based on dataset
case $dataset in
    lastfm)
        alpha=0.6; beta=-0.1; gamma=0.0;;
    citeulike)
        alpha=0.5; beta=-0.1; gamma=0.4;;
    movielens-1m)
        alpha=0.4; beta=0.1; gamma=0.0;;
    gowalla)
        alpha=0.6; beta=-0.1; gamma=0.2;;
    yelp)
        alpha=0.6; beta=-0.1; gamma=0.0;;
    amazon-sports)
        alpha=0.6; beta=-0.1; gamma=0.0;;
    amazon-beauty)
        alpha=0.5; beta=0.0; gamma=0.2;;
    amazon-book)
        alpha=0.6; beta=-0.1; gamma=0.2;;
    movielens-10m)
        alpha=0.6; beta=-0.1; gamma=0.0;;
    alibaba)
        alpha=0.6; beta=0.0; gamma=0.1;;
    *)
        echo "Unknown dataset: $dataset"; exit 1;;
esac

# Run
python main.py \
    --gpu $gpu \
    --dataset $dataset \
    --model_name $model_name \
    --model_type $model_type \
    --item_ranking $item_ranking \
    --embedding_size $embedding_size \
    --epoch $epoch \
    --batch_size $batch_size \
    --learning_rate $learning_rate \
    --reg_lambda $reg_lambda \
    --n_layer $n_layer \
    --alpha $alpha \
    --beta $beta \
    --gamma $gamma
