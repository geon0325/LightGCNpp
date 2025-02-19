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

dataset=lastfm
alpha=0.6
beta=-0.1
gamma=0.0

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
