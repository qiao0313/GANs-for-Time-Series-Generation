n_epochs=100
batch_size=64
model_size=64
input_dim=2
hidden_dim=16
dropout_rate=0.25
lr_g=0.0002
lr_d=0.0002
b1_g=0.5
b1_d=0.5
b2_g=0.999
b2_d=0.999
sample_interval=1000

python models/model_train.py --n_epochs $n_epochs --batch_size $batch_size  --model_size $model_size --input_dim $input_dim --hidden_dim $hidden_dim --dropout_rate $dropout_rate --lr_g $lr_g --b1_g $b1_g --b2_g $b2_g --lr_d $lr_d --b1_d $b1_d --b2_d $b2_d --sample_interval $sample_interval
