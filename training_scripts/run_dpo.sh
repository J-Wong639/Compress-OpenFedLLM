max_steps=10 #这个是iterations
# max_steps=1
num_rounds=200
batch_size=16
gradient_accumulation_steps=1
seq_length=512
num_clients=5
sample_clients=2
# sample_clients=1
lora_r=32
lora_alpha=64
lr=5e-4

# local_data_dir=""       # you may uncomment this line if your data is stored locally and include it in the python command
# dataset_name="Anthropic/hh-rlhf"
dataset_name="HuggingFaceH4/ultrafeedback_binarized"
dataset_sample=20000
model_name_or_path="ehartford/Wizard-Vicuna-7B-Uncensored"
output_dir=./output



gpu=4
fed_alg="fedyogi"


test_dataset="fpb"
test_batch_size=16
is_quantized=False
quantize_level=32


is_topk=False
topk_compression_ratio=0.1





save_model_freq=201
test_model_freq=200
use_peft=True
special_sign="fedyogi_dpo_clientPQ_1e-3_1e-4_ultra"
subspace_change_freq=10
is_change_subspace=False
subspace_rank=8



CUDA_VISIBLE_DEVICES=$gpu python main_dpo.py \
 --learning_rate $lr \
 --model_name_or_path $model_name_or_path \
 --dataset_name $dataset_name \
 --dataset_sample $dataset_sample \
 --fed_alg $fed_alg \
 --num_clients $num_clients \
 --sample_clients $sample_clients \
 --max_steps $max_steps \
 --num_rounds $num_rounds \
 --batch_size $batch_size \
 --gradient_accumulation_steps $gradient_accumulation_steps \
 --seq_length $seq_length \
 --peft_lora_r $lora_r \
 --peft_lora_alpha $lora_alpha \
 --use_peft $use_peft \
 --load_in_8bit \
 --output_dir $output_dir \
 --template "vicuna_v1.1" \
 --is_quantized $is_quantized \
 --quantize_level $quantize_level \
 --test_dataset $test_dataset \
 --test_batch_size $test_batch_size \
 --save_model_freq $save_model_freq \
 --test_model_freq $test_model_freq \
 --special_sign $special_sign \
 --subspace_change_freq $subspace_change_freq \
 --is_change_subspace $is_change_subspace \
 --subspace_rank $subspace_rank \
 --is_topk $is_topk \
 --topk_compression_ratio $topk_compression_ratio \