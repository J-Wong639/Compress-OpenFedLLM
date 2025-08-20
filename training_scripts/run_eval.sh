max_steps=1
num_rounds=1
batch_size=16
gradient_accumulation_steps=1
seq_length=512
num_clients=50
sample_clients=1
lora_r=32
lora_alpha=64   # twice of lora_r
lr=5e-5

# local_data_dir=""       # you may uncomment this line if your data is stored locally and include it in the python command
dataset_name="FinGPT/fingpt-sentiment-train"
dataset_sample=10000
model_name_or_path="meta-llama/Llama-2-7b-hf"
output_dir=./output

gpu=5
fed_alg="fedavgm"
is_quantized=False
quantize_level=32
test_dataset="fiqa"
test_batch_size=16
merge_lora=True
save_model_freq=1


# main_sft.py
# test2.py

CUDA_VISIBLE_DEVICES=$gpu python eval.py \
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
 --use_peft \
 --load_in_8bit \
 --output_dir $output_dir \
 --template "alpaca" \
 --is_quantized $is_quantized \
 --quantize_level $quantize_level \
 --test_dataset $test_dataset \
 --test_batch_size $test_batch_size \
 --merge_lora $merge_lora \
 --save_model_freq $save_model_freq \
