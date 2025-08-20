max_steps=10
num_rounds=200
# num_rounds=1
# batch_size=16
batch_size=32 #RTE
gradient_accumulation_steps=1
seq_length=512
num_clients=5
sample_clients=5
# lora_r=32
lora_r=16 #RTE
# lora_alpha=64   # twice of lora_r
lora_alpha=32 #RTE
lr=2e-4
# lr=5e-4
# lr=1e-5


# # local_data_dir=""       # you may uncomment this line if your data is stored locally and include it in the python command
# dataset_name="FinGPT/fingpt-sentiment-train"
# dataset_sample=10000



# dataset_name="vicgalle/alpaca-gpt4" #用远程数据集的时候
# dataset_name="local_alpaca-gpt4"  #用本地数据集的时候
# local_data_dir="/data2/share/alpaca-gpt4/" #用本地数据集的时候

# dataset_sample=20000

dataset_name="SetFit/rte" #用远程数据集的时候
# dataset_name="local_RTE"  #用本地数据集的时候
# local_data_dir="/data2/share/" #用本地数据集的时候
dataset_sample=2490




# dataset_name="SetFit/qnli" #用远程数据集的时候
# dataset_sample=105000


# dataset_name="COLA"
# local_data_dir="/mnt/sdb/huangjunlin/Compress-OpenFedLLM/"
# dataset_sample=8550



# dataset_name="20NG"
# dataset_sample=11310


# dataset_name="COLA"
# dataset_sample=8550


# dataset_name="TIGER-Lab/MathInstruct"
# dataset_sample=20000
# dataset_name="medalpaca/medical_meadow_medical_flashcards"
# dataset_sample=20000






# dataset_name="yahma/alpaca-cleaned"
# dataset_sample=20000

# dataset_name="WizardLM/WizardLM_evol_instruct_70k"
# dataset_sample=20000


# dataset_name="tatsu-lab/alpaca"
# dataset_sample=20000


# dataset_name="gbharti/finance-alpaca"
# dataset_sample=20000







# dataset_name="lighteval/MATH"
# dataset_sample=20000


# dataset_name="meta-math/MetaMathQA"
# dataset_sample=20000







# model_name_or_path="meta-llama/Llama-2-7b-hf"
model_name_or_path="FacebookAI/roberta-base"
# model_name_or_path="Qwen/Qwen-1_8B-Chat"
# model_name_or_path="meta-llama/Meta-Llama-3-8B"
# model_name_or_path="THUDM/chatglm-6b"



# local_model_name_or_path="/data2/share/roberta-base" #不用时设置为None




output_dir=./output

gpu=0
fed_alg="fedavg"

test_dataset="fpb"
test_batch_size=8


is_quantized=False
quantize_level=30

is_topk=False
topk_compression_ratio=0.1 #意思是只保留0.1的参数





save_model_freq=100
# save_model_freq=50
test_model_freq=301
use_peft=True
# special_sign="fedadam_sft_clientPQ_1e-3_1e-4_alpaca_topk1e-1"
# special_sign="fedadam_sft_clientPQ_1e-3_1e-4_MetaMathQA_sub2_savemodel"
# special_sign="fedadam_sft_clientPQ_1e-3_1e-4_fingpt_svdfed_sub16_savemodel"
# special_sign="fedadam_sft_clientPQ_1e-3_1e-4_meta-mathqa_qwen_sub32_savemodel"
# special_sign="training_acc_round200_maxsteps10_fedadam_sft_clientPQ_1e-3_1e-4_RTE_roberta_batchsize32_sub32_savemodel"
special_sign="center_fedavg"


subspace_change_freq=10
is_change_subspace=False
subspace_rank=16



CUDA_VISIBLE_DEVICES=$gpu python main_sft.py \
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
 --template "alpaca" \
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
#  --local_model_name_or_path $local_model_name_or_path \
#  --local_data_dir $local_data_dir \
