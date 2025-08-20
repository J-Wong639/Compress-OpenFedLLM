from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm
# from finetune.system_prompts import SYSTEM_PROMPT


from datasets import load_dataset


def main():
    model_name = "FacebookAI/roberta-base"
    sft_ckpt_path = "/mnt/sdb/huangjunlin/Compress-OpenFedLLM/output/rte_2490_fedavg_c1s1_i5_b32a1_l512_r16a32_qlevel30_topk_ratio0.1_fpb_center_fedavg_20250801154443/full-100"
    dataset_name= "SetFit/rte"

    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        sft_ckpt_path,
        torch_dtype="auto",
        device_map="auto"
    )

    inference_data = load_dataset(dataset_name, split="validation")


    inference_result = []
    for example in tqdm(inference_data):
        # messages = [
        #     {
        #         "role": "system", "content": SYSTEM_PROMPT
        #     },
        #     {
        #         "role": "user", "content": example["prompt"]
        #     }
        # ]

        # text = tokenizer.apply_chat_template(
        #     messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        # )

        # import pdb; pdb.set_trace()
        instruction= f"Sentence 1: {example['text1']} \n Sentence 2: {example['text2']}"


        alpaca_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{instruction} 
### Response:
"""
        text=alpaca_template.format(instruction=instruction)


        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=1024,
            temperature=0.9,
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

        content = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        # print(example["prompt_id"])
        print(f"🦍 {text}")
        print(f"🤖 {content}")
        print("-" * 100)
    #     inference_result.append({
    #         # "prompt_id": example["prompt_id"],
    #         "prompt": content,
    #         "raw_prompt": example["prompt"],
    #     })

    # with open("eval_prompts_v3_v0.2.json", "w", encoding="utf-8") as file:
    #     json.dump(inference_result, file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()