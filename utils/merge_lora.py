"""
Usage:
python merge_lora.py --base_model_path [BASE-MODEL-PATH] --lora_path [LORA-PATH]
"""
import argparse
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import RobertaConfig, RobertaTokenizer, AutoModelForMaskedLM


def merge_lora(base_model_name, lora_path):

    # if is_roberta:
    #     base_model = AutoModelForCausalLM.from_pretrained(base_model_name, is_decoder=True) #如果要用roberta就要加上is_decoder=True
    #     peft_model = PeftModel.from_pretrained(base_model, lora_path)
    #     tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)

    #     model = peft_model.merge_and_unload()
    #     target_model_path = lora_path.replace("checkpoint", "full")
    #     model.save_pretrained(target_model_path)
    #     tokenizer.save_pretrained(target_model_path)


    # else: 
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, is_decoder=True)
    peft_model = PeftModel.from_pretrained(base_model, lora_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)

    model = peft_model.merge_and_unload()
    target_model_path = lora_path.replace("checkpoint", "full")
    model.save_pretrained(target_model_path)
    tokenizer.save_pretrained(target_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, default=None)
    parser.add_argument("--lora_path", type=str, required=True)
    # parser.add_argument("--is_roberta", type=bool, default=False)

    args = parser.parse_args()

    merge_lora(args.base_model_path, args.lora_path)