import os
import sys
import ast
import math
import json
import shutil
import logging
import argparse
from tqdm import tqdm
import tempfile
import importlib.util
from pathlib import Path
from functools import partial

import torch
from torch.utils.data import DataLoader

from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

from evaluations.tools import TextAbbreviationEvaluator
from finetune.system_prompts import SYSTEM_PROMPT

logger = get_logger(__name__)


# def load_map_func(version):
#     file_path = f"finetune/{version}.py"
#     project_root = os.path.abspath(os.path.join(file_path, "..", ".."))
#     if project_root not in sys.path:
#         sys.path.insert(0, project_root)
#     spec = importlib.util.spec_from_file_location("version_module", file_path)
#     module = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(module)
#     map_func = module.map_func
#     return map_func


def valid_collate_fn(batch, tokenizer, max_length=1024):
    generate_chats = []
    forward_input_ids = []
    forward_labels = []
    short_caption, tags_character, tags_artist = [], [], []
    for example in batch:
        user = example["user"]
        assistant = example["assistant"]

        generate_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        generate_text = tokenizer.apply_chat_template(
            generate_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        generate_chats.append(generate_text)

        forward_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant}
        ]
        forward_text = tokenizer.apply_chat_template(
            forward_messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
        forward_model_inputs = tokenizer(forward_text, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
        ids = forward_model_inputs["input_ids"][0]
        lbl = ids.clone()
        assistant_start = forward_text.index(assistant)
        prefix = tokenizer(forward_text[:assistant_start], return_tensors="pt", truncation=True, max_length=max_length)["input_ids"][0]
        lbl[:len(prefix)] = -100
        forward_input_ids.append(ids)
        forward_labels.append(lbl)
        
        # evaluate info
        character = []
        if isinstance(example["tags_character"], str) and example["tags_character"]:
            character = json.loads(example["tags_character"])
        elif isinstance(example["tags_character"], list):
            character = example["tags_character"]

        artist = []
        if isinstance(example["tags_artist"], str) and example["tags_character"]:
            artist = json.loads(example["tags_artist"])
        elif isinstance(example["tags_artist"], list):
            artist = example["tags_artist"]

        short_caption.append(example["gen_user_input"])
        tags_character.append(character)
        tags_artist.append(artist)
    
    generate_model_inputs = tokenizer(generate_chats, return_tensors="pt", padding=True, padding_side="left")
    
    return {
        **generate_model_inputs,
        "forward_input_ids": torch.stack(forward_input_ids),
        "forward_labels": torch.stack(forward_labels),
        "short_caption": short_caption,
        "tags_character": tags_character,
        "tags_artist": tags_artist,
    }


def train_collate_fn(batch, tokenizer, max_length=1024):
    input_ids = []
    labels = []
    for example in batch:
        user = example["user"]
        assistant = example["assistant"]



        # process_dataset
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant}
        ]
        
        chat_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )

        encoded = tokenizer(
            chat_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
    
        ids = encoded["input_ids"][0]
        lbl = ids.clone()
    
        # mask 掉 assistant 前的部分（system + user）
        assistant_start = chat_text.index(assistant)
        prefix = tokenizer(chat_text[:assistant_start], return_tensors="pt", truncation=True, max_length=max_length)["input_ids"][0]
        lbl[:len(prefix)] = -100

        input_ids.append(ids)
        labels.append(lbl)

    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels)
    }


def validation_loss(model, tokenizer, valid_dataloader, args, accelerator, weight_dtype, step, epoch, is_final_validation=False):
    total_loss = 0.0
    total_tokens = 0

    for batch in tqdm(valid_dataloader, desc=f"Evaluating"):
        forward_model_inputs = {
            "input_ids": batch["forward_input_ids"],
            "labels": batch["forward_labels"],
        }

        with torch.no_grad():
            if not is_final_validation:
                outputs = model(**forward_model_inputs)
                loss = outputs.loss
                total_loss += loss.item() * forward_model_inputs["input_ids"].size(0)
                total_tokens += forward_model_inputs["input_ids"].size(0)

    # === 评估并记录 ===
    evaluate_dict = {
        "eval_loss": total_loss / (total_tokens + 1e-6),
        "epoch": epoch,
    }
    accelerator.log(evaluate_dict, step=step)


def validation(model, tokenizer, valid_dataloader, args, accelerator, weight_dtype, step, epoch, is_final_validation=False):
    rewritten_examples = []
    evaluator = TextAbbreviationEvaluator(lower_case=True)

    if not is_final_validation:
        eval_model = accelerator.unwrap_model(model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        eval_model = AutoModelForCausalLM.from_pretrained(args.output_dir, torch_dtype=weight_dtype)

    for batch in tqdm(valid_dataloader, desc=f"Evaluating"):
        model_inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
        }

        with torch.no_grad():
            generated_ids = eval_model.generate(**model_inputs, max_new_tokens=1024)

        for short_caption, character_list, artist_list, full_ids in zip(
            batch["short_caption"],
            batch["tags_character"],
            batch["tags_artist"],
            generated_ids,
        ):
            rewritten_text = tokenizer.decode(full_ids, skip_special_tokens=True)
            input_text, rewritten_text = rewritten_text.split("assistant\n<think>\n\n</think>\n\n")
            rewritten_examples.append({
                "long_text": rewritten_text,
                "input_text": input_text,
                "abbreviated_text": short_caption,
                "character_list": character_list,
                "artist_list": artist_list,
            })

    # === 评估并记录 ===
    evaluate_dict = evaluator.evaluate_list(rewritten_examples)
    evaluate_dict["epoch"] = epoch
    accelerator.log(evaluate_dict, step=step)
    

def parse_args():
    parser = argparse.ArgumentParser()

    # 基本模型与实验设置
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--exp_name", type=str, default="v0.2")
    parser.add_argument("--dataset", type=str, default="incantor/DiTPromptHelperSFT-v0.2")
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--gradient_checkpointing", action="store_true")

    # 输出路径与日志
    parser.add_argument("--output_dir", type=str, default="/data/ruizhe/DiTPromptHelperSFT/outputs")
    parser.add_argument("--logging_dir", type=str, default="logs")

    # 优化器与学习率设置
    parser.add_argument("--lr", "--learning_rate", dest="learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--optimizer_type", type=str, default="adamw", choices=["adamw", "prodigy", "8bit_adam"])
    parser.add_argument("--optimizer_args", nargs='*', default=None, help="Extra optimizer args as key=value")

    # batch与训练设置
    parser.add_argument("--train_batch_size", type=int, default=1)  # 可选但建议显式设置
    parser.add_argument("--valid_batch_size", type=int, default=4)  # 可选但建议显式设置
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--max_train_steps", type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # 日志与监控
    parser.add_argument("--init_tracker", action="store_true")
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--tracker_project_name", type=str, default="DiTPromptHelperSFT")

    # checkpoint相关
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--checkpointing_steps", type=int, default=1000)
    parser.add_argument("--checkpoints_total_limit", type=int, default=3)

    # 验证设置
    parser.add_argument("--validation_steps", type=int, default=100)
    parser.add_argument("--validation_at_begin", action="store_true")

    # 学习率调度器
    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["linear", "cosine", "constant", "polynomial"])
    parser.add_argument("--warmup_ratio", type=float, default=0.03)

    # 随机种子
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])

    args = parser.parse_args()
    return args


def main(args):
    args.output_dir = os.path.join(args.output_dir, args.exp_name)
    logging_dir = Path(args.output_dir, args.logging_dir)
    deepspeed_plugin = None
    if args.deepspeed:
        deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=args.deepspeed) 

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir),
        deepspeed_plugin=deepspeed_plugin,
    )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
    else:
        transformers.utils.logging.set_verbosity_error()
    
    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        attn_implementation="flash_attention_2",
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    model.to(accelerator.device, dtype=weight_dtype)
    model.train()

    # 数据加载
    with accelerator.main_process_first():
        cache_dir = tempfile.mkdtemp()
        dataset = load_dataset(args.dataset, cache_dir=cache_dir)
        train_dataset = dataset["train"]
        valid_dataset = dataset["valid"]

        # version = args.exp_name
        # map_func = load_map_func(version)
        # train_dataset = train_dataset.map(partial(map_func, is_valid=False), remove_columns=train_dataset.column_names, num_proc=20)
        # valid_dataset = valid_dataset.map(partial(map_func, is_valid=True), remove_columns=valid_dataset.column_names)

    accelerator.wait_for_everyone()
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.train_batch_size, 
        shuffle=True, 
        pin_memory=True,
        collate_fn=partial(train_collate_fn, tokenizer=tokenizer),
        num_workers=args.dataloader_num_workers,
    )
    valid_dataloader = DataLoader(
        valid_dataset, 
        batch_size=args.valid_batch_size, 
        pin_memory=True,
        collate_fn=partial(valid_collate_fn, tokenizer=tokenizer),
        num_workers=args.dataloader_num_workers,
    )

    # optimizer
    optimizer_kwargs = {}
    if args.optimizer_args is not None and len(args.optimizer_args) > 0:
        for arg in args.optimizer_args:
            key, value = arg.split("=")
            value = ast.literal_eval(value)
            optimizer_kwargs[key] = value

    params_to_optimize = model.parameters()

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.optimizer_type == "8bit_adam":
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )
        optimizer = bnb.optim.AdamW8bit(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.weight_decay,
            eps=args.adam_epsilon,
            **optimizer_kwargs
        )
    elif args.optimizer_type == "prodigy":
        from prodigyopt import Prodigy
        optimizer = Prodigy(
            params_to_optimize,
            lr=1.,
            **optimizer_kwargs
        )
    else:
        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.weight_decay,
            eps=args.adam_epsilon,
            **optimizer_kwargs
        )

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    num_warmup_steps_for_scheduler = args.warmup_ratio * num_training_steps_for_scheduler

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )

    # prepare with accelerate
    model, optimizer, train_dataloader, valid_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, valid_dataloader, lr_scheduler
    )

    # save_dir = os.path.join(args.output_dir, f"checkpoint-0000")
    # accelerator.save_state(save_dir)
    # if accelerator.is_main_process:
    #     unwrapped_model = accelerator.unwrap_model(model)
    #     unwrapped_model.save_pretrained(save_dir)
    #     tokenizer.save_pretrained(save_dir)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process and args.init_tracker:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(
            args.tracker_project_name, 
            config=tracker_config,
            init_kwargs={
                "wandb": {
                    "name": f"{os.path.basename(args.output_dir)}",
                }
            }
        )

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    if args.validation_at_begin:
        if accelerator.is_main_process:
            validation_log = validation(
                model=model,
                tokenizer=tokenizer,
                valid_dataloader=valid_dataloader,
                args=args,
                accelerator=accelerator,
                weight_dtype=weight_dtype,
                step=global_step,
                epoch=0,
            )

        validation_loss(
            model=model,
            tokenizer=tokenizer,
            valid_dataloader=valid_dataloader,
            args=args,
            accelerator=accelerator,
            weight_dtype=weight_dtype,
            step=global_step,
            epoch=0,
        )

    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                params_to_clip = model.parameters()
                accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_steps == 0:
                    save_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")

                    # === 1. 保存完整训练状态：支持 resume（每个卡都需要执行）
                    accelerator.save_state(save_dir)

                    # === 2. 只主进程保存 Hugging Face 权重：支持 from_pretrained 加载
                    if accelerator.is_main_process:
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(save_dir)
                        tokenizer.save_pretrained(save_dir)

                        logger.info(f"[Saved] HF-compatible model to {save_dir}")
                        logger.info(f"[Saved] full training state to {save_dir}")

                        # === 3. 删除旧 checkpoint（只保留最新 N 个）
                        if args.checkpoints_total_limit is not None:
                            checkpoints = [
                                d for d in os.listdir(args.output_dir)
                                if os.path.isdir(os.path.join(args.output_dir, d)) and d.startswith("checkpoint-")
                            ]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))

                            if len(checkpoints) > args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit
                                to_delete = checkpoints[:num_to_remove]

                                logger.info(f"Cleaning up {num_to_remove} old checkpoints: {to_delete}")
                                for ckpt in to_delete:
                                    ckpt_path = os.path.join(args.output_dir, ckpt)
                                    if os.path.exists(ckpt_path):
                                        try:
                                            shutil.rmtree(ckpt_path)
                                            logger.info(f"[Cleanup] Removed: {ckpt_path}")
                                        except Exception as e:
                                            logger.warning(f"[Cleanup] Failed to remove {ckpt_path}: {e}")

                if global_step % args.validation_steps == 0:
                    if accelerator.is_main_process:
                        validation_log = validation(
                            model=model,
                            tokenizer=tokenizer,
                            valid_dataloader=valid_dataloader,
                            args=args,
                            accelerator=accelerator,
                            weight_dtype=weight_dtype,
                            step=global_step,
                            epoch=epoch,
                        )
                    
                    validation_loss(
                        model=model,
                        tokenizer=tokenizer,
                        valid_dataloader=valid_dataloader,
                        args=args,
                        accelerator=accelerator,
                        weight_dtype=weight_dtype,
                        step=global_step,
                        epoch=epoch,
                    )

            if args.optimizer_type.lower() == "prodigy":
                d_lr = (
                    lr_scheduler.optimizers[0].param_groups[0].get("d", 1.0) *
                    lr_scheduler.optimizers[0].param_groups[0]["lr"]
                )
                logs = {
                    "loss": loss.detach().item(),
                    "d*lr": d_lr
                }
            else:
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
        
        if accelerator.is_main_process:
            epoch_save_dir = os.path.join(args.output_dir, f"epoch-{epoch}")
            os.makedirs(epoch_save_dir, exist_ok=True)
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(epoch_save_dir)
            tokenizer.save_pretrained(epoch_save_dir)

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        model.save_pretrained(args.output_dir)

        # Run a final round of validation.
        # validation_log = validation(
        #     model=None,
        #     tokenizer=None,
        #     valid_dataloader=valid_dataloader,
        #     args=args,
        #     accelerator=accelerator,
        #     weight_dtype=weight_dtype,
        #     step=global_step,
        #     epoch=args.num_train_epochs,
        #     is_final_validation=True,
        # )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)