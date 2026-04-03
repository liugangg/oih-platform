#!/usr/bin/env python3
"""
QLoRA fine-tuning script for Qwen3-14B on OIH distillation data.

Usage:
    python scripts/train_qlora.py

Requirements:
    pip install transformers>=4.46 peft>=0.13 trl>=0.12 bitsandbytes>=0.44 datasets accelerate
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
import logging
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH = "/data/oih/models/Qwen3-14B"
DATA_PATH = "/data/oih/oih-api/data/distillation/distillation_v2.jsonl"
OUTPUT_DIR = "/data/oih/models/Qwen3-14B-OIH-LoRA"
MAX_SEQ_LEN = 2048
TRAIN_RATIO = 0.9

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_gpu_info():
    """Print GPU information."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_gb = props.total_memory / (1024 ** 3)
            logger.info(
                f"GPU {i}: {props.name}  |  {total_gb:.1f} GB  |  "
                f"Compute capability {props.major}.{props.minor}"
            )
    else:
        logger.error("No CUDA GPUs detected. Exiting.")
        sys.exit(1)


SYSTEM_PROMPT = "You are the AI assistant for the OIH bio-computing platform. Analyze user bioinformatics requests and invoke the appropriate tools."


def format_to_chatml(entry):
    """Convert a distillation entry to ChatML messages format."""
    user_content = entry.get("instruction", "") + "\n" + entry.get("input", "")
    assistant_parts = []
    if entry.get("reasoning"):
        assistant_parts.append(entry["reasoning"])
    if entry.get("action"):
        assistant_parts.append(entry["action"])
    if entry.get("outcome"):
        assistant_parts.append(entry["outcome"])
    assistant_content = "\n".join(assistant_parts)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content.strip()},
            {"role": "assistant", "content": assistant_content.strip()},
        ]
    }


def load_data(tokenizer):
    """Load raw distillation JSONL → ChatML format, compute stats."""
    raw = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw.append(json.loads(line))

    logger.info(f"Loaded {len(raw)} raw entries from {DATA_PATH}")

    records = [format_to_chatml(e) for e in raw]

    # Compute token length stats
    token_lengths = []
    for rec in records:
        text = tokenizer.apply_chat_template(
            rec["messages"], tokenize=False, add_generation_prompt=False
        )
        toks = tokenizer(text, truncation=False)["input_ids"]
        token_lengths.append(len(toks))

    avg_len = sum(token_lengths) / len(token_lengths)
    max_len = max(token_lengths)
    min_len = min(token_lengths)
    over_limit = sum(1 for l in token_lengths if l > MAX_SEQ_LEN)

    logger.info(f"Token length stats: avg={avg_len:.0f}  min={min_len}  max={max_len}")
    logger.info(f"Sequences exceeding {MAX_SEQ_LEN} tokens: {over_limit}/{len(records)}")

    # Train / eval split
    import random
    random.seed(42)
    random.shuffle(records)
    split_idx = int(len(records) * TRAIN_RATIO)
    train_records = records[:split_idx]
    eval_records = records[split_idx:]

    logger.info(f"Train: {len(train_records)}  |  Eval: {len(eval_records)}")

    train_ds = Dataset.from_list(train_records)
    eval_ds = Dataset.from_list(eval_records)

    return train_ds, eval_ds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("=" * 60)
    logger.info("  QLoRA Fine-tuning: Qwen3-14B -> OIH Agent")
    logger.info("=" * 60)

    # --- GPU info ---
    print_gpu_info()

    # --- Check model path ---
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        logger.error(f"Model path does not exist: {MODEL_PATH}")
        logger.error("Qwen3-14B may still be downloading. Check and retry.")
        sys.exit(1)

    config_file = model_path / "config.json"
    if not config_file.exists():
        logger.error(f"No config.json found in {MODEL_PATH}. Download may be incomplete.")
        sys.exit(1)

    logger.info(f"Model path: {MODEL_PATH}")

    # --- Tokenizer ---
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token = eos_token ({tokenizer.eos_token})")

    # --- Data ---
    logger.info("Loading and tokenizing data...")
    train_ds, eval_ds = load_data(tokenizer)

    # --- Quantization config ---
    logger.info("Configuring 4-bit quantization (NF4 + BF16 compute)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # --- Model ---
    logger.info("Loading Qwen3-14B in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False

    # Prepare for k-bit training (freeze base, enable gradient checkpointing)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # --- LoRA config ---
    logger.info("Applying LoRA adapters (r=16, alpha=32)...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Training arguments ---
    logger.info("Setting up training arguments...")
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.1,
        weight_decay=0.01,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        lr_scheduler_type="cosine",
        dataloader_num_workers=2,
        remove_unused_columns=False,
        max_length=MAX_SEQ_LEN,
    )

    # --- Trainer ---
    logger.info("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    # --- Train ---
    logger.info("Starting training...")
    logger.info(
        f"  Epochs: {training_args.num_train_epochs}  |  "
        f"Batch: {training_args.per_device_train_batch_size}  |  "
        f"Grad accum: {training_args.gradient_accumulation_steps}  |  "
        f"Effective batch: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}  |  "
        f"Save: every epoch"
    )
    trainer.train()

    # --- Save LoRA adapter ---
    logger.info(f"Saving LoRA adapter to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    logger.info("=" * 60)
    logger.info("  Training complete!")
    logger.info("=" * 60)

    # --- Post-training instructions ---
    print(
        f"""
================================================================================
  QLoRA training finished. LoRA adapter saved to:
    {OUTPUT_DIR}

  To merge LoRA weights into the base model:
  ---------------------------------------------------------------------------
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(
        "{MODEL_PATH}",
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    model = PeftModel.from_pretrained(base, "{OUTPUT_DIR}")
    merged = model.merge_and_unload()
    merged.save_pretrained("/data/oih/models/Qwen3-14B-OIH-merged")
    AutoTokenizer.from_pretrained("{OUTPUT_DIR}").save_pretrained(
        "/data/oih/models/Qwen3-14B-OIH-merged"
    )

  To deploy with vLLM (without merging — LoRA adapter hot-swap):
  ---------------------------------------------------------------------------
    python -m vllm.entrypoints.openai.api_server \\
        --model {MODEL_PATH} \\
        --enable-lora \\
        --lora-modules oih-agent={OUTPUT_DIR} \\
        --port 8002 \\
        --dtype bfloat16 \\
        --max-model-len 32768 \\
        --gpu-memory-utilization 0.90 \\
        --tensor-parallel-size 1

    Then use model="oih-agent" in API requests.

  To deploy with vLLM (merged model):
  ---------------------------------------------------------------------------
    python -m vllm.entrypoints.openai.api_server \\
        --model /data/oih/models/Qwen3-14B-OIH-merged \\
        --port 8002 \\
        --dtype bfloat16 \\
        --max-model-len 32768 \\
        --gpu-memory-utilization 0.90 \\
        --tensor-parallel-size 1
================================================================================
"""
    )


if __name__ == "__main__":
    main()
