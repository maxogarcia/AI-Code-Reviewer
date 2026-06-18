"""
QLoRA fine-tuning of Qwen2.5-Coder-7B-Instruct on code review data.

Usage:
    python training/train.py [--smoke-test]

    --smoke-test: Run on 100 examples for 10 steps to verify the pipeline end-to-end.
"""

import argparse

import mlflow
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
DATASET_ID = "alenphilip/Code-Review-Assistant"
OUTPUT_DIR = "outputs/qlora-adapter"
MLFLOW_EXPERIMENT = "qlora-code-reviewer"

LORA_CONFIG = dict(
    r=16,
    lora_alpha=32,
    # Qwen2.5 attention + MLP projection layers
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

TRAIN_CONFIG = dict(
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,  # effective batch size = 16
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=50,
    bf16=True,
    logging_steps=10,
    save_steps=100,
    eval_steps=100,
    eval_strategy="steps",
    save_total_limit=2,
    load_best_model_at_end=False,  # incompatible with PEFT + 4-bit quantization
    report_to="none",  # MLflow handled manually below
    dataloader_num_workers=0,
    dataset_text_field="text",
    max_length=2048,
    # packing=True requires flash_attention_2 to avoid cross-sample contamination — enable once installed
    packing=False,
)


def build_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_model_and_tokenizer(bnb_config: BitsAndBytesConfig):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model.enable_input_require_grads()
    return model, tokenizer


def apply_lora(model):
    model = get_peft_model(model, LoraConfig(**LORA_CONFIG))
    model.print_trainable_parameters()
    return model


def load_and_split_dataset(smoke_test: bool):
    ds = load_dataset(DATASET_ID, split="train")
    if smoke_test:
        ds = ds.select(range(100))
    split = ds.train_test_split(test_size=0.05, seed=42)
    return split["train"], split["test"]


def main(smoke_test: bool = False):
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    bnb_config = build_bnb_config()
    model, tokenizer = load_model_and_tokenizer(bnb_config)
    model = apply_lora(model)
    train_ds, eval_ds = load_and_split_dataset(smoke_test)

    overrides = {}
    if smoke_test:
        overrides = dict(
            max_steps=10,
            num_train_epochs=1,
            eval_steps=5,
            save_steps=5,
            logging_steps=1,
        )

    sft_cfg = SFTConfig(
        output_dir=OUTPUT_DIR,
        **{**TRAIN_CONFIG, **overrides},
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    with mlflow.start_run() as run:
        mlflow.log_params({
            "model_id": MODEL_ID,
            "dataset_id": DATASET_ID,
            "lora_r": LORA_CONFIG["r"],
            "lora_alpha": LORA_CONFIG["lora_alpha"],
            "lora_target_modules": ",".join(LORA_CONFIG["target_modules"]),
            "learning_rate": TRAIN_CONFIG["learning_rate"],
            "epochs": TRAIN_CONFIG["num_train_epochs"] if not smoke_test else 1,
            "batch_size": TRAIN_CONFIG["per_device_train_batch_size"],
            "grad_accum": TRAIN_CONFIG["gradient_accumulation_steps"],
            "max_length": TRAIN_CONFIG["max_length"],
            "smoke_test": smoke_test,
        })

        print(f"MLflow run: {run.info.run_id}")
        train_result = trainer.train()
        trainer.save_model()
        tokenizer.save_pretrained(OUTPUT_DIR)

        mlflow.log_metrics({
            "train_loss": train_result.training_loss,
            "train_runtime_s": train_result.metrics.get("train_runtime", 0),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
        })
        mlflow.log_artifact(OUTPUT_DIR)

    print(f"Adapter saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="Run 10 steps on 100 examples")
    args = parser.parse_args()
    main(smoke_test=args.smoke_test)
