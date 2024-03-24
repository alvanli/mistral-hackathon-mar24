import time
import torch

from transformers import (
    BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, TrainingArguments
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from trl import SFTTrainer


def create_prompt(sample):
    bos_token = "<s>"
    original_system_message = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    system_message = "Use the provided input to create an instruction that could have been used to generate the response with an LLM."
    response = sample["prompt"].replace(original_system_message, "").replace("\n\n### Instruction\n", "").replace("\n### Response\n", "").strip()
    input = sample["response"]
    eos_token = "</s>"

    full_prompt = ""
    full_prompt += bos_token
    full_prompt += "### Instruction:"
    full_prompt += "\n" + system_message
    full_prompt += "\n\n### Input:"
    full_prompt += "\n" + input
    full_prompt += "\n\n### Response:"
    full_prompt += "\n" + response
    full_prompt += eos_token

    return full_prompt


if __name__ == "__main__":
    nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.1",
        device_map='auto',
        quantization_config=nf4_config,
        use_cache=False
    )

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    peft_config = LoraConfig(
        use_dora=True,
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    args = TrainingArguments(
        output_dir = "mistral_instruct_generation",
        #num_train_epochs=5,
        max_steps = 100, 
        per_device_train_batch_size = 4,
        warmup_steps = 0.03,
        logging_steps=10,
        save_strategy="epoch",
        #evaluation_strategy="epoch",
        evaluation_strategy="steps",
        eval_steps=20,
        learning_rate=2e-4,
        bf16=True,
        lr_scheduler_type='constant',
    )

    max_seq_length = 2048

    trainer = SFTTrainer(
        model=model,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        formatting_func=create_prompt,
        args=args,
        train_dataset=instruct_tune_dataset["train"],
        eval_dataset=instruct_tune_dataset["test"]
    )

    start = time.time()
    trainer.train()
    print(time.time()- start)

    trainer.save_model("mistral_instruct_generation")