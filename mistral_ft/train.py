import time
import json
import pandas as pd
import torch
import random
from transformers import (
    BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, TrainingArguments
)
from datasets import Dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from trl import SFTTrainer

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def add_text(row):
    golden = row['golden']
    distractors = [row['distractor1'],row['distractor2'],row['distractor3']]

    if golden:
        distractors.append(golden)
    
    random.shuffle(distractors)

    prompt = """
        <s>[INST] Context information is below.
        ---------------------
        {tractor}
        ---------------------
        Given the context information and not prior knowledge, answer the query.
        Query: {question} [/INST]
        {answer} </s>
    """.format(
        tractor="\n".join(distractors),
        question=row['question'],
        answer=row['answer']
    )
    row['text'] = prompt
    return row


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


    df = pd.read_csv("./data_new.csv")
    ds = Dataset.from_pandas(df)
    ds = ds.map(add_text)
    ds = ds.train_test_split(test_size=0.1, seed=42)

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
        num_train_epochs=3,
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
        dataset_text_field="text",
        packing=True,
        args=args,
        train_dataset=ds['train'],
        eval_dataset=ds["test"]
    )

    start = time.time()
    trainer.train()
    print(time.time()- start)

    trainer.save_model("mistral_instruct_generation")