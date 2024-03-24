from flask import Flask, jsonify, request
from flask_cors import CORS

from collections import defaultdict
import json
import numpy as np
import os
import requests
import json
import pandas as pd

import torch
from transformers import (
    BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
)

app = Flask(__name__)
CORS(app)

BASE_PATH = "/exp/mistral_instruct_generation/checkpoint-1000"

tokenizer = AutoTokenizer.from_pretrained(BASE_PATH)

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_PATH,
    device_map='auto',
    load_in_4bit=True, low_cpu_mem_usage=True, 
    # quantization_config=nf4_config,
    use_cache=False,
    attn_implementation="flash_attention_2",
    max_length=1000
)


# tokenizer_gm = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
# model_gm = AutoModelForCausalLM.from_pretrained(
#     "mistralai/Mistral-7B-Instruct-v0.2",
#     quantization_config=nf4_config,
#     device_map='auto',
#     attn_implementation="sdpa",
#     max_length=1000
# )

# model_gm = model_gm.eval()
model = model.eval()

@app.route("/chat", methods = ['POST'])
def get_embedding_map():
    data = request.get_json()
    text = data.get('text', '')
    with torch.no_grad():
        model_input = tokenizer.encode(text, return_tensors="pt").to('cuda')
        
        result = model.generate(input_ids=model_input, use_cache=True)
        # result = model.generate(input_ids=model_input, use_cache=True, assistant_model=model_gm)

        generated_text = tokenizer.decode(result[0], skip_special_tokens=True)
        print(generated_text)
    return jsonify({'output': generated_text})

if __name__=='__main__': 
    app.run(debug=True, host="0.0.0.0", port=5555)
