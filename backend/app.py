from flask import Flask, jsonify, request
from flask_cors import CORS

from collections import defaultdict
import json
import numpy as np

import weaviate
import weaviate.classes as wvc
import os
import requests
import json
import pandas as pd

from mistralai.client import MistralClient
import time


MISTRAL_API = "TWfVrlX659GSTS9hcsgUcPZ8uNzfoQsg"


app = Flask(__name__)
CORS(app)

BASE_PATH = "./last_1000"

@app.route("/get_embedding_map", methods = ['GET'])
def get_embedding_map():
    with open(f"{BASE_PATH}/projections.npy", "rb") as f:
        projections = np.load(f).tolist()

    with open(f"{BASE_PATH}/cluster_labels.npy", "rb") as f:
        cluster_labels = np.load(f).tolist()

    with open(f"{BASE_PATH}/texts.json", "r") as f:
        texts = json.load(f)

    with open(f"{BASE_PATH}/titles.json", "r") as f:
        titles = json.load(f)
    
    with open(f"{BASE_PATH}/cluster_summaries.json", "r") as f:
        cluster_summaries = json.load(f)
    keys = list(cluster_summaries.keys())
    for key in keys:
        cluster_summaries[int(key)] = cluster_summaries.pop(key)

    id2cluster = {
        index: label for index, label in enumerate(cluster_labels)
    }
    label2docs = defaultdict(list)
    for i, label in enumerate(cluster_labels):
        label2docs[label].append(i)
    cluster_centers = {}
    for label in label2docs.keys():
        x = np.mean([projections[doc][0] for doc in label2docs[label]])
        y = np.mean([projections[doc][1] for doc in label2docs[label]])
        cluster_centers[label] = (x, y)

    return jsonify({
        "companies": [{
            "projection": proj,
            "label": lab,
            "text": text,
            "title": title
        } for proj, lab, text, title in zip(projections, cluster_labels, texts, titles)],
        "clusters": {
            k: {
                "center": cluster_centers[k],
                "summary": cluster_summaries[k]
            } for k in cluster_centers.keys()
        }
        
    })


@app.route('/query_vector_db', methods=['POST'])
def process_text():
    data = request.get_json()
    text = data.get('text', '')
    result = handle_query(text)
    return result


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


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    query = data.get('text', '')

    # query vector db to get relevant company info
    rag_result = handle_query(query)

    context = "".join([f"""{company['company']} is a company located in {company['location']}. They are described as {company['description']}.\n""" for company in companies])

    # Stuff context into query
    prompt =  """
        <s>[INST] Context information is below.
        ---------------------
        {tractor}
        ---------------------
        Given the context information and not prior knowledge, answer the query.
        Query: {question} [/INST]
    """.format(
        tractor=context,
        question=query
    )
    
    # Send prompt to localhost 5555
    url = "http://localhost:5555/chat"
    response = requests.post(url, json={"text": prompt})
    return response.json()


def query_collection(query, collection, mistral_client):
    ### Embed query
    emb_query = mistral_client.embeddings(
        model="mistral-embed",
        input=[query]
    ).data[0].embedding

    result = (
        collection.query.near_vector(
        near_vector=emb_query, 
            limit=2
        )
    )

    return result


def query_collection2(vectordb_client, query, collection_name, mistral_client):
    ### Embed query
    emb_query = mistral_client.embeddings(
        model="mistral-embed",
        input=[query]
    ).data[0].embedding

    response = (
        vectordb_client.query
        .get(collection_name, ["company", "description", "location"])
        .with_near_vector(
            {
                "vector": emb_query,
            }
        )
        .with_limit(2)
        .do()
    )
    #response = response["data"]["Get"][collection_name]
    return response["data"]["Get"][collection_name]


def handle_query(query):
    WCS_CLUSTER_URL = "https://anthony-sandbox-7f0z4seo.weaviate.network"
    client = weaviate.Client(
        url=WCS_CLUSTER_URL,
        auth_client_secret=None
    )

    mistral_client = MistralClient(api_key=MISTRAL_API)
    
    result = query_collection2(client, query, "YCCompanies", mistral_client)
    #companies = client.collections.get("YCCompanies")
    #news = client.collections.get("News") # what 2 do here?
    #result = query_collection(query, companies, mistral_client)

    return result


if __name__=='__main__': 
    app.run(debug=True, host="0.0.0.0")
    