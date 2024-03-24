from flask import Flask, jsonify, request
from flask_cors import CORS

from collections import defaultdict
import json
import numpy as np

import weaviate
import weaviate.classes as wvc
import os
import requests
import joblib
import json
import pandas as pd

from mistralai.client import MistralClient
import time

from groq import Groq


MISTRAL_API = "TWfVrlX659GSTS9hcsgUcPZ8uNzfoQsg"
#GROQ_API = "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"
#GROQ_API2 = "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"


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


@app.route('/addNewIdea', methods=['POST'])
def addNewIdea():
    data = request.get_json()
    query = data.get('text', '')   
    mistral_client = MistralClient(api_key=MISTRAL_API)
    model = "mistral-large-latest"
    from mistralai.models.chat_completion import ChatMessage
    prompt = """
    Given the following chat history, summarize the new company that the user is pitching: \n
    """
    messages = [
        ChatMessage(role="user", content=prompt + query)
    ]

    chat_response = mistral_client.chat(
        model=model,
        messages=messages,
    )
    newCompSummary = chat_response.choices[0].message.content

    emb_query = mistral_client.embeddings(
        model="mistral-embed",
        input=[newCompSummary]
    ).data[0].embedding

    umap_model = joblib.load(f"./{BASE_PATH}/umap_model.pkl")
    two_dim_emb = umap_model.transform([emb_query])

    return jsonify({
        "summary": newCompSummary,
        "center": two_dim_emb.tolist()
    })



@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    query = data.get('text', '')
    # query vector db to get relevant company info
    rag_result = handle_query(query)

    context = "".join([f"""{company['company']} is a company located in {company['location']}. They are described as {company['description']}.\n""" for company in rag_result])

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

    # Send prompt to Mistralai
    mistral_client = MistralClient(api_key=MISTRAL_API)
    model = "mistral-large-latest"
    from mistralai.models.chat_completion import ChatMessage
    messages = [
        ChatMessage(role="user", content=prompt)
    ]

    chat_response = mistral_client.chat(
        model=model,
        messages=messages,
    )
    return {
        "prompt": prompt,
        "response": chat_response.choices[0].message.content
    }

    
    # Send prompt to groq
    # client = Groq(
    #     api_key=GROQ_API
    # )
    # chat_completion = client.chat.completions.create(
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": prompt,
    #         }
    #     ],
    #     model="mixtral-8x7b-32768",
    # )
    # return chat_completion.choices[0].message.content

    # Send prompt to localhost 5555
    # url = "http://localhost:5555/chat"
    # response = requests.post(url, json={"text": prompt})
    # return response.json()


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


@app.route('/query_vector_db', methods=['POST'])
def query_vector_db():
    pass


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
    