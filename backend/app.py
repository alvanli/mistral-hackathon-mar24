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

BASE_PATH = "C://Users//alvin//DATA//dsc//mis_thon//mistral-hackathon-mar24//text-clustering//last_1000"

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
    result = main(text)
    result = [obj.properties['company_name'] for obj in result.objects]
    return {'result': result}, 200


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


def handle_query(query):
    WCS_CLUSTER_URL = "https://anthony-sandbox-7f0z4seo.weaviate.network"
    client = weaviate.connect_to_wcs(
        cluster_url=WCS_CLUSTER_URL,
        auth_credentials=None
    )

    mistral_client = MistralClient(api_key=MISTRAL_API)
    companies = client.collections.get("YCCompanies")
    news = client.collections.get("News") # what 2 do here?
    result = query_collection(query, companies, mistral_client)

    return result


if __name__=='__main__': 
    app.run(debug=True)
    