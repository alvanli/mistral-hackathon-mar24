import weaviate
import weaviate.classes as wvc
import os
import requests
import json
import pandas as pd

from mistralai.client import MistralClient
import time

# from llama_index import SimpleDirectoryReader
# from llama_index.node_parser import SimpleNodeParser

from flask import Flask, request

MISTRAL_API = "TWfVrlX659GSTS9hcsgUcPZ8uNzfoQsg"

app = Flask(__name__)


@app.route('/query_vector_db', methods=['POST'])
def process_text():
    data = request.get_json()
    text = data.get('text', '')
    result = main(text)
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


def main(query):
    #client = weaviate.connect_to_local()
    WCS_CLUSTER_URL = "https://anthony-sandbox-7f0z4seo.weaviate.network"
    client = weaviate.connect_to_wcs(
        cluster_url=WCS_CLUSTER_URL,
        auth_credentials=None
    )

    mistral_client = MistralClient(api_key=MISTRAL_API)

    companies = client.collections.get("YCCompanies")
    result = query_collection(query, companies, mistral_client)

    return result


if __name__ == '__main__':
    app.run(debug=True)