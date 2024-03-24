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

WCS_CLUSTER_URL = "https://anthony-sandbox-7f0z4seo.weaviate.network"
MISTRAL_API = "TWfVrlX659GSTS9hcsgUcPZ8uNzfoQsg"

mistral_client = MistralClient(api_key=MISTRAL_API)
client = weaviate.connect_to_wcs(
    cluster_url=WCS_CLUSTER_URL,
    auth_credentials=None
)

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

query = "I want to build a company that will revolutionize the way we think about food"
companies = client.collections.get("YCCompanies")
result = query_collection(query, companies, mistral_client)

print([obj.properties['company_name'] for obj in result.objects])

