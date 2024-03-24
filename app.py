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


def init_vector_db(client, df, mistral_client):
    """
    requires df has columns
    - company_name
    - long_description
    - location

    modifies df with new column embedding based on long description
    """
    companies = client.collections.create(
        name="YCCompanies",
        vectorizer_config=None
    )

    ### EMBED
    long_descriptions = list(df['long_description'].dropna().values)
    all_vecs = []
    for batch in zip(*(iter(long_descriptions),) * 5):
        embeddings_response = mistral_client.embeddings(
            model="mistral-embed",
            input=list(batch)
        )
        vecs = [vec.embedding for vec in embeddings_response.data]
        all_vecs += vecs
        time.sleep(0.97)
    df['vector'] = all_vecs

    companies_objs = list()
    for idx, row in df.iterrows():
        companies_objs.append(wvc.data.DataObject(
                properties={
                    "company_name": row["company_name"],
                    "long_description": row["long_description"],
                    "location": row["location"],
                },
                vector=row["vector"]
            ))
    companies.data.insert_many(companies_objs)


def main(query):
    #client = weaviate.connect_to_local()
    WCS_CLUSTER_URL = "https://anthony-sandbox-7f0z4seo.weaviate.network"
    client = weaviate.connect_to_wcs(
        cluster_url=WCS_CLUSTER_URL,
        auth_credentials=None
    )

    mistral_client = MistralClient(api_key=MISTRAL_API)

    ## Start fresh:
    # TODO: check if it is populated, if so don't initialize
    # if client.collections.exists("YCCompanies"):
    #     client.collections.delete("YCCompanies")
        
    #df = pd.read_csv('data/2023-07-13-yc-companies.csv')
    #df = df.loc[:9]

    #init_vector_db(client, df, mistral_client)


    companies = client.collections.get("YCCompanies")
    result = query_collection(query, companies, mistral_client)

    return result


if __name__ == '__main__':
    app.run(debug=True)