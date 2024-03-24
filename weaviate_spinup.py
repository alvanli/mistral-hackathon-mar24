import weaviate
import weaviate.classes as wvc
import os
import requests
import json
import pandas as pd

from mistralai.client import MistralClient


def init_vector_db(client, df):
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
    mistral_client = MistralClient(api_key=MISTRAL_API)
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


def query_collection(query, collection):
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


def main():
    client = weaviate.connect_to_local()

    ## Start fresh:
    # TODO: check if it is populated, if so don't initialize
    if client.collections.exists("YCCompanies"):
        client.collections.delete("YCCompanies")
    
    # Tmp just read csv
    df = pd.read_csv('./data/2023-07-13-yc-companies.csv')

    init_vector_df(client, df)

    example_query = "What companies are developing AI to understand patients' genotype-phenotype relationships"
    companies = client.collections.get("YCCompanies")
    result = query_collection(example_query, companies)
