import weaviate
import weaviate.classes as wvc
import os
import requests
import json
import pandas as pd

from mistralai.client import MistralClient
import time


MISTRAL_API = "TWfVrlX659GSTS9hcsgUcPZ8uNzfoQsg"



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


def main():
    # client = weaviate.connect_to_local()
    WCS_CLUSTER_URL = "https://anthony-sandbox-7f0z4seo.weaviate.network"
    client = weaviate.connect_to_wcs(
        cluster_url=WCS_CLUSTER_URL,
        auth_credentials=None
    )
    mistral_client = MistralClient(api_key=MISTRAL_API)

    ## Start fresh:
    # TODO: check if it is populated, if so don't initialize
    if client.collections.exists("YCCompanies"):
        client.collections.delete("YCCompanies")
    
    # Tmp just read csv
    df = pd.read_csv('data/2023-07-13-yc-companies.csv')
    df = df.loc[:9]

    init_vector_db(client, df, mistral_client)
    client.close()

    return


if __name__ == '__main__':
    main()
