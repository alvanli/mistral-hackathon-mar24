import weaviate
import weaviate.classes as wvc
import os
import requests
import json
import pandas as pd

from mistralai.client import MistralClient
import time


MISTRAL_API = "TWfVrlX659GSTS9hcsgUcPZ8uNzfoQsg"


# Read jsonl in ./data/all_news_stores.jsonl
def read_jsonl(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return [json.loads(line) for line in lines]


def add_news_stories(client, mistral_client):
    news = client.collections.create(
        name="News",
        vectorizer_config=None
    )

    news_stories = read_jsonl('./data/all_news_stories.jsonl')

    ### EMBED
    descriptions = [story["description"] for story in news_stories]
    batched_descriptions = [descriptions[i:i+5] for i in range(0, len(descriptions), 5)]
    all_vecs = []
    for batch in batched_descriptions:
        embeddings_response = mistral_client.embeddings(
            model="mistral-embed",
            input=list(batch)
        )
        vecs = [vec.embedding for vec in embeddings_response.data]
        all_vecs += vecs
        time.sleep(0.97)

    news_objs = list()
    
    for idx, data in enumerate(zip(news_stories, all_vecs)):
        story, vec = data
        news_objs.append(wvc.data.DataObject(
                properties={
                    "company": story["company"],
                    "newsstory": story["newsstory"],
                    "link": story["link"],
                    "description": story["description"],
                },
                vector=vec
            ))
        if idx % 100 == 0:
            news.data.insert_many(news_objs)
            news_objs = []
    news.data.insert_many(news_objs)


def add_ycombinator_companies(client, mistral_client):
    companies = client.collections.create(
        name="YCCompanies",
        vectorizer_config=None
    )

    df = pd.read_csv('data/2023-07-13-yc-companies.csv')
    df2 = pd.read_csv('data/2023-02-27-yc-companies.csv')
    df = pd.concat([df, df2])

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
                    "company": row["company_name"],
                    "description": row["long_description"],
                    "location": row["location"],
                },
                vector=row["vector"]
            ))
        if idx % 100 == 0:
            companies.data.insert_many(companies_objs)
            companies_objs = []
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
    # TODO: check if it is populated, if so don't reinitialize, do something less dumb Anthony
    if client.collections.exists("YCCompanies"):
        client.collections.delete("YCCompanies")

    ## Init vector db
    add_ycombinator_companies(client, mistral_client)
    add_news_stories(client, mistral_client)

    client.close()

    return


if __name__ == '__main__':
    main()
