import requests
import os
import json
import random
import time
from bs4 import BeautifulSoup as bs
import numpy as np
from tqdm import tqdm

from groq import Groq

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

def get_web(link):
    res = requests.get(link)
    time.sleep(1.5)
    try:
        content = res.content
        ssoup = bs(content, 'html.parser')
        return ssoup
    except:
        return None

def get_story_from_link(link):
    story = get_web(link)
    story_chunks = story.find_all('article')
    story_content_list = [s.text for s in story_chunks]
    story_content_str = " ".join(story_content_list) 
    return story_content_str

def get_startups_from_story(story):
    return link_href

def get_groq_response(message):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": message,
            }
        ],
        model="mixtral-8x7b-32768",
    )
    time.sleep(2)
    return chat_completion.choices[0].message.content


def get_groq_is_startup(message):
    return \
        get_groq_response("Is a company mentioned in the news story below? List the major companies involved separated by commas, if no companies are mentioned in the article, say 'NO COMPANIES'. Keep the answer short: \n\n{newsstory}".format(
            newsstory=message
        ))


def get_groq_summarize_startup(company, newsstory):
    return \
        get_groq_response("Summarize the description and core functions and services of the company {company} based on the following news story. If no explicit information is given, or if it is not a company, say 'NO INFORMATION': \n\n{newsstory}".format(
            company=company,
            newsstory=newsstory
        ))


if __name__ == "__main__":
    for page_idx in range(20,100):
        print("="*10)
        print(f"Doing page {page_idx}")
        print("="*10)

        soup = get_web(f"https://betakit.com/page/{page_idx}/")
        posts = soup.find_all("section", {"class": "posts-section"})
        links = posts[0].find_all("a", href=True)

        for link in tqdm(links):
            link_href = link['href']
            if 'page/' not in link_href and 'author/' not in link_href:
                story_content_str = get_story_from_link(link_href)
                is_startup = get_groq_is_startup(story_content_str)
                if "no companies" not in is_startup.lower():
                    companies = [a.strip() for a in is_startup.split(",")]
                    for comp in companies:
                        comp_summary = get_groq_summarize_startup(comp, story_content_str)
                        if 'no information' not in comp_summary.lower():
                            all_info = dict()
                            all_info['company'] = comp
                            all_info['newsstory'] = story_content_str
                            all_info['link'] = link_href
                            all_info['description'] = comp_summary
                            with open("./all_news_stories.jsonl", 'a') as f:
                                f.write(json.dumps(all_info) + "\n")