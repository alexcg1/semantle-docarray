import os

import numpy as np
import openai
from docarray import DocList
from docarray.documents import TextDoc
from docarray.typing import NdArray
from dotenv import load_dotenv

load_dotenv()


class WordDoc(TextDoc):
    embedding: NdArray[300] | None


def wordlist_to_doclist(filename: str):
    docs = DocList[TextDoc]()
    with open(filename, 'r') as file:
        for line in file:
            doc = TextDoc(text=line.strip())
            docs.append(doc)

    return docs


def encode_word2vec(doc, model_path):
    if doc.text in model:
        # print(f'Finding vector for {doc.text}')
        embedding = model[doc.text]
        doc.embedding = embedding

    return doc


def gpt_encode(doc, model_name='text-embedding-ada-002'):
    openai.api_key = os.environ['OPENAI_API_KEY']
    response = openai.Embedding.create(input=doc.text, model=model_name)
    doc.embedding = response['data'][0]['embedding']

    return doc


def check_same_embeddings(embedding1, embedding2):
    if np.array_equal(embedding1, embedding2):
        return True
    else:
        return False