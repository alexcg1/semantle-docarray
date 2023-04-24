import os

import numpy as np
import openai
from docarray import DocList
from docarray.documents import TextDoc
from docarray.typing import NdArray
from dotenv import load_dotenv

if os.path.isfile('.env'):
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


def get_word_temp(distance, hot=0.1, warm=0.2):
    if distance <= hot:
        return 'Hot'
    elif hot < distance < warm:
        return 'Warm'
    else:
        return 'Cold'


def get_hint(doc, model_name='text-davinci-003'):
    response = openai.Completion.create(
        model=model_name,
        prompt=f'Create a crossword clue for the word "{doc.text}". You will give a brief dictionary definition of the word, but not say the word itself.',
        max_tokens=100,
        temperature=0.3,
    )

    hint = response.choices[0].text.strip()

    return hint
