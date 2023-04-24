from docarray import DocList
from docarray.documents import TextDoc
from docarray.typing import NdArray
from docarray.utils.find import find
from docarray.utils.map import map_docs
from gensim.models import KeyedVectors

model_path = 'model/GoogleNews-vectors-negative300.bin.gz'
print('Loading model')
model = KeyedVectors.load_word2vec_format(model_path, binary=True)
print('Model loaded')

print(type(model['apple']))

import numpy as np


def check_same_embeddings(embedding1, embedding2):
    if np.array_equal(embedding1, embedding2):
        return True
    else:
        return False


class WordDoc(TextDoc):
    embedding: NdArray[300] | None


def encode_word2vec(doc):
    if doc.text in model:
        print(f'Encoding {doc.text}')
        embedding = model[doc.text]
        doc.embedding = embedding

    return doc


docs = DocList([WordDoc(text='apple'), WordDoc(text='banana')])

encoded_docs = DocList[WordDoc](
    list(map_docs(docs, encode_word2vec, show_progress=True))
)

print([doc for doc in encoded_docs])

query = WordDoc(text='pear')
query = encode_word2vec(query)

results = find(
    query=query, index=encoded_docs, search_field='embedding', limit=2
)

print(results)
