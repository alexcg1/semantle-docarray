from docarray import DocList
from docarray.index import HnswDocumentIndex
from docarray.utils.map import map_docs
from gensim.models import KeyedVectors

from helper import WordDoc, wordlist_to_doclist

WORD_LIST = 'wordlists/fruits.txt'
WORKDIR = 'workspace/fruits'

model_path = 'model/GoogleNews-vectors-negative300.bin.gz'
print('Loading model')
print('Model loaded')
model = KeyedVectors.load_word2vec_format(model_path, binary=True)


def encode_word2vec(doc):
    if doc.text in model:
        # print(f'Finding vector for {doc.text}')
        embedding = model[doc.text]
        doc.embedding = embedding

    return doc


db = HnswDocumentIndex[WordDoc](work_dir=WORKDIR)

if db.num_docs() == 0:
    docs = wordlist_to_doclist(WORD_LIST)

    encoded_docs = DocList[WordDoc](
        list(map_docs(docs, encode_word2vec, show_progress=True))
    )

    db.index(encoded_docs)

query = WordDoc(text='Apple')
query = encode_word2vec(query)

matches, scores = db.find(query=query, search_field='embedding', limit=5)

# matches = find(
# query=query, index=encoded_docs, search_field='embedding', limit=5
# )

print(matches.text)
print(scores)

# for match in matches.documents:
# print(match.text)
