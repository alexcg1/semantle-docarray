import random

from docarray import DocList
from docarray.documents import TextDoc
from docarray.typing import NdArray
from scipy.spatial.distance import cosine, euclidean

from helper import WordDoc, gpt_encode, wordlist_to_doclist


class WordDoc(TextDoc):
    embedding: NdArray[1536] | None


all_words = wordlist_to_doclist('wordlists/10k.txt')

target_word = random.choice(all_words)
gpt_encode(target_word)
print(target_word.text)

guess = WordDoc(text='')

counter = 0

while guess.text != target_word.text:
    counter += 1
    user_input = input('What is your guess? ')
    guess = WordDoc(user_input)
    guess = gpt_encode(guess)

    euc_distance = euclidean(target_word.embedding, guess.embedding)
    print(f'Try again. Your distance is {round(euc_distance, 2)}')

print(f'Congrats! You guessed in {counter} turns')
