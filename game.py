import random
import sys

from docarray.documents import TextDoc
from docarray.typing import NdArray
from scipy.spatial.distance import cosine, euclidean

from helper import (get_hint, get_word_temp, gpt_encode, welcome_string,
                    wordlist_to_doclist)

print(welcome_string)


class WordDoc(TextDoc):
    embedding: NdArray[1536] | None


all_words = wordlist_to_doclist('wordlists/10k.txt')

target_word = random.choice(all_words)
gpt_encode(target_word)
# print(target_word.text)

guess = WordDoc(text='')

counter = 0

while guess.text != target_word.text:
    counter += 1
    guess = WordDoc(text=input('What is your guess? '))

    if guess.text.lower() == '/hint':
        print(get_hint(target_word))
    elif guess.text.lower() == '/quit' or guess.text.lower() == '/exit':
        sys.exit()
    elif guess.text.lower() == '/answer':
        print(target_word.text)

    else:
        gpt_encode(guess)

        distance = cosine(target_word.embedding, guess.embedding)
        temperature = get_word_temp(distance)

        if guess.text != target_word.text:
            print(
                f'Try again. Your distance is {round(distance, 2)}. Your guess is {temperature}.'
            )

print(f'Congrats! You guessed in {counter} turns')
