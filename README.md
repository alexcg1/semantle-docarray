# Semantle-DocArray

This is a simple game that leverages [DocArray v2]() and the [OpenAI embeddings API](), similar to [Semantle]().

## How do I play?

1. `pip install -r requirements.txt`
2. Set your `OPENAPI_API_KEY` environment variable.
3. `python game.py`
4. A target word will be chosen from the top 10,000 words in English.
5. Your task is to guess the target. You'll be given an input field.
6. If you get it wrong, you'll see how far off your are. The higher the number, the further away you are.
7. When you guess correctly you win the game.

## FAQ

### 
