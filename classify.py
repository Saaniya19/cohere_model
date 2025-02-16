import os
from dotenv import load_dotenv
import cohere

load_dotenv()
api_key = os.getenv("COHERE_API_KEY")
co = cohere.ClientV2(api_key)

examples = [
    {"text": "This product is amazing! I love it.", "label": "Positive"},
    {"text": "Terrible experience, never buying again.", "label": "Negative"},
    {"text": "It was okay, nothing special.", "label": "Neutral"},
    {"text": "Great quality for the price!", "label": "Positive"},
    {"text": "Horrible customer service.", "label": "Negative"},
]

inputs = [
    "I absolutely love this phone, it's fantastic!",
    "The food was awful, I wouldn't recommend it.",
    "It's an average experience, nothing too great or bad.",
]

response = co.classify(
    model="embed-english-v3.0",
    inputs=inputs,
    examples=examples
)

for i, classification in enumerate(response.classifications):
    print(f"Input: {inputs[i]}")
    print(f"Predicted Label: {classification.prediction}")
    print(f"Confidence Scores: {classification.confidence}")
    print("-" * 40)
