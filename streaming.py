import os
from dotenv import load_dotenv
import cohere

load_dotenv()
api_key = os.getenv("COHERE_API_KEY")
co = cohere.ClientV2(api_key)

# chat_stream is best for chatbot responses
# tokens are streamed (ie. one token sent one at a time so model starts generating tokens immediately)
response = co.chat_stream(
    model="command-r-plus-08-2024",
    messages=[{"role": "user", "content": "what's the capital of Canada?"}],
)

for event in response:
    if event.type == "content-delta":
        print(event.delta.message.content.text, end="")