import os
from dotenv import load_dotenv
import cohere

load_dotenv()
api_key = os.getenv("COHERE_API_KEY")
co = cohere.Client(api_key)


# generate is best for summarization, content generation, code generation, etc.
# model generates the entire output internally and sends the output back in one go
response = co.generate(
    model ='command-light',
    prompt = """"Summarize this dialogue:
    Once upon a time there was an old mother pig who had three little pigs and not enough food to feed them. So when they were old enough, she sent them out into the world to seek their fortunes.

    The first little pig was very lazy. He didn't want to work at all and he built his house out of straw. The second little pig worked a little bit harder but he was somewhat lazy too and he built his house out of sticks. Then, they sang and danced and played together the rest of the day.

    The third little pig worked hard all day and built his house with bricks. It was a sturdy house complete with a fine fireplace and chimney. It looked like it could withstand the strongest winds.

    The next day, a wolf happened to pass by the lane where the three little pigs lived; and he saw the straw house, and he smelled the pig inside. He thought the pig would make a mighty fine meal and his mouth began to water.

    So he knocked on the door and said:

    Little pig! Little pig!
    Let me in! Let me in!
    But the little pig saw the wolf's big paws through the keyhole, so he answered back:

    No! No! No! 
    Not by the hairs on my chinny chin chin!
    Three Little Pigs, the straw houseThen the wolf showed his teeth and said:

    Then I'll huff 
    and I'll puff 
    and I'll blow your house down.
    So he huffed and he puffed and he blew the house down! The wolf opened his jaws very wide and bit down as hard as he could, but the first little pig escaped and ran away to hide with the second little pig.

    The wolf continued down the lane and he passed by the second house made of sticks; and he saw the house, and he smelled the pigs inside, and his mouth began to water as he thought about the fine dinner they would make.
    --
    """,
    max_tokens=50,
    temperature=0.6,
    k=0,
    p=1,
    frequency_penalty=0,
    presence_penalty=0,
    stop_sequences=["--"], #stop generating when this sequence is reached
)

print('Summarization: {}'.format(response.generations[0].text))