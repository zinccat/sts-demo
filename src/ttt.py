import os
from openai import OpenAI
from dotenv import load_dotenv

# Load the API key from the environment
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def chat(inputs: str) -> str:
    completion = client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": inputs}]
    )
    return completion.choices[0].message.content


if __name__ == "__main__":
    inputs = "Hi, this is Bill. I'm calling to let you know that your order has been delayed."
    print(chat(inputs))
