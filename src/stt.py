import os
from openai import OpenAI
from dotenv import load_dotenv

# Load the API key from the environment
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def transcribe_audio(audio_file):
    transcription = client.audio.transcriptions.create(
        model="gpt-4o-mini-transcribe", file=audio_file
    )
    return transcription.text


if __name__ == "__main__":
    audio_file = open("./output.mp3", "rb")
    print(transcribe_audio(audio_file))

# stream = client.audio.transcriptions.create(
#   model="gpt-4o-mini-transcribe",
#   file=audio_file,
#   response_format="text",
#   stream=True
# )

# for event in stream:
#     if event.type == 'transcript.text.delta':
#         print(event.delta, end="")
#     # elif event.type == 'transcript.text.done':
#     #     print(event.text, end="")
#     #     break
