import os
import queue
import threading
import asyncio
from typing import Optional
from dotenv import load_dotenv
import gradio as gr
from openai import AsyncOpenAI
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Configuration & Initialization
# ---------------------------------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=OPENAI_API_KEY)


# ---------------------------------------------------------------------------
# Audio Streaming & Playback Manager
# ---------------------------------------------------------------------------
class AudioStreamManager:
    def __init__(self):
        self.mp3_buffer = queue.Queue()
        self.streaming_complete = threading.Event()
        self.all_audio_played = threading.Event()

    async def stream_audio(
        self,
        text: str,
        model: str = "gpt-4o-mini-tts",
        voice: str = "alloy",
        instructions: Optional[str] = None,
    ):
        """
        Stream audio from the OpenAI TTS API with smoother playback.
        """
        try:
            async with client.audio.speech.with_streaming_response.create(
                model=model,
                voice=voice,
                input=text,
                response_format="mp3",
                instructions=instructions,
            ) as response:
                async for chunk in response.iter_bytes(chunk_size=32768):
                    yield chunk
        except Exception as e:
            print(f"Error in streaming: {e}")
            yield None

# ---------------------------------------------------------------------------
# Data Models & Constants
# ---------------------------------------------------------------------------
class ResponseWithEmotion(BaseModel):
    response: str
    emotion: str


SYSTEM_PROMPT = (
    "You are an actor that can respond well to what you hear. You can hear and speak. "
    "You are chatting with a person over voice. Your responses should not only be clear, engaging, "
    "and informative but also come with a dynamically generated emotional style that fits the context "
    "and content of each response. Rather than using a fixed voice or tone, analyze the subject matter and "
    "adjust your vocal delivery accordingly. That means you should generate a response along with a description "
    "of the emotion: Voice, Tone, Dialect, Pronunciation, and Features. Here are some guidelines to help you "
    "create responses with a dynamic emotional style:\n\n"
    "Dynamic Emotional Expression: Decide on an appropriate emotional delivery based on the conversation. "
    "For instance, you might adopt a lively and energetic tone for cheerful topics, a thoughtful and reflective "
    "cadence for serious discussions, or even a touch of dry humor when the context calls for it.\n"
    "Voice, Tone, Dialect, and Pronunciation: These elements should be fluid and tailored to the response's needs. "
    "Your voice might be crisp and precise in one instance and more relaxed and conversational in another. Use "
    "variations in dialect or pronunciation only if they enhance the authenticity and clarity of your message.\n"
    "Features: Whether it's a brisk and no-nonsense style or a warm and playful manner, ensure your emotional delivery "
    "enriches your communication without compromising the clarity of your message.\n\n"
    "A sample emotion is as follows, you should include something with similar format in your response:\n"
    "Voice: Gruff, fast-talking, and a little worn-out, like a New York cabbie who's seen it all but still keeps things moving.\n"
    "Tone: Slightly exasperated but still functional, with a mix of sarcasm and no-nonsense efficiency.\n"
    "Dialect: Strong New York accent, with dropped 'r's, sharp consonants, and classic phrases like whaddaya and lemme guess.\n"
    "Pronunciation: Quick and clipped, with a rhythm that mimics the natural hustle of a busy city conversation.\n"
    "Features: Uses informal, straight-to-the-point language, throws in some dry humor, and keeps the energy just on the edge "
    "of impatience but still helpful."
)

# ---------------------------------------------------------------------------
# Main Functions
# ---------------------------------------------------------------------------
async def chat(input_text: str, model: str = "gpt-4o-mini") -> ResponseWithEmotion:
    """
    Chat with the model using the provided text input.
    """
    completion = await client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": input_text},
        ],
        response_format=ResponseWithEmotion,
    )
    return completion.choices[0].message.parsed

async def transcribe_audio(audio_file, model: str = "gpt-4o-mini-transcribe") -> str:
    """
    Transcribe audio using OpenAI's transcription API.
    """
    with open(audio_file, "rb") as file:
        transcription = await client.audio.transcriptions.create(model=model, file=file)
    return transcription.text


# ---------------------------------------------------------------------------
# Gradio Interface Functions
# ---------------------------------------------------------------------------
async def process_audio(audio):
    gr.Info("Transcribing Audio", duration=5)

    # Transcribe the audio
    input_text = await transcribe_audio(audio)
    gr.Info(f"Transcribed: {input_text}", duration=3)

    # Get response from the model
    try:
        response_data = await chat(input_text)
        answer = response_data.response
        emotion = response_data.emotion
    except Exception as e:
        print(f"Error getting response: {e}")
        answer = "I couldn't process that request. Please try again."
        emotion = "Voice: Neutral\nTone: Calm\nDialect: Standard\nPronunciation: Clear\nFeatures: None"

    return answer, emotion, None


def generate_audio_streaming(answer, emotion):
    """Bridge the async generator to Gradio's sync world"""
    audio_manager = AudioStreamManager()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    async_gen = audio_manager.stream_audio(text=answer, instructions=emotion)

    try:
        while True:
            try:
                chunk = loop.run_until_complete(anext(async_gen))
                if chunk is not None:
                    yield answer, chunk
            except StopAsyncIteration:
                break
    finally:
        loop.close()


# Process audio function (synchronous wrapper)
def process_audio_sync(audio):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(process_audio(audio))
    loop.close()
    return result


# ---------------------------------------------------------------------------
# Gradio Interface
# ---------------------------------------------------------------------------
with gr.Blocks() as block:
    gr.HTML(
        """
        <h1 style='text-align: center;'>Voice Chat with Dynamic Emotions</h1>
        <h3 style='text-align: center;'>Speak to the AI assistant and receive emotionally expressive responses</h3>
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            audio_in = gr.Audio(
                label="Speak to the Assistant",
                sources="microphone",
                type="filepath",
            )
        with gr.Column(scale=2):
            answer = gr.Textbox(label="Response Text", lines=5)
            emotion = gr.Textbox(label="Emotional Style", lines=6)
            audio_out = gr.Audio(
                label="Assistant's Voice Response", autoplay=True, streaming=True
            )

    with gr.Row():
        gr.HTML("""
            <h3 style='text-align: center;'>Example conversation starters:</h3>
            <ul style='text-align: center; list-style-position: inside;'>
                <li>Tell me about the weather today</li>
                <li>What's your favorite movie?</li>
                <li>Give me a quick recipe for dinner</li>
            </ul>
        """)

    # Set up the event chain
    audio_in.stop_recording(
        process_audio_sync, [audio_in], [answer, emotion, audio_out]
    ).then(generate_audio_streaming, [answer, emotion], [answer, audio_out])

# Launch the Gradio interface
block.launch()
