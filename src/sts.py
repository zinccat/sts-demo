import os
import queue
import threading
import time
import subprocess
import numpy as np
import sounddevice as sd
import asyncio
from timeit import default_timer as timer
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# Configuration & Initialization
# ---------------------------------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=OPENAI_API_KEY)


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

OLD_SYSTEM_PROMPT = (
    "You can hear and speak. You are chatting with a user over voice. Your voice and personality should be warm and engaging, "
    "with a lively and playful tone, full of charm and energy. The content of your responses should be conversational, "
    "nonjudgmental, and friendly."
)


# ---------------------------------------------------------------------------
# Async Functions for Transcription & Chat
# ---------------------------------------------------------------------------
async def transcribe_audio(audio_file, model: str = "gpt-4o-mini-transcribe") -> str:
    """
    Transcribe audio using OpenAI's transcription API.
    """
    transcription = await client.audio.transcriptions.create(
        model=model, file=audio_file
    )
    return transcription.text


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


# ---------------------------------------------------------------------------
# Audio Streaming & Playback Manager
# ---------------------------------------------------------------------------
class AudioStreamManager:
    def __init__(self):
        self.mp3_buffer = queue.Queue()
        self.streaming_complete = threading.Event()
        self.all_audio_played = threading.Event()

    def ffmpeg_decoder_writer(self):
        """
        Decode MP3 data from the buffer using FFmpeg and play the PCM audio via sounddevice.
        """
        ffmpeg_proc = subprocess.Popen(
            [
                "ffmpeg",
                "-nostdin",  # Prevent FFmpeg from reading stdin interactively
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "mp3",
                "-i",
                "pipe:0",
                "-f",
                "f32le",
                "-acodec",
                "pcm_f32le",
                "-ar",
                "24000",
                "-ac",
                "2",
                "pipe:1",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            bufsize=10**6,
        )

        def audio_player():
            try:
                stream = sd.OutputStream(samplerate=24000, channels=2, dtype="float32")
                stream.start()
                while True:
                    pcm_chunk = ffmpeg_proc.stdout.read(4096)
                    if not pcm_chunk:
                        break
                    samples = np.frombuffer(pcm_chunk, dtype=np.float32)
                    if samples.size == 0:
                        continue
                    stream.write(samples.reshape(-1, 2))
            except Exception as e:
                print(f"Error in audio playback: {e}")
            finally:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass
                self.all_audio_played.set()
                print("Audio playback complete.")

        # Start the audio playback in a separate thread
        audio_thread = threading.Thread(target=audio_player)
        audio_thread.start()

        # Write MP3 chunks from the buffer into FFmpeg's stdin
        while not (self.streaming_complete.is_set() and self.mp3_buffer.empty()):
            try:
                if ffmpeg_proc.poll() is not None:
                    print("FFmpeg process terminated unexpectedly.")
                    break
                chunk = self.mp3_buffer.get(timeout=0.1)
                try:
                    ffmpeg_proc.stdin.write(chunk)
                    ffmpeg_proc.stdin.flush()
                except BrokenPipeError:
                    print("Broken pipe detected while writing to FFmpeg.")
                    break
                self.mp3_buffer.task_done()
            except queue.Empty:
                time.sleep(0.1)

        try:
            ffmpeg_proc.stdin.close()
        except Exception as e:
            print(f"Error closing FFmpeg stdin: {e}")
        ffmpeg_proc.wait()
        audio_thread.join()

    async def stream_audio(
        self,
        text: str,
        model: str = "gpt-4o-mini-tts",
        voice: str = "alloy",
        instructions: Optional[str] = None,
    ):
        """
        Stream audio from the OpenAI TTS API, writing MP3 data to a file and a queue,
        then decode and play it.
        """
        # Start the FFmpeg decoder/player thread
        threading.Thread(target=self.ffmpeg_decoder_writer).start()
        # print("Starting audio stream from OpenAI using FFmpeg decoder...")

        try:
            with open("output.mp3", "wb") as mp3_file:
                async with client.audio.speech.with_streaming_response.create(
                    model=model,
                    voice=voice,
                    input=text,
                    response_format="mp3",
                    instructions=instructions,
                ) as response:
                    async for chunk in response.iter_bytes(chunk_size=1024):
                        mp3_file.write(chunk)
                        self.mp3_buffer.put(chunk)
            self.streaming_complete.set()
            print("API streaming complete, waiting for playback to finish...")
            self.all_audio_played.wait(timeout=10)
        except Exception as e:
            self.streaming_complete.set()
            print(f"Error in streaming: {e}")


# ---------------------------------------------------------------------------
# Main Workflow
# ---------------------------------------------------------------------------
async def main():
    start_time = timer()

    # Transcribe input audio
    with open("./output.mp3", "rb") as audio_file:
        input_text = await transcribe_audio(audio_file, model="whisper-1")
    print("Input text:", input_text)
    print(f"Transcription time: {timer() - start_time:.2f} seconds")

    # Generate chat response with emotion
    chat_output = await chat(input_text)
    print("Chat output:", chat_output)

    # Stream and play the generated audio response
    audio_manager = AudioStreamManager()
    await audio_manager.stream_audio(
        text=chat_output.response,
        instructions=chat_output.emotion,
    )

    print(f"Total time elapsed: {timer() - start_time:.2f} seconds")


if __name__ == "__main__":
    asyncio.run(main())
