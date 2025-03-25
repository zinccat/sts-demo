import os
import queue
import threading
import time
import subprocess
import numpy as np
import sounddevice as sd
import asyncio
from typing import Optional
from dotenv import load_dotenv
import gradio as gr
from openai import AsyncOpenAI
from shutil import copyfile

from src.sts import chat

# ---------------------------------------------------------------------------
# Configuration & Initialization
# ---------------------------------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=OPENAI_API_KEY)


# ---------------------------------------------------------------------------
# Data Models & Constants
# ---------------------------------------------------------------------------


async def transcribe_audio(audio_file, model: str = "whisper-1") -> str:
    """
    Transcribe audio using OpenAI's transcription API.
    """
    with open(audio_file, "rb") as file:
        transcription = await client.audio.transcriptions.create(model=model, file=file)
    return transcription.text


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
        model: str = "tts-1",
        voice: str = "alloy",
        instructions: Optional[str] = None,
    ):
        """
        Stream audio from the OpenAI TTS API, writing MP3 data to a queue,
        then decode and play it.
        """
        # Start the FFmpeg decoder/player thread
        threading.Thread(target=self.ffmpeg_decoder_writer).start()

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
            self.all_audio_played.wait(timeout=10)
            return "output.mp3"
        except Exception as e:
            self.streaming_complete.set()
            print(f"Error in streaming: {e}")
            return None


# ---------------------------------------------------------------------------
# Main Functions
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Gradio Interface Functions
# ---------------------------------------------------------------------------
async def process_audio(audio: str):
    gr.Info("Transcribing Audio", duration=5)
    # copy audio file to input.wav
    copyfile(audio, "input.wav")

    # Transcribe the audio
    input_text = await transcribe_audio("input.wav")
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


async def generate_audio(answer, emotion):
    if not answer:
        return None

    # Create audio stream manager
    audio_manager = AudioStreamManager()

    # Stream the audio
    audio_path = await audio_manager.stream_audio(text=answer, instructions=emotion)

    return audio_path


# Create synchronous wrapper functions for Gradio
def process_audio_sync(audio):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(process_audio(audio))
    loop.close()
    return result


def generate_audio_sync(answer, emotion):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(generate_audio(answer, emotion))
    loop.close()
    return result, answer, emotion


# ---------------------------------------------------------------------------
# Gradio Interface
# ---------------------------------------------------------------------------
with gr.Blocks() as block:
    gr.HTML(
        """
        <h1 style='text-align: center;'>Voice Chat with Dynamic Emotions</h1>
        <h3 style='text-align: center;'>Speak to the AI assistant and receive emotionally expressive responses</h3>
        <p style='text-align: center;'>Powered by OpenAI APIs</p>
        """
    )

    with gr.Group():
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
                audio_out = gr.Audio(label="Assistant's Voice Response", autoplay=True)

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
        process_audio_sync, audio_in, [answer, emotion, audio_out]
    ).then(generate_audio_sync, [answer, emotion], [audio_out, answer, emotion])

# Launch the Gradio interface
block.launch()
