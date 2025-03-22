import os
import queue
import threading
import time
import subprocess
import numpy as np
import sounddevice as sd
from openai import OpenAI
from dotenv import load_dotenv
from typing import Optional
from timeit import default_timer as timer

# Initialize environment and OpenAI client
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Shared resources for MP3 streaming and playback
mp3_buffer = queue.Queue()
streaming_complete = threading.Event()
all_audio_played = threading.Event()


def transcribe_audio(audio_file, model="gpt-4o-mini-transcribe"):
    """Transcribe audio using the specified transcription model."""
    transcription = client.audio.transcriptions.create(model=model, file=audio_file)
    return transcription.text


def chat(inputs: str, model="gpt-4o-mini") -> str:
    """Chat with the model using the provided text input."""
    completion = client.chat.completions.create(
        model=model, messages=[{"role": "system", "content": '''You can hear and speak. You are chatting with a user over voice. Your voice and personality should be warm and engaging, with a lively and playful tone, full of charm and energy. The content of your responses should be conversational, nonjudgmental, and friendly.'''},{"role": "user", "content": inputs}]
    )
    return completion.choices[0].message.content


def ffmpeg_decoder_writer():
    """
    Use FFmpeg to decode MP3 data (from mp3_buffer) into 32-bit float PCM output,
    and play the decoded audio using sounddevice.
    """
    ffmpeg_proc = subprocess.Popen(
        [
            "ffmpeg",
            "-nostdin",             # Prevent ffmpeg from trying to read stdin interactively
            "-hide_banner",
            "-loglevel", "error",
            "-f", "mp3",
            "-i", "pipe:0",
            "-f", "f32le",
            "-acodec", "pcm_f32le",
            "-ar", "24000",
            "-ac", "2",
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
            all_audio_played.set()
            print("Audio playback complete.")

    # Start the audio playback thread (non-daemon so we can join later)
    audio_thread = threading.Thread(target=audio_player)
    audio_thread.start()

    # Write MP3 chunks from the buffer into ffmpeg's stdin
    while not (streaming_complete.is_set() and mp3_buffer.empty()):
        try:
            if ffmpeg_proc.poll() is not None:
                print("FFmpeg process terminated unexpectedly.")
                break
            chunk = mp3_buffer.get(timeout=0.1)
            try:
                ffmpeg_proc.stdin.write(chunk)
                ffmpeg_proc.stdin.flush()
            except BrokenPipeError:
                print("Broken pipe detected while writing to ffmpeg.")
                break
            mp3_buffer.task_done()
        except queue.Empty:
            time.sleep(0.1)

    try:
        ffmpeg_proc.stdin.close()
    except Exception as e:
        print(f"Error closing ffmpeg stdin: {e}")
    ffmpeg_proc.wait()
    audio_thread.join()


def stream_audio(
    text: str,
    model: str = "gpt-4o-mini-tts",
    voice: str = "alloy",
    instructions: Optional[str] = None,
):
    """
    Stream audio from the OpenAI TTS API, writing MP3 data to a file and to a queue,
    then decode and play it.
    """
    # Start the FFmpeg decoder/player thread
    threading.Thread(target=ffmpeg_decoder_writer).start()
    print("Starting audio stream from OpenAI using FFmpeg decoder...")

    try:
        with open("output.mp3", "wb") as mp3_file:
            with client.audio.speech.with_streaming_response.create(
                model=model,
                voice=voice,
                input=text,
                response_format="mp3",
                instructions=instructions,
            ) as response:
                for chunk in response.iter_bytes(chunk_size=1024):
                    mp3_file.write(chunk)
                    mp3_buffer.put(chunk)
        streaming_complete.set()
        print("API streaming complete, waiting for playback to finish...")
        all_audio_played.wait(timeout=30)
    except Exception as e:
        streaming_complete.set()
        print(f"Error in streaming: {e}")


if __name__ == "__main__":
    start = timer()
    with open("./output.mp3", "rb") as audio_file:
        input_text = transcribe_audio(audio_file, model="whisper-1")
    print(timer() - start)

    output_text = chat(input_text)
    print(output_text)

    end = timer()
    print(f"Time elapsed: {end - start} seconds")

    stream_audio(
        text=output_text,
        instructions="Speak in a calm and professional tone.",
    )
