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

# Load the API key from the environment
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Create a queue to hold incoming MP3 chunks
mp3_buffer = queue.Queue()
# Events to coordinate shutdown of the threads
streaming_complete = threading.Event()
all_audio_played = threading.Event()


def ffmpeg_decoder_writer():
    """
    Launches an FFmpeg subprocess to decode MP3 data from stdin into 32-bit float PCM output.
    Feeds the decoded PCM data to a sounddevice output stream in a dedicated playback thread.
    """
    # Start FFmpeg to read from stdin (-i pipe:0) and output raw PCM to stdout.
    ffmpeg_proc = subprocess.Popen(
        [
            "ffmpeg",
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
        bufsize=10**8,
    )

    def audio_player():
        """
        Reads raw PCM data from FFmpeg's stdout and writes it to a sounddevice output stream.
        """
        # Configure sounddevice with sample rate 24000 Hz and 2 channels.
        stream = sd.OutputStream(samplerate=24000, channels=2, dtype="float32")
        stream.start()
        try:
            while True:
                # Read a fixed-size PCM chunk (adjust this value if needed)
                pcm_chunk = ffmpeg_proc.stdout.read(4096)
                if not pcm_chunk:
                    break
                # Convert bytes to numpy array of float32 samples
                samples = np.frombuffer(pcm_chunk, dtype=np.float32)
                if samples.size == 0:
                    continue
                # Reshape for stereo output (2 channels)
                samples = samples.reshape(-1, 2)
                stream.write(samples)
        finally:
            stream.stop()
            stream.close()
            all_audio_played.set()
            print("Audio playback complete.")

    # Start the audio playback thread
    player_thread = threading.Thread(target=audio_player, daemon=True)
    player_thread.start()

    # Feed MP3 chunks from the mp3_buffer into FFmpeg's stdin
    while not (streaming_complete.is_set() and mp3_buffer.empty()):
        try:
            chunk = mp3_buffer.get(timeout=0.1)
            ffmpeg_proc.stdin.write(chunk)
            ffmpeg_proc.stdin.flush()
            mp3_buffer.task_done()
        except queue.Empty:
            time.sleep(0.1)

    # Signal end-of-stream to FFmpeg and wait for it to finish
    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()


def stream_audio(
    text: str,
    model: str = "gpt-4o-mini-tts",
    voice: str = "alloy",
    instructions: Optional[str] = None,
):
    """
    Streams audio from the OpenAI TTS API, writes the MP3 chunks to a file, puts them onto a queue,
    and starts the FFmpeg decoder.
    """
    # Start the FFmpeg-based decoder in its own thread
    decoder_thread = threading.Thread(target=ffmpeg_decoder_writer, daemon=True)
    decoder_thread.start()

    print("Starting audio stream from OpenAI using FFmpeg decoder...")
    try:
        # Open file for saving the MP3 data
        with open("output.mp3", "wb") as mp3_file:
            # Create a streaming response from the TTS API
            with client.audio.speech.with_streaming_response.create(
                model=model,
                voice=voice,
                input=text,
                response_format="mp3",
                instructions=instructions,
            ) as response:
                # For each MP3 chunk received from the API, write to file and put it on the queue.
                for chunk in response.iter_bytes(chunk_size=1024):
                    mp3_file.write(chunk)
                    mp3_buffer.put(chunk)
        # Once streaming is complete, set the event flag.
        streaming_complete.set()
        print("API streaming complete, waiting for playback to finish...")
        # Wait for the playback thread to finish (or timeout after 30 seconds)
        all_audio_played.wait(timeout=30)
    except Exception as e:
        streaming_complete.set()
        print(f"Error in streaming: {e}")


if __name__ == "__main__":
    try:
        stream_audio(
            text="Hello, world! This is a test of the OpenAI TTS API.",
            instructions="Speak in a angry tone.",
        )
        print("Audio streaming process finished")
    except KeyboardInterrupt:
        print("\nStreaming interrupted by user")
    except Exception as e:
        print(f"An error occurred: {e}")
