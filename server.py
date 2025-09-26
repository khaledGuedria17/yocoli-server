import os
import uuid
import shutil
import asyncio
import imageio
import ffmpeg
from rembg import new_session, remove
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ✅ Initialize rembg ONCE with GPU session
session = new_session("u2net_human_seg")  # "u2net" also works


async def extract_frames(video_path: str, queue: asyncio.Queue):
    """Extract frames with ffmpeg and push them into a queue."""
    process = (
        ffmpeg
        .input(video_path)
        .filter("fps", fps=10)
        .filter("scale", 320, -1)
        .output("pipe:", format="image2pipe", vcodec="png")
        .run_async(pipe_stdout=True, pipe_stderr=True)
    )

    while True:
        # Read PNG headers from ffmpeg pipe
        in_bytes = process.stdout.read(4096)
        if not in_bytes:
            break
        await queue.put(in_bytes)

    await queue.put(None)  # Signal end of stream


def process_frame(raw_bytes: bytes):
    """Remove background on a single frame (GPU accelerated)."""
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGBA")
    return remove(img, session=session)


async def consumer(queue: asyncio.Queue, frames: list, executor):
    """Consume frames, clean background, and append to list."""
    loop = asyncio.get_event_loop()
    while True:
        raw_bytes = await queue.get()
        if raw_bytes is None:
            break
        frame = await loop.run_in_executor(executor, process_frame, raw_bytes)
        frames.append(frame)


@app.post("/process-video/")
async def process_video(file: UploadFile = File(...)):
    # Generate unique ID for job
    job_id = str(uuid.uuid4())
    work_dir = os.path.join(UPLOAD_DIR, job_id)
    os.makedirs(work_dir, exist_ok=True)

    # Save uploaded video
    video_path = os.path.join(work_dir, file.filename)
    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Shared structures
    queue = asyncio.Queue(maxsize=10)  # backpressure control
    frames: list = []

    # Run producer/consumer concurrently
    executor = ThreadPoolExecutor(max_workers=os.cpu_count())
    producer = extract_frames(video_path, queue)
    consumer_task = consumer(queue, frames, executor)

    await asyncio.gather(producer, consumer_task)

    # 3. Make GIF from processed frames
    gif_path = os.path.join(work_dir, "output.gif")
    imageio.mimsave(
        gif_path, frames, duration=0.12, loop=0, disposal=2
    )

    return FileResponse(gif_path, media_type="image/gif", filename="output.gif")
