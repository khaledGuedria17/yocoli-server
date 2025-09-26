# =========================
# 1. Base image with CUDA
# =========================
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04


# =========================
# 2. System setup
# =========================
ENV DEBIAN_FRONTEND=noninteractive

# Install basic tools + ffmpeg
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    ffmpeg git wget curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# =========================
# 3. Python dependencies
# =========================
WORKDIR /app

COPY requirements.txt .

# Install requirements
RUN pip3 install --no-cache-dir -r requirements.txt

# =========================
# 4. App code
# =========================
COPY . .

# =========================
# 5. Expose & run
# =========================
EXPOSE 8000

# Use uvicorn as server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
