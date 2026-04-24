FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y     ffmpeg     libgl1     libglib2.0-0     git     && rm -rf /var/lib/apt/lists/*

COPY requirements_worker.txt /app/requirements_worker.txt
RUN pip install --no-cache-dir -r /app/requirements_worker.txt

COPY handler.py /app/handler.py
COPY pipeline_module.py /app/pipeline_module.py

CMD ["python", "-c", "import handler; print('Runpod worker image ready')"]
