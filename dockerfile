# CPU-only version (smallest and cheapest for Azure)
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

WORKDIR /app
COPY requirements.txt .

# Install only CPU version of PyTorch (smaller)
RUN pip uninstall -y torch torchvision torchaudio
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install Flask and other dependencies
RUN pip install --no-cache-dir flask
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 5000

ENV FLASK_APP=app_openai.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_ENV=production

CMD ["flask", "run"]