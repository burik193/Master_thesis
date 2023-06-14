# syntax=docker/dockerfile:1

FROM python:3.9-slim-buster

WORKDIR /inference

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

COPY . .

CMD ["jupyter", "notebook", "-ip", "0.0.0.0", "--no-browser", "--allow-root"]