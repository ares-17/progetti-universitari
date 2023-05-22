FROM python:3.8

WORKDIR /app
RUN pip install --upgrade pip
RUN pip install opencv-python-headless
RUN pip install matplotlib keras tensorflow
COPY . .
CMD [ "python3", "./main.py" ]