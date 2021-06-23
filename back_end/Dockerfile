FROM python:3.8.6-buster

COPY main.py /main.py
COPY model.h5 /model.h5
COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip && pip install -r requirements.txt

CMD uvicorn main:app --host 0.0.0.0 --port $PORT