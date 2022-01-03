# FROM ubuntu:latest

# RUN apt-get update
# RUN apt-get install ffmpeg libsm6 libxext6  -y 

# RUN set -xe \
#     && apt-get update -y \
#     && apt-get install python3-pip -y
# RUN pip3 install --upgrade pip


# COPY . /app

# WORKDIR /app

# RUN pip3 install -r requirements.txt 

# EXPOSE 7860

# CMD ["python3", "app.py"]

FROM python:3.7


COPY . /app

WORKDIR /app

RUN pip install -r requirements.txt 

EXPOSE 7860

CMD ["python", "app.py"]