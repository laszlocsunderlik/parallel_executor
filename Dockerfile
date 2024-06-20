FROM python:3.11

# Install system dependencies
RUN apt-get update && \
    apt-get install -y libgdal-dev && \
    apt-get clean

COPY requirements.txt /requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app

COPY main.py models.py /app/

RUN mkdir /app/data

CMD ["python", "/app/main.py"]
