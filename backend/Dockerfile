# Create our image based on Python 3.9
FROM python:3.11.5-slim
#FROM cr.weaviate.io/semitechnologies/weaviate:1.23.0-rc.1

# Expose ports
EXPOSE 5050

# Tell Python to not generate .pyc
ENV PYTHONDONTWRITEBYTECODE 1

# Turn off buffering
ENV PYTHONUNBUFFERED 1

# Install requirements using pip
ADD requirements.txt .
RUN python3 -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt

# Set working directory and addour Flask API files
WORKDIR /app
ADD . /app

CMD ["python", "app.py"]