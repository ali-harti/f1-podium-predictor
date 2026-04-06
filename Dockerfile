FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    default-jdk \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH=$PATH:$JAVA_HOME/bin

RUN pip install --no-cache-dir \
    pyspark==3.5.0 \
    streamlit==1.32.0 \
    pandas \
    numpy \
    plotly \
    scikit-learn

WORKDIR /app

EXPOSE 8501