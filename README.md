# 🏎️ F1 Podium Predictor

[![Python](https://img.shields.io/badge/Python-3.10-3776AB.svg?logo=python)](https://www.python.org/)
[![Apache Spark](https://img.shields.io/badge/Apache_Spark-3.5.0-E25A1C.svg?logo=apachespark)](https://spark.apache.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B.svg?logo=streamlit)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg?logo=docker)](https://docker.com)

A Big Data machine learning application that predicts whether a Formula 1 driver will finish on the **podium (Top 3)** based on pre-race parameters. Built using Apache Spark MLlib, Streamlit, and Docker for scalable, distributed processing.

## ✨ Features

- **Big Data Processing**: Leverages PySpark to analyze and process 26,759 historic Formula 1 race records dating from 1950 to 2024.
- **Machine Learning**: Uses a distributed Random Forest Classifier (150 trees, depth 8) achieving an **87.9% AUC Score**.
- **Interactive UI**: A clean, responsive web interface built with Streamlit for testing predictions in real-time.
- **Feature Engineering**: Incorporates grid position, historical driver/constructor podium rates, circuit specifics, and F1 era categorization.
- **Containerized**: Fully packaged with Docker and Docker Compose for easy environment replication.

## 🚀 Quick Start

### Prerequisites

Ensure you have **Docker** and **Docker Compose** installed on your system.

### Installation

1. Clone the repository:
```bash
git clone https://github.com/ali-harti/f1-podium-predictor.git
cd f1-podium-predictor
```

2. Download the Dataset:
Download the dataset from [Kaggle](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020) and place all the CSV files inside the `data/` folder.

3. Build the Docker environment:
```bash
docker-compose up --build -d
```

### Training & Running the App

1. Access the container bash shell:
```bash
docker exec -it f1_predictor bash
```

2. Train the PySpark Model (this will generate the model artifacts):
```bash
python /app/train.py
```

3. Launch the Streamlit application:
```bash
streamlit run /app/streamlit_app.py --server.address=0.0.0.0 --server.port=8501
```

Open your browser at **http://localhost:8501** to start predicting!

## 🧠 Tech Stack & Architecture

- **Big Data Processing**: Apache Spark 3.5.0 (PySpark)
- **ML Algorithm**: Random Forest Classifier
- **Frontend**: Streamlit
- **Language**: Python 3.10
- **Infrastructure**: Docker

## 📂 Project Structure

```text
f1-podium-predictor/
├── app/
│   ├── train.py             # Spark ML pipeline (data prep, training, model saving)
│   └── streamlit_app.py     # Streamlit prediction UI
├── data/                    # Kaggle CSV files (Ignored by Git)
├── model/                   # Generated Spark Model artifacts (Ignored by Git)
├── Dockerfile               # Environment definition
├── docker-compose.yml       # Container orchestration
└── README.md                # Project documentation
```

## 📝 License

This project is open-source and available under the [MIT License](LICENSE).

**Developed by Ali Harti**
