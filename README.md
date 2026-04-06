# F1 Podium Predictor — Big Data Project

> A machine learning application that predicts whether a Formula 1 driver will finish on the **podium (Top 3)** based on race parameters — built with Apache Spark ML, Streamlit, and Docker.

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![Spark](https://img.shields.io/badge/Apache%20Spark-3.5.0-orange?style=flat-square&logo=apachespark)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red?style=flat-square&logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=flat-square&logo=docker)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Big Data Processing | Apache Spark 3.5.0 (PySpark) |
| ML Algorithm | Random Forest Classifier |
| Frontend | Streamlit |
| Containerization | Docker + Docker Compose |
| Language | Python 3.10 |

---

## Dataset

**Formula 1 World Championship (1950–2024)** — [Kaggle](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020)

- 26,759 race records
- 861 drivers, 211 constructors, 77 circuits
- Binary classification: Podium `1` vs No Podium `0`

---

## Model Performance

| Parameter | Value |
|---|---|
| Algorithm | Random Forest (150 trees, depth 8) |
| AUC Score | **87.9%** |
| Features | Grid position, driver & constructor podium rate, circuit, F1 era |

---

## Project Structure
```
f1_predictor/
├── Dockerfile
├── docker-compose.yml
├── app/
│   ├── train.py             # Spark ML pipeline — data prep, training, model save
│   └── streamlit_app.py     # Streamlit prediction UI
├── data/                    # Place Kaggle CSV files here (not tracked by git)
└── model/                   # Auto-generated after training (not tracked by git)
```

---

## How to Run

**1. Clone the repo**
```bash
git clone https://github.com/ali-harti/f1-podium-predictor.git
cd f1-podium-predictor
```

**2. Download the dataset**

Download from [Kaggle](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020) and place the CSV files inside the `data/` folder.

**3. Build and start Docker**
```bash
docker-compose up --build
```

**4. Train the model**
```bash
docker exec -it f1_predictor bash
python /app/train.py
```

**5. Launch the app**
```bash
streamlit run /app/streamlit_app.py --server.address=0.0.0.0 --server.port=8501
```

Open your browser at **[http://localhost:8501](http://localhost:8501)**
