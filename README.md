🏎️ F1 Podium Predictor — Big Data Project

A machine learning application that predicts whether a Formula 1 driver will finish on the podium (Top 3) based on race parameters.

Built with Apache Spark ML + Streamlit, fully containerized with Docker.



Tech Stack

LayerTechnologyBig Data ProcessingApache Spark 3.5.0 (PySpark)ML AlgorithmRandom Forest ClassifierFrontendStreamlitContainerizationDocker + Docker ComposeLanguagePython 3.10



Dataset

Formula 1 World Championship (1950–2024) — available on Kaggle



26,759 race records

861 drivers, 211 constructors, 77 circuits

Binary classification: Podium (1) vs No Podium (0)





Model Performance



Algorithm: Random Forest (150 trees, depth 8)

AUC Score: 87.9%

Features: Grid position, driver historical podium rate, constructor podium rate, circuit, F1 era





Project Structure

f1\_predictor/

├── Dockerfile

├── docker-compose.yml

├── app/

│   ├── train.py          # Spark ML pipeline — data prep, training, model save

│   └── streamlit\_app.py  # Streamlit prediction UI

├── data/                 # Place Kaggle CSV files here (not tracked by git)

└── model/                # Auto-generated after training (not tracked by git)



How to Run

1\. Clone the repo

bashgit clone https://github.com/YOUR\_USERNAME/f1-podium-predictor.git

cd f1-podium-predictor

2\. Download the dataset

Download from Kaggle and place the CSV files inside the data/ folder.

3\. Build and start Docker

bashdocker-compose up --build

4\. Train the model

bashdocker exec -it f1\_predictor bash

python /app/train.py

5\. Launch the app

bashstreamlit run /app/streamlit\_app.py --server.address=0.0.0.0 --server.port=8501

Open your browser at: http://localhost:8501

