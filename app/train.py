from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, concat_ws
from pyspark.ml.feature import VectorAssembler, FeatureHasher
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import json, shutil, os

# ── 1. Start Spark ──────────────────────────────────────────────
spark = SparkSession.builder \
    .appName("F1PodiumPredictor") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
print("✅ Spark started")

# ── 2. Load CSVs ─────────────────────────────────────────────────
BASE = "/data/Formula 1 World Championship Dataset (1950 - 2024)"

results      = spark.read.csv(f"{BASE}/results.csv",      header=True, inferSchema=True)
qualifying   = spark.read.csv(f"{BASE}/qualifying.csv",   header=True, inferSchema=True)
drivers      = spark.read.csv(f"{BASE}/drivers.csv",      header=True, inferSchema=True)
constructors = spark.read.csv(f"{BASE}/constructors.csv", header=True, inferSchema=True)
races        = spark.read.csv(f"{BASE}/races.csv",        header=True, inferSchema=True)
circuits     = spark.read.csv(f"{BASE}/circuits.csv",     header=True, inferSchema=True)

print("✅ CSVs loaded")

# ── 3. Build target column ───────────────────────────────────────
results = results.withColumn(
    "podium",
    when(col("positionOrder") <= 3, 1).otherwise(0)
)

# ── 4. Build driver full name ────────────────────────────────────
drivers = drivers.withColumn(
    "driver_name",
    concat_ws(" ", col("forename"), col("surname"))
)

# ── 5. Join all tables ───────────────────────────────────────────
df = results \
    .join(qualifying.select("raceId", "driverId", "position")
          .withColumnRenamed("position", "grid_position"),
          on=["raceId", "driverId"], how="left") \
    .join(drivers.select("driverId", "driver_name"),
          on="driverId", how="left") \
    .join(constructors.select("constructorId", "name")
          .withColumnRenamed("name", "constructor_name"),
          on="constructorId", how="left") \
    .join(races.select("raceId", "year", "round", "circuitId"),
          on="raceId", how="left") \
    .join(circuits.select("circuitId", "name")
          .withColumnRenamed("name", "circuit_name"),
          on="circuitId", how="left")

print("✅ Tables joined")

# ── 6. Select & clean ────────────────────────────────────────────
df = df.select(
    "podium", "grid", "year", "round",
    "driver_name", "constructor_name", "circuit_name"
).dropna()

df = df.withColumn("grid",   col("grid").cast("double")) \
       .withColumn("year",   col("year").cast("double")) \
       .withColumn("round",  col("round").cast("double")) \
       .withColumn("podium", col("podium").cast("int"))

# ── 7. Fix class imbalance with oversampling ─────────────────────
podium_df     = df.filter(col("podium") == 1)
no_podium_df  = df.filter(col("podium") == 0)

podium_count    = podium_df.count()
no_podium_count = no_podium_df.count()
ratio = no_podium_count / podium_count

print(f"✅ Class distribution — Podium: {podium_count} | No Podium: {no_podium_count} | Ratio: {ratio:.1f}x")

# Oversample the minority class (podium) to balance
podium_oversampled = podium_df.sample(
    withReplacement=True,
    fraction=ratio * 0.8,
    seed=42
)

balanced_df = no_podium_df.union(podium_oversampled)
balanced_df = balanced_df.sample(withReplacement=False, fraction=1.0, seed=42)

podium_final    = balanced_df.filter(col("podium") == 1).count()
no_podium_final = balanced_df.filter(col("podium") == 0).count()
print(f"✅ Balanced dataset — Podium: {podium_final} | No Podium: {no_podium_final}")
print(f"✅ Total rows: {balanced_df.count()}")

# ── 8. Save lookup lists ─────────────────────────────────────────
driver_list      = sorted([r[0] for r in df.select("driver_name").distinct().collect() if r[0]])
constructor_list = sorted([r[0] for r in df.select("constructor_name").distinct().collect() if r[0]])
circuit_list     = sorted([r[0] for r in df.select("circuit_name").distinct().collect() if r[0]])

with open("/model/lookups.json", "w") as f:
    json.dump({
        "drivers":      driver_list,
        "constructors": constructor_list,
        "circuits":     circuit_list
    }, f)

print(f"✅ Lookups saved: {len(driver_list)} drivers | {len(constructor_list)} constructors | {len(circuit_list)} circuits")

# ── 9. Feature engineering ───────────────────────────────────────
hasher = FeatureHasher(
    inputCols=["driver_name", "constructor_name", "circuit_name"],
    outputCol="cat_features",
    numFeatures=1024
)

num_assembler = VectorAssembler(
    inputCols=["grid", "year", "round"],
    outputCol="num_features"
)

final_assembler = VectorAssembler(
    inputCols=["num_features", "cat_features"],
    outputCol="features"
)

# ── 10. Model ────────────────────────────────────────────────────
rf = RandomForestClassifier(
    labelCol="podium",
    featuresCol="features",
    numTrees=150,
    maxDepth=8,
    maxBins=64,
    seed=42
)

# ── 11. Pipeline ─────────────────────────────────────────────────
pipeline = Pipeline(stages=[
    hasher,
    num_assembler,
    final_assembler,
    rf
])

# ── 12. Train / test split ───────────────────────────────────────
train_df, test_df = balanced_df.randomSplit([0.8, 0.2], seed=42)
print(f"✅ Train: {train_df.count()} | Test: {test_df.count()}")

# ── 13. Train ────────────────────────────────────────────────────
print("⏳ Training model...")
model = pipeline.fit(train_df)
print("✅ Model trained!")

# ── 14. Evaluate ─────────────────────────────────────────────────
predictions = model.transform(test_df)

auc_evaluator = BinaryClassificationEvaluator(
    labelCol="podium",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)
acc_evaluator = MulticlassClassificationEvaluator(
    labelCol="podium",
    predictionCol="prediction",
    metricName="accuracy"
)

auc      = auc_evaluator.evaluate(predictions)
accuracy = acc_evaluator.evaluate(predictions)

print(f"✅ AUC Score : {auc:.4f}")
print(f"✅ Accuracy  : {accuracy:.4f}")

# ── 15. Save model ───────────────────────────────────────────────
if os.path.exists("/model/f1_podium_model"):
    shutil.rmtree("/model/f1_podium_model")

model.save("/model/f1_podium_model")
print("✅ Model saved!")

spark.stop()
print("✅ Done!")