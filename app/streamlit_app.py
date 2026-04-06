import streamlit as st
import pandas as pd
import json
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.types import StructType, StructField, DoubleType, StringType

# ── Page config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="F1 Podium Predictor",
    page_icon="🏎️",
    layout="wide"
)

# ── Premium CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Titillium+Web:wght@300;400;600;700;900&display=swap');

* { font-family: 'Titillium Web', sans-serif !important; }

.stApp { background: #080808; color: #ffffff; }

.hero {
    background: #e10600;
    padding: 2.5rem 3rem 2rem 3rem;
    margin: -1rem -1rem 2rem -1rem;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: 'F1';
    position: absolute;
    right: -20px;
    top: -30px;
    font-size: 12rem;
    font-weight: 900;
    color: rgba(0,0,0,0.12);
    letter-spacing: -10px;
    pointer-events: none;
}
.hero-label {
    font-size: 0.65rem;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: rgba(255,255,255,0.6);
    margin-bottom: 0.4rem;
}
.hero-title {
    font-size: 3rem;
    font-weight: 900;
    color: white;
    letter-spacing: 6px;
    text-transform: uppercase;
    line-height: 1;
    margin-bottom: 0.5rem;
}
.hero-sub {
    font-size: 0.75rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: rgba(255,255,255,0.55);
}

.stat-card {
    background: #111;
    border: 1px solid #1e1e1e;
    border-top: 2px solid #e10600;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.5rem;
}
.stat-label {
    font-size: 0.6rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #555;
    margin-bottom: 0.4rem;
}
.stat-value {
    font-size: 1.8rem;
    font-weight: 900;
    color: #fff;
    line-height: 1;
}
.stat-sub {
    font-size: 0.65rem;
    color: #e10600;
    letter-spacing: 1px;
    margin-top: 0.2rem;
}

.panel-title {
    font-size: 0.6rem;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: #e10600;
    font-weight: 700;
    margin-bottom: 1.2rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid #1a1a1a;
}

.result-win {
    background: #050f05;
    border: 1px solid #1a1a1a;
    border-left: 4px solid #00c851;
    padding: 2rem;
    text-align: center;
    margin-bottom: 1rem;
}
.result-loss {
    background: #0f0505;
    border: 1px solid #1a1a1a;
    border-left: 4px solid #e10600;
    padding: 2rem;
    text-align: center;
    margin-bottom: 1rem;
}
.result-verdict {
    font-size: 1.8rem;
    font-weight: 900;
    letter-spacing: 5px;
    text-transform: uppercase;
}
.result-desc {
    font-size: 0.8rem;
    letter-spacing: 1px;
    color: #666;
    margin-top: 0.5rem;
}

.prob-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.3rem;
}
.prob-label {
    font-size: 0.65rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    font-weight: 700;
}
.prob-pct { font-size: 1.4rem; font-weight: 900; }
.bar-track {
    background: #1a1a1a;
    height: 6px;
    margin-bottom: 1.2rem;
    overflow: hidden;
}
.bar-fill-green { height: 100%; background: #00c851; }
.bar-fill-red   { height: 100%; background: #e10600; }

.summary-row {
    display: flex;
    justify-content: space-between;
    padding: 0.6rem 0;
    border-bottom: 1px solid #151515;
    font-size: 0.8rem;
}
.summary-key {
    color: #555;
    letter-spacing: 1px;
    text-transform: uppercase;
    font-size: 0.65rem;
}
.summary-val { color: #fff; font-weight: 600; }

.placeholder {
    border: 1px dashed #1a1a1a;
    padding: 5rem 2rem;
    text-align: center;
    color: #2a2a2a;
}
.placeholder-text {
    font-size: 0.7rem;
    letter-spacing: 3px;
    text-transform: uppercase;
}

div[data-baseweb="select"] > div {
    background-color: #111 !important;
    border-color: #222 !important;
    color: white !important;
    border-radius: 0 !important;
}
div[data-baseweb="input"] > div {
    background-color: #111 !important;
    border-color: #222 !important;
    border-radius: 0 !important;
}
.stSlider > div > div > div { background: #e10600 !important; }
label {
    color: #555 !important;
    font-size: 0.65rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
}
.stButton > button {
    background: #e10600 !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 0.75rem !important;
    letter-spacing: 4px !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 0.9rem !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: #ff1a00 !important;
    box-shadow: 0 0 30px rgba(225,6,0,0.3) !important;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Hero ─────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-label">Powered by Apache Spark ML</div>
    <div class="hero-title">Podium Predictor</div>
    <div class="hero-sub">
        Formula 1 World Championship · 1950–2024 · 26,759 Race Records · Random Forest Model
    </div>
</div>
""", unsafe_allow_html=True)

# ── Load Spark & Model ───────────────────────────────────────────
@st.cache_resource
def load_spark_and_model():
    spark = SparkSession.builder \
        .appName("F1PodiumPredictor") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    model = PipelineModel.load("/model/f1_podium_model")
    return spark, model

@st.cache_data
def load_lookups():
    with open("/model/lookups.json", "r") as f:
        return json.load(f)

with st.spinner("Initializing Spark engine..."):
    spark, model = load_spark_and_model()
    lookups = load_lookups()

# ── Stats Row ────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
stats = [
    ("Race Records",  "26,759", "1950 — 2024"),
    ("Model AUC",     "87.9%",  "Random Forest · 100 trees"),
    ("Drivers",       "861",    "Across all seasons"),
    ("Constructors",  "211",    "All teams included"),
    ("Circuits",      "77",     "Worldwide locations"),
]
for col_obj, (label, value, sub) in zip([c1, c2, c3, c4, c5], stats):
    with col_obj:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">{label}</div>
            <div class="stat-value">{value}</div>
            <div class="stat-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Main Layout ──────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="panel-title">Race Configuration</div>', unsafe_allow_html=True)
    driver      = st.selectbox("Driver",      options=lookups["drivers"])
    constructor = st.selectbox("Constructor", options=lookups["constructors"])
    circuit     = st.selectbox("Circuit",     options=lookups["circuits"])

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Race Parameters</div>', unsafe_allow_html=True)

    ca, cb = st.columns(2)
    with ca:
        grid = st.slider("Grid Position", min_value=1, max_value=20, value=1)
    with cb:
        year = st.selectbox("Season", options=list(range(2024, 1949, -1)))

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("ANALYZE PODIUM PROBABILITY", use_container_width=True)

# ── Right Panel ──────────────────────────────────────────────────
with right:
    st.markdown('<div class="panel-title">Prediction Output</div>', unsafe_allow_html=True)

    if predict_btn:

        # ── Run prediction inside spinner ────────────────────────
        with st.spinner("Analyzing race data..."):
            schema = StructType([
                StructField("grid",             DoubleType(), True),
                StructField("year",             DoubleType(), True),
                StructField("round",            DoubleType(), True),
                StructField("driver_name",      StringType(), True),
                StructField("constructor_name", StringType(), True),
                StructField("circuit_name",     StringType(), True),
            ])

            round_num = float(lookups["circuits"].index(circuit) % 24 + 1)

            input_data = spark.createDataFrame(
                [(float(grid), float(year), round_num, driver, constructor, circuit)],
                schema=schema
            )

            prediction  = model.transform(input_data)
            result      = prediction.select("prediction", "probability").collect()[0]
            pred_label  = int(result["prediction"])
            probability = result["probability"]
            podium_prob    = round(float(probability[1]) * 100, 1)
            no_podium_prob = round(float(probability[0]) * 100, 1)

        # ── Result card ──────────────────────────────────────────
        if pred_label == 1:
            st.markdown(f"""
            <div class="result-win">
                <div class="result-verdict" style="color:#00c851">Podium Finish</div>
                <div class="result-desc">
                    <strong style="color:#ccc">{driver}</strong> is predicted to finish in the
                    <strong style="color:#00c851">Top 3</strong> at {circuit}
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-loss">
                <div class="result-verdict" style="color:#e10600">No Podium</div>
                <div class="result-desc">
                    <strong style="color:#ccc">{driver}</strong> is predicted to finish
                    <strong style="color:#e10600">outside the Top 3</strong> at {circuit}
                </div>
            </div>""", unsafe_allow_html=True)

        # ── Confidence bars ──────────────────────────────────────
        st.markdown('<div class="panel-title">Confidence Breakdown</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="prob-row">
            <span class="prob-label" style="color:#00c851">Podium</span>
            <span class="prob-pct"   style="color:#00c851">{podium_prob}%</span>
        </div>
        <div class="bar-track">
            <div class="bar-fill-green" style="width:{podium_prob}%"></div>
        </div>
        <div class="prob-row">
            <span class="prob-label" style="color:#e10600">No Podium</span>
            <span class="prob-pct"   style="color:#e10600">{no_podium_prob}%</span>
        </div>
        <div class="bar-track">
            <div class="bar-fill-red" style="width:{no_podium_prob}%"></div>
        </div>
        """, unsafe_allow_html=True)

        # ── Summary table ────────────────────────────────────────
        st.markdown(
            '<div class="panel-title" style="margin-top:1.5rem">Analysis Summary</div>',
            unsafe_allow_html=True
        )
        rows = [
            ("Driver",       driver),
            ("Constructor",  constructor),
            ("Circuit",      circuit),
            ("Grid Position",f"P{grid}"),
            ("Season",       str(year)),
            ("Model",        "Random Forest · AUC 87.9%"),
        ]
        html = ""
        for k, v in rows:
            html += f"""
            <div class="summary-row">
                <span class="summary-key">{k}</span>
                <span class="summary-val">{v}</span>
            </div>"""
        st.markdown(html, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="placeholder">
            <div class="placeholder-text">
                Configure parameters<br>and run analysis
            </div>
        </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="
    text-align:center;
    color:#222;
    font-size:0.6rem;
    letter-spacing:3px;
    text-transform:uppercase;
    border-top:1px solid #111;
    padding-top:1.5rem
">
    F1 Podium Predictor &nbsp;·&nbsp; Apache Spark ML &nbsp;·&nbsp;
    Streamlit &nbsp;·&nbsp; Championship Data 1950–2024
</div>
""", unsafe_allow_html=True)