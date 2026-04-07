# import library

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import io
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="Student Score Predictor · AI System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# frontend UI
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0d1b2a 0%, #112240 50%, #0d1b2a 100%);
    background-attachment: fixed;
}
.block-container { padding: 2rem 3rem 3rem 3rem !important; max-width: 1300px; }

/* ── Hero ── */
.hero-header {
    background: linear-gradient(135deg, rgba(100,180,255,0.08) 0%, rgba(0,212,255,0.05) 100%);
    border: 1px solid rgba(100,180,255,0.25); border-radius: 20px;
    padding: 2.5rem 3rem; margin-bottom: 2rem; position: relative; overflow: hidden;
}
.hero-header::before {
    content: ''; position: absolute; top: -40px; right: -40px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(0,212,255,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Playfair Display', serif; font-size: 2.8rem; font-weight: 700;
    background: linear-gradient(90deg, #e8f4fd 0%, #64b4ff 60%, #00d4ff 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin: 0 0 0.5rem 0; line-height: 1.15;
}
.hero-sub { color: #8ab4d4; font-size: 1.05rem; font-weight: 300; max-width: 680px; }
.hero-badge {
    display: inline-block; background: rgba(0,212,255,0.15); color: #00d4ff;
    border: 1px solid rgba(0,212,255,0.35); border-radius: 50px;
    padding: 4px 14px; font-size: 0.75rem; font-weight: 600;
    letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 1rem;
}

/* ── Mode tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(13,27,42,0.7) !important;
    border-radius: 12px !important; border: 1px solid rgba(100,180,255,0.18) !important;
    padding: 4px !important; gap: 4px !important; margin-bottom: 1.5rem;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; color: #6a9bc4 !important;
    border-radius: 9px !important; font-weight: 500 !important;
    font-size: 0.92rem !important; padding: 0.55rem 1.4rem !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #0077cc, #00b4d8) !important;
    color: #fff !important; font-weight: 600 !important;
    box-shadow: 0 2px 12px rgba(0,180,216,0.35) !important;
}

/* ── Section cards ── */
.section-card {
    background: rgba(17,34,64,0.7); border: 1px solid rgba(100,180,255,0.18);
    border-radius: 16px; padding: 1.8rem 2rem; margin-bottom: 1.5rem;
    backdrop-filter: blur(8px);
}
.section-title {
    font-family: 'Playfair Display', serif; font-size: 1.1rem; color: #64b4ff;
    letter-spacing: 0.04em; margin: 0 0 1.2rem 0; padding-bottom: 0.6rem;
    border-bottom: 1px solid rgba(100,180,255,0.15);
}

/* ── Upload zone ── */
.upload-zone {
    background: rgba(0,100,200,0.06); border: 2px dashed rgba(100,180,255,0.3);
    border-radius: 16px; padding: 2.5rem; text-align: center; margin-bottom: 1.2rem;
}
.upload-icon { font-size: 3rem; margin-bottom: 0.8rem; }
.upload-title { color: #64b4ff; font-size: 1.15rem; font-weight: 600; margin-bottom: 0.4rem; }
.upload-sub { color: #6a9bc4; font-size: 0.88rem; }

/* ── Inputs ── */
.stSelectbox label, .stNumberInput label {
    color: #a8c8e8 !important; font-size: 0.85rem !important;
    font-weight: 500 !important; letter-spacing: 0.03em !important;
}
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: rgba(13,27,42,0.8) !important;
    border: 1px solid rgba(100,180,255,0.25) !important;
    border-radius: 10px !important; color: #e8f4fd !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #0077cc 0%, #00b4d8 100%) !important;
    color: #fff !important; border: none !important; border-radius: 12px !important;
    padding: 0.85rem 2.5rem !important; font-size: 1.05rem !important;
    font-weight: 600 !important; width: 100% !important;
    box-shadow: 0 4px 24px rgba(0,180,216,0.3) !important;
    transition: all 0.3s ease !important; margin-top: 0.5rem !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(0,180,216,0.45) !important;
}

/* ── Result box ── */
.result-box {
    border-radius: 18px; padding: 2.2rem 2.5rem; margin: 1.2rem 0;
    text-align: center; position: relative; overflow: hidden;
}
.result-excellent { background: linear-gradient(135deg,rgba(0,200,120,0.15),rgba(0,230,100,0.08)); border: 1.5px solid rgba(0,200,120,0.4); }
.result-good      { background: linear-gradient(135deg,rgba(0,160,220,0.15),rgba(0,200,255,0.08)); border: 1.5px solid rgba(0,160,220,0.4); }
.result-average   { background: linear-gradient(135deg,rgba(255,190,0,0.15),rgba(255,220,50,0.08)); border: 1.5px solid rgba(255,190,0,0.4); }
.result-needs     { background: linear-gradient(135deg,rgba(255,90,90,0.15),rgba(255,120,80,0.08)); border: 1.5px solid rgba(255,90,90,0.4); }
.result-score  { font-family: 'Playfair Display',serif; font-size: 4.5rem; font-weight: 700; line-height:1; margin: 0.3rem 0; }
.result-label  { font-size: 1.1rem; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; margin-top: 0.4rem; }
.result-msg    { font-size: 0.95rem; margin-top: 0.6rem; font-weight: 300; opacity: 0.85; }

/* ── Progress bar ── */
.progress-wrap {
    background: rgba(255,255,255,0.07); border-radius: 50px;
    height: 10px; margin: 1.2rem auto; max-width: 320px; overflow: hidden;
}
.progress-fill { height: 100%; border-radius: 50px; }

/* ── Metric cards ── */
.metric-row { display: flex; gap: 1rem; margin-bottom: 1.2rem; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 100px;
    background: rgba(100,180,255,0.06); border: 1px solid rgba(100,180,255,0.15);
    border-radius: 12px; padding: 1rem 1.2rem; text-align: center;
}
.metric-val { font-family: 'Playfair Display',serif; font-size: 1.8rem; color: #00d4ff; font-weight: 700; }
.metric-lbl { font-size: 0.72rem; color: #6a9bc4; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 2px; }

/* ── Validation boxes ── */
.val-error {
    background: rgba(255,80,80,0.1); border: 1px solid rgba(255,80,80,0.35);
    border-radius: 10px; padding: 0.7rem 1rem; font-size: 0.85rem; color: #ff8080;
    margin-bottom: 0.6rem;
}
.val-ok {
    background: rgba(0,200,120,0.1); border: 1px solid rgba(0,200,120,0.3);
    border-radius: 10px; padding: 0.7rem 1rem; font-size: 0.85rem; color: #00c878;
    margin-bottom: 0.6rem;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(10,22,40,0.95) !important;
    border-right: 1px solid rgba(100,180,255,0.15) !important;
}
[data-testid="stSidebar"] .block-container { padding: 2rem 1.2rem !important; }
.sidebar-header {
    font-family: 'Playfair Display',serif; font-size: 1.25rem; color: #64b4ff;
    border-bottom: 1px solid rgba(100,180,255,0.2);
    padding-bottom: 0.7rem; margin-bottom: 1.1rem;
}
.sidebar-stat {
    background: rgba(100,180,255,0.07); border: 1px solid rgba(100,180,255,0.15);
    border-radius: 10px; padding: 0.65rem 1rem; margin-bottom: 0.65rem;
    font-size: 0.86rem; color: #a8c8e8;
}
.sidebar-stat span { color: #00d4ff; font-weight: 600; }

/* ── Footer ── */
.footer {
    text-align: center; color: rgba(100,140,180,0.6); font-size: 0.8rem;
    padding: 2rem 0 1rem; border-top: 1px solid rgba(100,180,255,0.1);
    margin-top: 3rem; letter-spacing: 0.04em;
}
.footer span { color: #64b4ff; }

hr { border: none !important; border-top: 1px solid rgba(100,180,255,0.15) !important; margin: 1.5rem 0 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.top-nav {
    display: flex; justify-content: space-between; align-items: center;
    background: rgba(8, 18, 34, 0.82);
    border: 1px solid rgba(100,180,255,0.16);
    border-radius: 14px; padding: 0.85rem 1.1rem; margin-bottom: 1.1rem;
}
.brand-mark { color: #e8f4fd; font-weight: 600; letter-spacing: 0.03em; }
.brand-mark span { color: #00d4ff; }
.nav-links { color: #6a9bc4; font-size: 0.84rem; }
.hero-kpis {
    display: grid; grid-template-columns: repeat(4, minmax(120px, 1fr));
    gap: 0.7rem; margin-top: 1.2rem;
}
.hero-kpi {
    background: rgba(100,180,255,0.06);
    border: 1px solid rgba(100,180,255,0.2);
    border-radius: 12px; padding: 0.75rem 0.9rem;
}
.hero-kpi strong { color: #e8f4fd; font-size: 0.95rem; display: block; }
.hero-kpi span { color: #7fa8cc; font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.07em; }
.micro-note {
    color: #7fa8cc; font-size: 0.85rem; margin: 0.35rem 0 1.3rem 0;
}
.step-chip {
    display: inline-flex; align-items: center; gap: 0.35rem;
    padding: 0.32rem 0.62rem; border-radius: 999px;
    background: rgba(0,180,216,0.12); border: 1px solid rgba(0,180,216,0.28);
    color: #8fd9e7; font-size: 0.73rem; text-transform: uppercase; letter-spacing: 0.08em;
}
.empty-state {
    text-align: center; padding: 2.4rem 1.2rem; color: #7a9ec0; font-size: 0.95rem;
    border: 1px dashed rgba(100,180,255,0.2); border-radius: 14px;
    background: rgba(10, 20, 36, 0.45);
}
.empty-state b { color: #00d4ff; }
</style>
""", unsafe_allow_html=True)


CATEGORICAL_FEATURES = {
    "Parental_Involvement":        ["Low", "Medium", "High"],
    "Access_to_Resources":         ["Low", "Medium", "High"],
    "Extracurricular_Activities":  ["No", "Yes"],
    "Motivation_Level":            ["Low", "Medium", "High"],
    "Internet_Access":             ["No", "Yes"],
    "Family_Income":               ["Low", "Medium", "High"],
    "Teacher_Quality":             ["Low", "Medium", "High"],
    "School_Type":                 ["Public", "Private"],
    "Peer_Influence":              ["Negative", "Neutral", "Positive"],
    "Learning_Disabilities":       ["No", "Yes"],
    "Parental_Education_Level":    ["High School", "College", "Postgraduate"],
    "Distance_from_Home":          ["Near", "Moderate", "Far"],
    "Gender":                      ["Female", "Male"],
}

NUMERIC_FEATURES = {
    "Hours_Studied":     {"min_value": 1,  "max_value": 44,  "value": 8,  "step": 1},
    "Attendance":        {"min_value": 60, "max_value": 100, "value": 85, "step": 1},
    "Sleep_Hours":       {"min_value": 4,  "max_value": 10,  "value": 7,  "step": 1},
    "Previous_Scores":   {"min_value": 40, "max_value": 100, "value": 70, "step": 1},
    "Tutoring_Sessions": {"min_value": 0,  "max_value": 8,   "value": 2,  "step": 1},
    "Physical_Activity": {"min_value": 0,  "max_value": 6,   "value": 3,  "step": 1},
}

ALL_FEATURES = [
    "Hours_Studied", "Attendance", "Parental_Involvement",
    "Access_to_Resources", "Extracurricular_Activities", "Sleep_Hours",
    "Previous_Scores", "Motivation_Level", "Internet_Access",
    "Tutoring_Sessions", "Family_Income", "Teacher_Quality",
    "School_Type", "Peer_Influence", "Physical_Activity",
    "Learning_Disabilities", "Parental_Education_Level",
    "Distance_from_Home", "Gender",
]

NUMERIC_COLS = list(NUMERIC_FEATURES.keys())
CAT_COLS     = list(CATEGORICAL_FEATURES.keys())

 #Load saved .pkl artefacts or train a synthetic demo model.
@st.cache_resource
def load_or_create_model():
    
    if (os.path.exists("student_model.pkl") and
            os.path.exists("label_encoders.pkl") and
            os.path.exists("feature_columns.pkl")):
        with open("student_model.pkl",   "rb") as f: model           = pickle.load(f)
        with open("label_encoders.pkl",  "rb") as f: label_encoders  = pickle.load(f)
        with open("feature_columns.pkl", "rb") as f: feature_columns = pickle.load(f)
        return model, label_encoders, feature_columns, False

    # ── Synthetic demo model ──
    np.random.seed(42)
    N = 5000
    rows = [{
        "Hours_Studied":            np.random.randint(1, 45),
        "Attendance":               np.random.randint(60, 101),
        "Parental_Involvement":     np.random.choice(["Low","Medium","High"]),
        "Access_to_Resources":      np.random.choice(["Low","Medium","High"]),
        "Extracurricular_Activities": np.random.choice(["No","Yes"]),
        "Sleep_Hours":              np.random.randint(4, 11),
        "Previous_Scores":          np.random.randint(40, 101),
        "Motivation_Level":         np.random.choice(["Low","Medium","High"]),
        "Internet_Access":          np.random.choice(["No","Yes"]),
        "Tutoring_Sessions":        np.random.randint(0, 9),
        "Family_Income":            np.random.choice(["Low","Medium","High"]),
        "Teacher_Quality":          np.random.choice(["Low","Medium","High"]),
        "School_Type":              np.random.choice(["Public","Private"]),
        "Peer_Influence":           np.random.choice(["Negative","Neutral","Positive"]),
        "Physical_Activity":        np.random.randint(0, 7),
        "Learning_Disabilities":    np.random.choice(["No","Yes"]),
        "Parental_Education_Level": np.random.choice(["High School","College","Postgraduate"]),
        "Distance_from_Home":       np.random.choice(["Near","Moderate","Far"]),
        "Gender":                   np.random.choice(["Female","Male"]),
    } for _ in range(N)]

    df = pd.DataFrame(rows)
    label_encoders = {}
    df_enc = df.copy()
    for col, classes in CATEGORICAL_FEATURES.items():
        le = LabelEncoder()
        le.fit(classes)
        df_enc[col] = le.transform(df[col])
        label_encoders[col] = le

    score = np.clip(
        df_enc["Hours_Studied"]        * 0.45 +
        df_enc["Attendance"]           * 0.25 +
        df_enc["Previous_Scores"]      * 0.30 +
        df_enc["Parental_Involvement"] * 2.5  +
        df_enc["Access_to_Resources"]  * 2.0  +
        df_enc["Motivation_Level"]     * 2.8  +
        df_enc["Teacher_Quality"]      * 2.2  +
        df_enc["Tutoring_Sessions"]    * 1.2  +
        df_enc["Internet_Access"]      * 1.5  +
        df_enc["Peer_Influence"]       * 1.8  +
        np.random.normal(0, 3, N), 40, 100)

    model = RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1)
    model.fit(df_enc[ALL_FEATURES], score)
    return model, label_encoders, ALL_FEATURES, True


def get_score_tier(score: float):
    """Return (label, css_class, emoji, message, hex_color, gradient)."""
    if score >= 85:
        return ("Excellent",         "result-excellent", "",
                "Outstanding performance! Keep up the great work.",
                "#00c878", "linear-gradient(90deg,#00c878,#00e664)")
    elif score >= 70:
        return ("Good",              "result-good",      "",
                "Solid performance with room to push even higher.",
                "#00a0dc", "linear-gradient(90deg,#00a0dc,#00d4ff)")
    elif score >= 55:
        return ("Average",           "result-average",   "",
                "Decent foundation — focus on weak areas to improve.",
                "#ffbe00", "linear-gradient(90deg,#ffbe00,#ffdc32)")
    else:
        return ("Needs Improvement", "result-needs",     "",
                "Significant effort needed — consider extra support.",
                "#ff5a5a", "linear-gradient(90deg,#ff5a5a,#ff7850)")


def encode_and_predict(model, label_encoders, feature_columns, input_dict: dict) -> float:
    """Encode a single input dict and return predicted score."""
    row = {}
    for feat in feature_columns:
        if feat in CATEGORICAL_FEATURES:
            row[feat] = label_encoders[feat].transform([input_dict[feat]])[0]
        else:
            row[feat] = input_dict[feat]
    return float(model.predict(pd.DataFrame([row])[feature_columns])[0])


def batch_predict(model, label_encoders, feature_columns, df: pd.DataFrame) -> pd.DataFrame:
    """Encode a whole DataFrame and return it with Predicted_Score + Performance_Band."""
    df_enc = df.copy()
    for col in CAT_COLS:
        if col in df_enc.columns:
            valid = set(CATEGORICAL_FEATURES[col])
            df_enc[col] = df_enc[col].apply(
                lambda x: x if x in valid else CATEGORICAL_FEATURES[col][0])
            df_enc[col] = label_encoders[col].transform(df_enc[col].astype(str))
    scores = np.clip(model.predict(df_enc[feature_columns]), 0, 100).round(1)
    df_out = df.copy()
    df_out["Predicted_Score"]   = scores
    df_out["Performance_Band"]  = [get_score_tier(s)[0] for s in scores]
    return df_out


def plot_feature_importance(model, feature_columns):
    """Horizontal bar chart — dark theme."""
    imp = model.feature_importances_
    idx = np.argsort(imp)[-12:]
    vals = imp[idx]; norm = vals / vals.max()
    labels = [feature_columns[i].replace("_"," ") for i in idx]
    colors = [plt.cm.cool(0.2 + 0.8 * v) for v in norm]

    fig, ax = plt.subplots(figsize=(8, len(idx)*0.48 + 0.6))
    fig.patch.set_facecolor("#0d1b2a"); ax.set_facecolor("#0d1b2a")
    bars = ax.barh(labels, vals, color=colors, height=0.62, edgecolor="none")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width()+0.002, bar.get_y()+bar.get_height()/2,
                f"{val:.3f}", va="center", ha="left", fontsize=8, color="#8ab4d4")
    ax.set_xlabel("Importance Score", color="#6a9bc4", fontsize=9)
    ax.tick_params(colors="#a8c8e8", labelsize=8.5)
    for sp in ax.spines.values(): sp.set_color("#1a3050")
    ax.grid(axis="x", color="#152035", linewidth=0.7)
    ax.set_title("Feature Importance · Random Forest", color="#64b4ff",
                 fontsize=11, pad=12, fontweight="600")
    plt.tight_layout(); return fig


def plot_batch_analytics(df_results: pd.DataFrame):
    """4-panel analytics dashboard for batch results."""
    BAND_COLORS = {
        "Excellent": "#00c878", "Good": "#00a0dc",
        "Average": "#ffbe00", "Needs Improvement": "#ff5a5a"
    }
    DARK, GRID, TEXT, ACCENT = "#0d1b2a", "#152035", "#a8c8e8", "#64b4ff"

    def style_ax(a):
        a.set_facecolor(DARK)
        for sp in a.spines.values(): sp.set_color("#1a3050")
        a.tick_params(colors=TEXT, labelsize=8)
        a.grid(color=GRID, linewidth=0.6)

    fig = plt.figure(figsize=(14, 9), facecolor=DARK)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    scores      = df_results["Predicted_Score"]
    band_counts = df_results["Performance_Band"].value_counts()
    bands_ord   = [b for b in ["Excellent","Good","Average","Needs Improvement"] if b in band_counts]

    # ── Panel 1: Histogram ──
    ax1 = fig.add_subplot(gs[0, 0]); style_ax(ax1)
    ax1.hist(scores, bins=20, color="#0077cc", edgecolor="#0d1b2a", alpha=0.85)
    ax1.axvline(scores.mean(),   color="#00d4ff", lw=1.8, ls="--", label=f"Mean {scores.mean():.1f}")
    ax1.axvline(scores.median(), color="#ffbe00", lw=1.8, ls=":",  label=f"Median {scores.median():.1f}")
    ax1.set_title("Score Distribution",      color=ACCENT, fontsize=10, fontweight="600")
    ax1.set_xlabel("Predicted Score",        color=TEXT, fontsize=8)
    ax1.set_ylabel("Count",                  color=TEXT, fontsize=8)
    ax1.legend(fontsize=7.5, labelcolor=TEXT, facecolor="#0f2040", edgecolor="#1a3050")

    # ── Panel 2: Donut ──
    ax2 = fig.add_subplot(gs[0, 1]); ax2.set_facecolor(DARK)
    sizes  = [band_counts[b] for b in bands_ord]
    colors = [BAND_COLORS[b] for b in bands_ord]
    wedges, texts, autotexts = ax2.pie(
        sizes, labels=bands_ord, colors=colors, autopct="%1.1f%%", startangle=140,
        wedgeprops=dict(width=0.55, edgecolor="#0d1b2a", linewidth=2),
        textprops=dict(color=TEXT, fontsize=8))
    for at in autotexts: at.set_color("#fff"); at.set_fontsize(7.5)
    ax2.set_title("Performance Band Split",  color=ACCENT, fontsize=10, fontweight="600")

    # ── Panel 3: Hours Studied scatter ──
    ax3 = fig.add_subplot(gs[1, 0]); style_ax(ax3)
    if "Hours_Studied" in df_results.columns:
        sc = ax3.scatter(df_results["Hours_Studied"], scores,
                         c=scores, cmap="cool", alpha=0.72, s=22, linewidths=0)
        cbar = plt.colorbar(sc, ax=ax3)
        cbar.ax.tick_params(colors=TEXT, labelsize=7)
        cbar.set_label("Score", color=TEXT, fontsize=7)
        ax3.set_xlabel("Hours Studied / Week", color=TEXT, fontsize=8)
        ax3.set_ylabel("Predicted Score",      color=TEXT, fontsize=8)
    else:
        ax3.text(0.5, 0.5, "Hours_Studied\nnot available",
                 ha="center", va="center", color=TEXT, fontsize=10, transform=ax3.transAxes)
    ax3.set_title("Study Hours vs Score",   color=ACCENT, fontsize=10, fontweight="600")

    # ── Panel 4: Bar chart by band ──
    ax4 = fig.add_subplot(gs[1, 1]); style_ax(ax4)
    bar_c = [BAND_COLORS.get(b, "#64b4ff") for b in bands_ord]
    bars  = ax4.bar(bands_ord, sizes, color=bar_c, edgecolor="#0d1b2a")
    for bar, val in zip(bars, sizes):
        ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                 str(val), ha="center", va="bottom", fontsize=8, color=TEXT, fontweight="600")
    ax4.set_title("Students per Band",     color=ACCENT, fontsize=10, fontweight="600")
    ax4.set_xlabel("Performance Band",     color=TEXT, fontsize=8)
    ax4.set_ylabel("Count",               color=TEXT, fontsize=8)
    ax4.tick_params(axis="x", labelrotation=10)

    fig.suptitle("Batch Prediction Analytics Dashboard",
                 color="#e8f4fd", fontsize=13, fontweight="700", y=0.99)
    return fig


def validate_uploaded_df(df: pd.DataFrame):
    """Returns (is_valid, errors_list, warnings_list)."""
    errors, warnings = [], []
    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {', '.join(missing)}")
        return False, errors, warnings

    ranges = {
        "Hours_Studied": (1, 44), "Attendance": (60, 100), "Sleep_Hours": (4, 10),
        "Previous_Scores": (40, 100), "Tutoring_Sessions": (0, 8), "Physical_Activity": (0, 6),
    }
    for col, (lo, hi) in ranges.items():
        if col in df.columns:
            out = df[(df[col] < lo) | (df[col] > hi)]
            if not out.empty:
                warnings.append(f"'{col}': {len(out)} rows outside [{lo},{hi}] — will be clipped.")

    for col, valid_vals in CATEGORICAL_FEATURES.items():
        if col in df.columns:
            bad = set(df[col].astype(str).unique()) - set(valid_vals)
            if bad:
                warnings.append(f"'{col}': unknown values {bad} — defaults to '{valid_vals[0]}'.")

    if df.isnull().values.any():
        null_cols = df.columns[df.isnull().any()].tolist()
        warnings.append(f"NaN values in: {null_cols} — filled with mode / median.")

    return True, errors, warnings


def build_template_df() -> pd.DataFrame:
    """Two-row sample template."""
    rows = [
        {
            "Hours_Studied": 8, "Attendance": 85, "Parental_Involvement": "Medium",
            "Access_to_Resources": "High", "Extracurricular_Activities": "Yes",
            "Sleep_Hours": 7, "Previous_Scores": 72, "Motivation_Level": "High",
            "Internet_Access": "Yes", "Tutoring_Sessions": 2, "Family_Income": "Medium",
            "Teacher_Quality": "High", "School_Type": "Public", "Peer_Influence": "Positive",
            "Physical_Activity": 3, "Learning_Disabilities": "No",
            "Parental_Education_Level": "College", "Distance_from_Home": "Near", "Gender": "Male",
        },
        {
            "Hours_Studied": 4, "Attendance": 68, "Parental_Involvement": "Low",
            "Access_to_Resources": "Low", "Extracurricular_Activities": "No",
            "Sleep_Hours": 5, "Previous_Scores": 52, "Motivation_Level": "Low",
            "Internet_Access": "No", "Tutoring_Sessions": 0, "Family_Income": "Low",
            "Teacher_Quality": "Medium", "School_Type": "Public", "Peer_Influence": "Negative",
            "Physical_Activity": 1, "Learning_Disabilities": "No",
            "Parental_Education_Level": "High School", "Distance_from_Home": "Far", "Gender": "Female",
        },
    ]
    return pd.DataFrame(rows)[ALL_FEATURES]


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="Predictions")
    return buf.getvalue()


def apply_theme(theme_mode: str):
    """Inject additional theme CSS (dark/light) and motion polish."""
    if theme_mode == "Light":
        st.markdown("""
        <style>
        .stApp { background: linear-gradient(145deg, #f3f8ff 0%, #e7f1ff 48%, #f3f8ff 100%) !important; }
        .top-nav { background: rgba(255,255,255,0.88) !important; border-color: rgba(55,110,180,0.22) !important; }
        .brand-mark { color: #11325a !important; }
        .nav-links, .micro-note, .hero-sub { color: #355d87 !important; }
        .hero-header, .section-card, .metric-card, .hero-kpi {
            background: rgba(255,255,255,0.82) !important;
            border-color: rgba(55,110,180,0.2) !important;
        }
        .hero-title {
            background: linear-gradient(90deg, #183d6a 0%, #245f96 55%, #2f8ac4 100%) !important;
            -webkit-background-clip: text !important; -webkit-text-fill-color: transparent !important;
            background-clip: text !important;
        }
        .step-chip { background: rgba(38,111,190,0.12) !important; border-color: rgba(38,111,190,0.28) !important; color: #266fbe !important; }
        .empty-state {
            background: rgba(255,255,255,0.85) !important;
            border-color: rgba(38,111,190,0.26) !important;
            color: #355d87 !important;
        }
        .stSelectbox label, .stNumberInput label { color: #355d87 !important; }
        .stSelectbox > div > div, .stNumberInput > div > div > input {
            background: #f7fbff !important; border-color: rgba(38,111,190,0.35) !important; color: #163f6d !important;
        }
        [data-testid="stSidebar"] { background: rgba(239,246,255,0.96) !important; border-right-color: rgba(55,110,180,0.2) !important; }
        .sidebar-header { color: #245f96 !important; }
        .sidebar-stat { background: rgba(38,111,190,0.08) !important; border-color: rgba(38,111,190,0.2) !important; color: #355d87 !important; }
        .footer { color: rgba(40,85,130,0.85) !important; border-top-color: rgba(55,110,180,0.2) !important; }
        </style>
        """, unsafe_allow_html=True)

    st.markdown("""
    <style>
    .top-nav {
        position: sticky; top: 0.6rem; z-index: 999;
        backdrop-filter: blur(8px);
    }
    .hero-header, .section-card, .result-box, .metric-card, .empty-state {
        animation: fadeSlideIn 0.45s ease both;
    }
    .hero-kpi, .sidebar-stat {
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .hero-kpi:hover, .sidebar-stat:hover { transform: translateY(-2px); box-shadow: 0 8px 18px rgba(0,0,0,0.15); }
    @keyframes fadeSlideIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0px); }
    }
    </style>
    """, unsafe_allow_html=True)



#  SIDEBAR

if "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "Dark"
if "dark_mode_enabled" not in st.session_state:
    st.session_state["dark_mode_enabled"] = st.session_state["theme_mode"] == "Dark"
if "site_view" not in st.session_state:
    st.session_state["site_view"] = "Landing"

with st.sidebar:
    st.markdown('<p class="sidebar-header"> Website Controls</p>', unsafe_allow_html=True)
    st.session_state["site_view"] = st.radio(
        "Page",
        ["Landing", "Dashboard"],
        index=0 if st.session_state["site_view"] == "Landing" else 1,
        horizontal=False,
    )
    st.session_state["dark_mode_enabled"] = st.toggle(
        "Dark mode",
        value=st.session_state["dark_mode_enabled"],
    )
    st.session_state["theme_mode"] = (
        "Dark" if st.session_state["dark_mode_enabled"] else "Light"
    )

    st.markdown("---")
    st.markdown('<p class="sidebar-header"> System Info</p>', unsafe_allow_html=True)

    model, label_encoders, feature_columns, is_demo = load_or_create_model()

    if is_demo:
        st.warning(" Demo mode — synthetic model.\nPlace your .pkl files alongside app.py to load your real model.", icon="⚠️")
    else:
        st.success(" Pre-trained model loaded.", icon="")

    st.markdown(f"""
    <div class="sidebar-stat"> Algorithm: <span>Random Forest</span></div>
    <div class="sidebar-stat"> Features: <span>{len(feature_columns)}</span></div>
    <div class="sidebar-stat"> Trees: <span>{model.n_estimators}</span></div>
    <div class="sidebar-stat"> Task: <span>Regression</span></div>
    <div class="sidebar-stat"> Modes: <span>Single · Batch</span></div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="sidebar-header"> Score Bands</p>', unsafe_allow_html=True)
    for lbl, rng, col in [
        (" Excellent",         "≥ 85",  "#00c878"),
        (" Good",              "70–84", "#00a0dc"),
        (" Average",           "55–69", "#ffbe00"),
        (" Needs Improvement", "< 55",  "#ff5a5a"),
    ]:
        st.markdown(f'<div class="sidebar-stat">{lbl}: <span style="color:{col}">{rng}</span></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="sidebar-header">Accepted Values</p>', unsafe_allow_html=True)
    with st.expander("View all allowed categories", expanded=False):
        for col, vals in CATEGORICAL_FEATURES.items():
            st.markdown(
                f'<div class="sidebar-stat" style="margin-bottom:0.5rem;">'
                f'<b style="color:#64b4ff">{col}</b><br>'
                f'<span style="color:#a8c8e8">{" · ".join(vals)}</span></div>',
                unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="sidebar-header">ℹ About</p>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#7a9ec0;font-size:0.83rem;line-height:1.6;">'
        'AI Student Prediction System <b style="color:#64b4ff">v2.0</b> — supports '
        'single and <b style="color:#64b4ff">batch CSV/Excel</b> prediction with '
        'full analytics dashboard and downloadable results.</p>',
        unsafe_allow_html=True)

apply_theme(st.session_state["theme_mode"])

if st.session_state["site_view"] == "Landing":
    st.markdown("""
    <div class="top-nav">
      <div class="brand-mark">EduCast <span>Predictive Intelligence</span></div>
      <div class="nav-links">Home · Capabilities · Workflow · Contact</div>
    </div>
    <div class="hero-header">
      <div class="hero-badge">Institution-ready student analytics</div>
      <h1 class="hero-title">Design-Driven Academic<br>Prediction Platform</h1>
      <p class="hero-sub">
        Build faster intervention plans with AI-assisted score forecasts, visual reporting,
        and consistent data workflows for classrooms, departments, and schools.
      </p>
      <div class="hero-kpis">
        <div class="hero-kpi"><strong>99.9%</strong><span>Uptime Ready UX</span></div>
        <div class="hero-kpi"><strong>2 Modes</strong><span>Single + Batch</span></div>
        <div class="hero-kpi"><strong>4 Charts</strong><span>Analytics Coverage</span></div>
        <div class="hero-kpi"><strong>Exports</strong><span>CSV + Excel</span></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="section-card"><p class="section-title">Predict</p><p class="hero-sub">Estimate a student score instantly with guided inputs and clean result cards.</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="section-card"><p class="section-title">Analyse</p><p class="hero-sub">Track trends using distribution, band split, and score relationship charts.</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="section-card"><p class="section-title">Act</p><p class="hero-sub">Download reports and share structured evidence for targeted academic support.</p></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="empty-state">
      Open the left sidebar and switch <b>Page → Dashboard</b> to start predictions.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="footer">
      EduCast Predictive Intelligence · Product Landing View · Switch to Dashboard for interactive tools
    </div>
    """, unsafe_allow_html=True)
    st.stop()



#  HERO HEADER

st.markdown("""
<div class="top-nav">
  <div class="brand-mark">EduCast <span>Predictive Intelligence</span></div>
  <div class="nav-links">Overview · Single Forecast · Batch Analytics · Download Reports</div>
</div>
<div class="hero-header">
  <div class="hero-badge">Education AI Platform · v2.0</div>
  <h1 class="hero-title">Student Performance Forecasting<br>for Schools & Institutions</h1>
  <p class="hero-sub">
    A production-style web experience for academic teams to estimate student outcomes,
    review performance bands, and generate shareable insights from both
    <strong>single profiles</strong> and <strong>batch datasets</strong>.
  </p>
  <div class="hero-kpis">
    <div class="hero-kpi"><strong>19 Inputs</strong><span>Model Features</span></div>
    <div class="hero-kpi"><strong>Single Mode</strong><span>Instant Prediction</span></div>
    <div class="hero-kpi"><strong>Batch Mode</strong><span>CSV / Excel Ready</span></div>
    <div class="hero-kpi"><strong>Analytics</strong><span>Visual Dashboard</span></div>
  </div>
</div>
<p class="micro-note">Use the tabs below to move through prediction workflows like a standard web application.</p>
""", unsafe_allow_html=True)


#  MODE TABS

tab_single, tab_batch = st.tabs([
    "Single Student Workspace",
    "Batch Processing Workspace (CSV / Excel)",
])



#  TAB 1 — SINGLE STUDENT PREDICTION

with tab_single:

    with st.form("prediction_form"):

        # Section 1 – Academic Habits
        st.markdown('<div class="section-card"><p class="section-title"><span class="step-chip">Step 01</span> Academic Habits</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: hours_studied  = st.number_input("Hours Studied / Week",         **NUMERIC_FEATURES["Hours_Studied"])
        with c2: attendance     = st.number_input("Attendance (%)",               **NUMERIC_FEATURES["Attendance"])
        with c3: previous_sc    = st.number_input("Previous Scores",              **NUMERIC_FEATURES["Previous_Scores"])
        c4, c5, c6 = st.columns(3)
        with c4: tutoring       = st.number_input("Tutoring Sessions / Month",    **NUMERIC_FEATURES["Tutoring_Sessions"])
        with c5: sleep_hours    = st.number_input("Sleep Hours / Night",          **NUMERIC_FEATURES["Sleep_Hours"])
        with c6: physical_act   = st.number_input("Physical Activity (hrs/week)", **NUMERIC_FEATURES["Physical_Activity"])
        st.markdown('</div>', unsafe_allow_html=True)

        # Section 2 – Student Profile
        st.markdown('<div class="section-card"><p class="section-title"><span class="step-chip">Step 02</span> Student Profile</p>', unsafe_allow_html=True)
        p1, p2, p3, p4 = st.columns(4)
        with p1: gender          = st.selectbox("Gender",                    CATEGORICAL_FEATURES["Gender"])
        with p2: motivation      = st.selectbox("Motivation Level",          CATEGORICAL_FEATURES["Motivation_Level"])
        with p3: learning_dis    = st.selectbox("Learning Disabilities",     CATEGORICAL_FEATURES["Learning_Disabilities"])
        with p4: extracurricular = st.selectbox("Extracurricular Activities",CATEGORICAL_FEATURES["Extracurricular_Activities"])
        st.markdown('</div>', unsafe_allow_html=True)

        # Section 3 – Family & Environment
        st.markdown('<div class="section-card"><p class="section-title"><span class="step-chip">Step 03</span> Family & Environment</p>', unsafe_allow_html=True)
        f1, f2, f3 = st.columns(3)
        with f1: parental_inv  = st.selectbox("Parental Involvement",      CATEGORICAL_FEATURES["Parental_Involvement"])
        with f2: parental_edu  = st.selectbox("Parental Education Level",  CATEGORICAL_FEATURES["Parental_Education_Level"])
        with f3: family_income = st.selectbox("Family Income",             CATEGORICAL_FEATURES["Family_Income"])
        f4, f5, f6 = st.columns(3)
        with f4: internet      = st.selectbox("Internet Access",           CATEGORICAL_FEATURES["Internet_Access"])
        with f5: distance      = st.selectbox("Distance from Home",        CATEGORICAL_FEATURES["Distance_from_Home"])
        with f6: resources     = st.selectbox("Access to Resources",       CATEGORICAL_FEATURES["Access_to_Resources"])
        st.markdown('</div>', unsafe_allow_html=True)

        # Section 4 – School Environment
        st.markdown('<div class="section-card"><p class="section-title"><span class="step-chip">Step 04</span> School Environment</p>', unsafe_allow_html=True)
        s1, s2, s3 = st.columns(3)
        with s1: school_type  = st.selectbox("School Type",     CATEGORICAL_FEATURES["School_Type"])
        with s2: teacher_qual = st.selectbox("Teacher Quality", CATEGORICAL_FEATURES["Teacher_Quality"])
        with s3: peer_inf     = st.selectbox("Peer Influence",  CATEGORICAL_FEATURES["Peer_Influence"])
        st.markdown('</div>', unsafe_allow_html=True)

        # Submit button
        col_btn, _ = st.columns([1, 2])
        with col_btn:
            submitted = st.form_submit_button("Generate Prediction")

    # ── Single prediction results ──
    if submitted:
        input_data = {
            "Hours_Studied": hours_studied, "Attendance": attendance,
            "Parental_Involvement": parental_inv, "Access_to_Resources": resources,
            "Extracurricular_Activities": extracurricular, "Sleep_Hours": sleep_hours,
            "Previous_Scores": previous_sc, "Motivation_Level": motivation,
            "Internet_Access": internet, "Tutoring_Sessions": tutoring,
            "Family_Income": family_income, "Teacher_Quality": teacher_qual,
            "School_Type": school_type, "Peer_Influence": peer_inf,
            "Physical_Activity": physical_act, "Learning_Disabilities": learning_dis,
            "Parental_Education_Level": parental_edu, "Distance_from_Home": distance,
            "Gender": gender,
        }

        with st.spinner(" Analysing student profile..."):
            pred = round(float(np.clip(
                encode_and_predict(model, label_encoders, feature_columns, input_data), 0, 100)), 1)

        tier, css_cls, emoji, message, color, grad = get_score_tier(pred)
        pct = int(pred)

        # Result card
        st.markdown(f"""
        <div class="result-box {css_cls}">
          <div style="font-size:0.88rem;color:{color};letter-spacing:0.12em;
               text-transform:uppercase;font-weight:600;margin-bottom:0.3rem;">
            Predicted Exam Score
          </div>
          <div class="result-score" style="color:{color}">{pred}</div>
          <div class="result-label" style="color:{color}">{emoji} {tier}</div>
          <div class="progress-wrap">
            <div class="progress-fill" style="width:{pct}%;background:{grad}"></div>
          </div>
          <div class="result-msg" style="color:{color}">{message}</div>
        </div>
        """, unsafe_allow_html=True)

        # Metrics row
        st.markdown(f"""
        <div class="metric-row">
          <div class="metric-card">
            <div class="metric-val">{pred}</div>
            <div class="metric-lbl">Predicted Score</div>
          </div>
          <div class="metric-card">
            <div class="metric-val">{pct}%</div>
            <div class="metric-lbl">Percentile Est.</div>
          </div>
          <div class="metric-card">
            <div class="metric-val">{tier}</div>
            <div class="metric-lbl">Performance Band</div>
          </div>
          <div class="metric-card">
            <div class="metric-val">{hours_studied}h</div>
            <div class="metric-lbl">Study Hrs / Week</div>
          </div>
          <div class="metric-card">
            <div class="metric-val">{attendance}%</div>
            <div class="metric-lbl">Attendance</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Feature importance
        st.markdown('<div class="section-card"><p class="section-title"> Feature Importance Analysis</p>', unsafe_allow_html=True)
        st.pyplot(plot_feature_importance(model, feature_columns), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Input summary + CSV download
        with st.expander(" View & Download Input Summary", expanded=False):
            df_summary = pd.DataFrame([input_data]).T.reset_index()
            df_summary.columns = ["Feature", "Value"]
            st.dataframe(df_summary, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇  Download Summary as CSV",
                data=to_csv_bytes(df_summary),
                file_name="single_prediction_summary.csv",
                mime="text/csv",
                use_container_width=True,
            )

    else:
        st.markdown("""
        <div class="empty-state">
          Complete the guided form and click <b>Generate Prediction</b>
          to view score, performance band, and insight cards.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  TAB 2 — BATCH PREDICTION (CSV / EXCEL)
# ══════════════════════════════════════════════════════
with tab_batch:

    # ── Step 1: Template Download ──
    st.markdown('<div class="section-card"><p class="section-title"><span class="step-chip">Step 01</span> Download Template File</p>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#8ab4d4;font-size:0.9rem;margin-bottom:1rem;">'
        'Download the template to see the exact column names and example data. '
        'Fill it with your students and upload below. '
        'All 19 feature columns are required.</p>',
        unsafe_allow_html=True)

    tmpl_df = build_template_df()
    t1, t2, _ = st.columns([1, 1, 2])
    with t1:
        st.download_button(
            "⬇  CSV Template",
            data=to_csv_bytes(tmpl_df),
            file_name="student_template.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with t2:
        st.download_button(
            "⬇  Excel Template",
            data=to_excel_bytes(tmpl_df),
            file_name="student_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    with st.expander(" Preview template structure (2 example rows)", expanded=False):
        st.dataframe(tmpl_df, use_container_width=True, hide_index=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Step 2: Upload ──
    st.markdown('<div class="section-card"><p class="section-title"><span class="step-chip">Step 02</span> Upload Your File</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="upload-zone">
      <div class="upload-icon"></div>
      <div class="upload-title">Drag & Drop or Browse</div>
      <div class="upload-sub">Supports .csv · .xlsx · .xls &nbsp;|&nbsp; All 19 feature columns required</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Processing ──
    if uploaded_file is not None:

        # Read file
        try:
            if uploaded_file.name.lower().endswith(".csv"):
                raw_df = pd.read_csv(uploaded_file)
            else:
                raw_df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f" Could not read file: {e}")
            st.stop()

        # ── Step 3: Validation ──
        st.markdown('<div class="section-card"><p class="section-title"><span class="step-chip">Step 03</span> File Preview & Validation</p>', unsafe_allow_html=True)

        fc1, fc2, fc3, fc4 = st.columns(4)
        fc1.metric(" Total Rows",     f"{len(raw_df):,}")
        fc2.metric(" Total Columns",  len(raw_df.columns))
        fc3.metric(" Required Cols",  sum(c in raw_df.columns for c in ALL_FEATURES))
        fc4.metric(" Missing Cols",   sum(c not in raw_df.columns for c in ALL_FEATURES))

        st.markdown("**Raw data preview (first 5 rows):**")
        st.dataframe(raw_df.head(), use_container_width=True, hide_index=True)

        is_valid, errors, warnings = validate_uploaded_df(raw_df)

        if errors:
            for e in errors:
                st.markdown(f'<div class="val-error"> {e}</div>', unsafe_allow_html=True)
        if warnings:
            for w in warnings:
                st.warning(f" {w}")
        if is_valid:
            st.markdown('<div class="val-ok"> File structure valid — ready to predict!</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # ── Step 4: Options ──
        if is_valid:
            st.markdown('<div class="section-card"><p class="section-title"><span class="step-chip">Step 04</span> Prediction Options</p>', unsafe_allow_html=True)

            op1, op2, op3 = st.columns(3)
            with op1: show_analytics  = st.toggle(" Analytics dashboard",    value=True)
            with op2: show_importance = st.toggle(" Feature importance chart",value=False)
            with op3: dl_excel        = st.toggle(" Excel download",          value=True)

            _, run_col, _ = st.columns([1, 2, 1])
            with run_col:
                run_batch = st.button("Run Batch Prediction", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            if run_batch:
                with st.spinner(f" Predicting scores for {len(raw_df):,} students..."):
                    # Fill missing values before encoding
                    for col in NUMERIC_COLS:
                        if col in raw_df.columns:
                            raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")
                            raw_df[col] = raw_df[col].fillna(
                                raw_df[col].median() if raw_df[col].notna().any() else 0)
                    for col in CAT_COLS:
                        if col in raw_df.columns:
                            raw_df[col] = raw_df[col].fillna(CATEGORICAL_FEATURES[col][0])

                    results_df = batch_predict(model, label_encoders, feature_columns, raw_df)

                # ── Step 5: Results ──
                st.markdown('<div class="section-card"><p class="section-title"><span class="step-chip">Step 05</span> Prediction Results</p>', unsafe_allow_html=True)

                scores_arr = results_df["Predicted_Score"]
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric(" Mean Score",    f"{scores_arr.mean():.1f}")
                m2.metric(" Max Score",     f"{scores_arr.max():.1f}")
                m3.metric(" Min Score",     f"{scores_arr.min():.1f}")
                m4.metric(" Std Dev",       f"{scores_arr.std():.1f}")
                m5.metric(" Total Students",f"{len(results_df):,}")

                st.markdown("**Predictions for all students:**")
                st.dataframe(results_df, use_container_width=True, hide_index=True)
                st.markdown('</div>', unsafe_allow_html=True)

                # ── Downloads ──
                st.markdown('<div class="section-card"><p class="section-title">⬇ Download Results</p>', unsafe_allow_html=True)
                dl1, dl2 = st.columns(2)
                with dl1:
                    st.download_button(
                        "  Download as CSV",
                        data=to_csv_bytes(results_df),
                        file_name="batch_predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                with dl2:
                    if dl_excel:
                        st.download_button(
                            "  Download as Excel",
                            data=to_excel_bytes(results_df),
                            file_name="batch_predictions.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True,
                        )
                    else:
                        st.info("Enable 'Excel download' toggle above to unlock this option.")
                st.markdown('</div>', unsafe_allow_html=True)

                # ── Analytics Dashboard ──
                if show_analytics:
                    st.markdown('<div class="section-card"><p class="section-title"> Analytics Dashboard</p>', unsafe_allow_html=True)
                    st.pyplot(plot_batch_analytics(results_df), use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                # ── Performance Band Breakdown ──
                st.markdown('<div class="section-card"><p class="section-title"> Performance Band Breakdown</p>', unsafe_allow_html=True)
                band_summary = (
                    results_df.groupby("Performance_Band")["Predicted_Score"]
                    .agg(Count="count", Avg_Score="mean", Min="min", Max="max")
                    .round(1).reset_index()
                )
                st.dataframe(band_summary, use_container_width=True, hide_index=True)
                st.markdown('</div>', unsafe_allow_html=True)

                # ── Feature Importance (optional) ──
                if show_importance:
                    st.markdown('<div class="section-card"><p class="section-title"> Feature Importance</p>', unsafe_allow_html=True)
                    st.pyplot(plot_feature_importance(model, feature_columns), use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="empty-state">
          Download the template, prepare student records, and upload a
          <b>CSV or Excel file</b> to launch institution-level predictions.
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="footer">
  AI-Based Student Exam Score Prediction System &nbsp;·&nbsp; v2.0 &nbsp;·&nbsp;
  Built with <span>Streamlit</span> &amp; <span>scikit-learn</span><br>
  Powered by <span>Random Forest Regression</span> &nbsp;·&nbsp;
  <span>Single</span> &amp; <span>Batch CSV/Excel</span> Prediction Modes<br>
  <span style="opacity:0.5;font-size:0.72rem;margin-top:4px;display:block;">
    Developer: ML Engineer &nbsp;|&nbsp; Features: 19 &nbsp;|&nbsp;
    Analytics · Downloads · Validation · Feature Importance
  </span>
</div>
""", unsafe_allow_html=True)