from __future__ import annotations

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
from deep_translator import GoogleTranslator
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# =============================================================================
# Paths
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
PIPELINE_PATH   = MODEL_DIR / "lgb_pipeline.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "lgb_preprocessor.pkl"
MODEL_PATH      = MODEL_DIR / "lgb_model.pkl"
FEATURES_PATH   = MODEL_DIR / "lgb_features.pkl"

# =============================================================================
# Streamlit page setup
# =============================================================================
st.set_page_config(
    page_title="Restaurant Rating Classification",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =============================================================================
# Custom transformer required for joblib deserialization
# =============================================================================
class ReviewLengthExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        lengths = np.array([len(str(text).split()) for text in X]).reshape(-1, 1)
        self.scaler_ = MinMaxScaler().fit(lengths)
        return self

    def transform(self, X):
        lengths = np.array([len(str(text).split()) for text in X]).reshape(-1, 1)
        return csr_matrix(self.scaler_.transform(lengths))



# =============================================================================
# Styling
# =============================================================================
def inject_styles() -> None:
    
    st.markdown(
        '<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=5.0">',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <style>
        /* ── Fonts ── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="st-"] {
            font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }

        .stApp {
            background: #FAFAFA !important;
            color: #1A1A1A !important;
        }

        .material-symbols-rounded, 
        .material-icons, 
        [data-testid="stIconMaterial"] {
            font-family: 'Material Symbols Rounded' !important;
        }

        /* ── Hide Streamlit chrome ── */
        #MainMenu { visibility: hidden; }
        footer     { visibility: hidden; }
        header     { display: none !important; }
        [data-testid="stSidebar"],
        [data-testid="collapsedControl"] { display: none !important; }

        /* ── Responsive page container ── */
        .block-container {
            max-width: 800px !important;
            margin-left: auto !important;
            margin-right: auto !important;
            padding-top: 2rem !important;
            padding-bottom: 5rem !important;
            padding-left: 1.5rem !important;
            padding-right: 1.5rem !important;
            width: 100% !important;
            box-sizing: border-box !important;
        }

        .stApp, .main, .block-container, .element-container {
            overflow-x: hidden !important;
        }
        img, iframe, video, table, .stDataFrame, .stPlotlyChart, canvas {
            max-width: 100% !important;
        }

        /* ── App header ── */
        .app-header {
            margin-bottom: 1.5rem;
            border-bottom: 1px solid #E5E5E5;
            padding-bottom: 1rem;
        }
        .app-title {
            font-size: clamp(18px, 4vw, 22px);
            font-weight: 600;
            color: #1A1A1A;
            margin: 0 0 4px 0;
            line-height: 1.3;
        }
        .app-subtitle {
            font-size: clamp(12px, 3vw, 13px);
            color: #9AA0A6;
            margin: 0;
            font-weight: 400;
            line-height: 1.4;
        }

        [data-testid="stVerticalBlockBorderWrapper"], 
        [data-testid="stVerticalBlockBorderWrapper"] > div,
        .element-container {
            overflow: visible !important;
        }

        [data-testid="stMarkdownContainer"] {
            padding-top: 4px !important; 
        }

        /* ── Section label (Modern Eyebrow Style) ── */
        .section-label {
            font-size: 18px !important;
            font-weight: 700 !important;
            color: #5F6368 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.1em !important;
            padding-bottom: 4px !important;
            margin-top: 2px !important;
            margin-bottom: 12px !important;
            border-bottom: 1px solid #E5E5E5 !important;
            display: block !important;
            line-height: 1.4 !important;
        }

        .element-container:has(.section-label) {
            overflow: hidden !important;
            height: auto !important;
        }

        /* ── Cards ── */
        .card {
            background: #FFFFFF;
            border: 1px solid #E5E5E5;
            border-radius: 8px;
            padding: 20px 24px;
            margin-bottom: 16px;
        }

        /* ── Result number ── */
        .rating-hero {
            font-size: clamp(36px, 10vw, 48px);
            font-weight: 700;
            color: #1A1A1A;
            line-height: 1;
            margin-bottom: 6px;
        }
        .rating-stars {
            font-size: clamp(18px, 5vw, 22px);
            letter-spacing: 3px;
            margin-bottom: 4px;
        }
        .star-filled { color: #1A1A1A; }
        .star-empty  { color: #D1D5DB; }
        .rating-meta {
            font-size: 13px;
            color: #9AA0A6;
            margin-top: 4px;
        }

        /* ── Confidence bar label ── */
        .conf-label {
            font-size: 13px;
            color: #5F6368;
            margin-bottom: 4px;
        }
        .conf-value {
            font-size: 13px;
            font-weight: 600;
            color: #1A1A1A;
        }

        /* ── Semantic color tags ── */
        .tag {
            display: inline-block;
            font-size: 12px;
            font-weight: 600;
            padding: 3px 12px;
            border-radius: 4px;
            margin-top: 8px;
            margin-bottom: 4px;
        }
        .tag-success { background: #E8F5E9; color: #2E7D32; }
        .tag-warning { background: #FFF8E1; color: #B8860B; }
        .tag-danger  { background: #FDECEA; color: #C0392B; }
        .tag-orange  { background: #FFF3E0; color: #E65100; }

        /* ── Radio button ── */
        [data-testid="stRadio"] [data-testid="stMarkdownContainer"] p {
            font-size: 14px;
        }

        /* Mengubah warna border luar saat dipilih */
        div[data-testid="stRadio"] div[role="radio"][aria-checked="true"] > div:first-child,
        [data-testid="stRadio"] input[type="radio"]:checked + div > div,
        [data-baseweb="radio"] div[aria-checked="true"] > div,
        [data-baseweb="radio"] div[data-checked="true"] > div,
        label[data-baseweb="radio"][aria-checked="true"] > div {
            border-color: #2F5D50 !important;
        }

        /* Mengubah warna titik (dot) bagian dalam saat dipilih */
        div[data-testid="stRadio"] div[role="radio"][aria-checked="true"] > div:first-child > div,
        [data-testid="stRadio"] input[type="radio"]:checked + div > div > div,
        [data-baseweb="radio"] div[aria-checked="true"] > div > div,
        [data-baseweb="radio"] div[data-checked="true"] > div > div,
        label[data-baseweb="radio"][aria-checked="true"] > div > div {
            background-color: #2F5D50 !important;
        }

        /* Mengubah warna teks radio button yang terpilih */
        div[data-testid="stRadio"] div[role="radio"][aria-checked="true"] p,
        [data-testid="stRadio"] label[data-baseweb="radio"][aria-checked="true"],
        [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked),
        [data-testid="stRadio"] input[type="radio"]:checked + div {
            color: #2F5D50 !important;
            font-weight: 600 !important;
        }

        /* Efek saat di-hover */
        div[data-testid="stRadio"] div[role="radio"]:hover > div:first-child,
        div[data-testid="stRadio"] label[data-baseweb="radio"]:hover div {
            border-color: #2F5D50 !important;
        }

        /* ── Inputs ── */
        div[data-testid="stTextArea"] textarea {
            border-radius: 6px !important;
            border: 1px solid #E5E5E5 !important;
            background: #FFFFFF !important;
            font-size: 16px !important;
            color: #1A1A1A !important;
            -webkit-appearance: none;
        }
        div[data-testid="stTextArea"] textarea:focus {
            border-color: #2F5D50 !important;
            box-shadow: 0 0 0 2px rgba(47,93,80,0.12) !important;
        }
        .stSelectbox div[data-baseweb="select"] > div {
            border-radius: 6px !important;
            border: 1px solid #E5E5E5 !important;
            background: #FFFFFF !important;
            font-size: 16px !important;
        }

        /* ── Buttons ── */
        .stButton > button {
            background-color: #2F5D50 !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 6px !important;
            font-size: 14px !important;
            font-weight: 600 !important;
            padding: 0.6rem 1.5rem !important;
            min-height: 2.75rem !important;
            width: 100% !important;
            transition: background-color 0.15s ease !important;
            -webkit-tap-highlight-color: transparent;
        }
        .stButton > button:hover {
            background-color: #264A40 !important;
        }
        .stButton > button:active {
            background-color: #1e3b32 !important;
        }
        .stDownloadButton > button {
            border-radius: 6px !important;
            font-weight: 600 !important;
            min-height: 2.75rem !important;
        }

        /* ── Metric widget ── */
        [data-testid="stMetric"] {
            background: #FFFFFF;
            border: 1px solid #E5E5E5;
            border-radius: 8px;
            padding: 14px 18px;
        }
        [data-testid="stMetricLabel"] { color: #5F6368 !important; font-size: 13px !important; }
        [data-testid="stMetricValue"] { color: #1A1A1A !important; }

        /* ── DataFrames — horizontal scroll on narrow screens ── */
        [data-testid="stDataFrame"] {
            border-radius: 8px !important;
            border: 1px solid #E5E5E5 !important;
            overflow-x: auto !important;
            -webkit-overflow-scrolling: touch;
        }

        /* ── Tabs ── */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0;
            border-bottom: 1px solid #E5E5E5;
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
            scrollbar-width: none;
        }
        .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar { display: none; }
        .stTabs [data-baseweb="tab"] {
            font-size: 14px !important;
            font-weight: 500 !important;
            color: #5F6368 !important;
            padding: 0.55rem 1rem !important;
            white-space: nowrap;
            min-height: 2.75rem;
        }
        .stTabs [aria-selected="true"] {
            color: #1A1A1A !important;
            font-weight: 600 !important;
        }

        /* ── Expander ── */
        [data-testid="stExpander"] {
            border: 1px solid #E5E5E5 !important;
            border-radius: 8px !important;
            background: #FFFFFF !important;
        }
        [data-testid="stExpander"] details summary {
            font-size: 13px;
            font-weight: 600;
            color: #5F6368;
            padding: 0.65rem 1rem;
            min-height: 2.75rem;
            display: flex;
            align-items: center;
        }
        [data-testid="stExpander"] details summary:hover {
            background: #FAFAFA;
        }

        /* ── Word count caption ── */
        .word-count {
            font-size: 12px;
            color: #9AA0A6;
            margin-top: -8px;
            margin-bottom: 8px;
            text-align: right;
        }

        /* ── Footer (fixed bottom bar) ── */
        .app-footer {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            z-index: 999;
            text-align: center;
            color: #9AA0A6;
            font-size: 12px;
            padding: 8px 1rem;
            background: #FAFAFA;
            border-top: 1px solid #E5E5E5;
        }

        /* ── Progress bar ── */
        .custom-progress-wrap {
            background: #E5E5E5;
            border-radius: 4px;
            height: 8px;
            width: 100%;
            margin-top: 6px;
        }
        .custom-progress-fill {
            background: #1A1A1A;
            border-radius: 4px;
            height: 8px;
        }
        .custom-progress-label {
            font-size: 13px;
            font-weight: 600;
            color: #1A1A1A;
            margin-top: 4px;
        }

        /* ================================================================
           RESPONSIVE BREAKPOINTS
           ================================================================ */

        @media screen and (max-width: 768px) {
            .block-container {
                padding-left: 1rem !important;
                padding-right: 1rem !important;
                padding-top: 1.25rem !important;
            }
            .app-title  { font-size: 20px; }
            .rating-hero { font-size: 40px; }
            .card { padding: 16px 18px; }
        }

        @media screen and (max-width: 480px) {
            .block-container {
                padding-left: 0.75rem !important;
                padding-right: 0.75rem !important;
                padding-top: 1rem !important;
                padding-bottom: 3rem !important;
            }
            .app-header { margin-bottom: 1.25rem; }
            .app-title  { font-size: 18px; }
            .app-subtitle { font-size: 12px; }
            .rating-hero { font-size: 36px; }
            .rating-stars { font-size: 18px; }
            .card { padding: 14px 14px; }
            .tag { font-size: 11px; padding: 2px 9px; }
            .stTabs [data-baseweb="tab"] { font-size: 13px !important; padding: 0.5rem 0.75rem !important; }
        }

        @media screen and (max-width: 360px) {
            .block-container {
                padding-left: 0.5rem !important;
                padding-right: 0.5rem !important;
            }
            .app-title { font-size: 16px; }
            .rating-hero { font-size: 32px; }
        }

        @media screen and (min-width: 1200px) {
            .block-container {
                max-width: 860px !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# Helper functions
# =============================================================================
CLASS_LABELS = {
    0: "1 Star",
    1: "2 Stars",
    2: "3 Stars",
    3: "4 Stars",
    4: "5 Stars",
}

STAR_DISPLAY = {
    0: ("1", "★☆☆☆☆"),
    1: ("2", "★★☆☆☆"),
    2: ("3", "★★★☆☆"),
    3: ("4", "★★★★☆"),
    4: ("5", "★★★★★"),
}


def star_label(pred: int) -> tuple[str, str]:
    """Return (display_number, star_icons) for the predicted class."""
    return STAR_DISPLAY.get(pred, ("?", "☆☆☆☆☆"))


def rating_tag(pred: int) -> str:
    tag_map = {
        0: ("tag-danger",  "1 Star"),
        1: ("tag-orange",  "2 Stars"),
        2: ("tag-warning", "3 Stars"),
        3: ("tag-success", "4 Stars"),
        4: ("tag-success", "5 Stars"),
    }
    cls, text = tag_map.get(pred, ("tag-warning", "Unknown"))
    return f'<span class="tag {cls}">{text}</span>'


def normalize_prediction(pred_raw) -> int:
    """Clamp model output to valid range 0–4 (representing 1–5 stars)."""
    try:
        return max(0, min(4, int(pred_raw)))
    except Exception:
        return 0


def word_count(text: str) -> int:
    return len(text.split()) if text.strip() else 0


def translate_to_english(text: str) -> tuple[str, bool]:
    """Translate Indonesian text to English using Google Translate (free).
    Returns (translated_text, was_translated).
    Falls back to original text on any error.
    """
    try:
        translated = GoogleTranslator(source="id", target="en").translate(text)
        return (translated or text), True
    except Exception:
        return text, False


@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load model artifacts once."""
    if not PIPELINE_PATH.exists():
        raise FileNotFoundError(f"Pipeline file not found: {PIPELINE_PATH}")

    pipeline = joblib.load(PIPELINE_PATH)

    preprocessor = None
    lgb_model = None
    feature_names = None

    if PREPROCESSOR_PATH.exists():
        preprocessor = joblib.load(PREPROCESSOR_PATH)
    elif hasattr(pipeline, "named_steps") and "prep" in pipeline.named_steps:
        preprocessor = pipeline.named_steps["prep"]

    if MODEL_PATH.exists():
        lgb_model = joblib.load(MODEL_PATH)
    elif hasattr(pipeline, "named_steps") and "model" in pipeline.named_steps:
        lgb_model = pipeline.named_steps["model"]

    if FEATURES_PATH.exists():
        feature_names = joblib.load(FEATURES_PATH)

    if lgb_model is None:
        raise RuntimeError("LightGBM model could not be loaded.")

    if feature_names is None and preprocessor is not None:
        try:
            feature_names = list(preprocessor.get_feature_names_out())
        except Exception:
            feature_names = None

    explainer = shap.TreeExplainer(lgb_model, feature_perturbation="tree_path_dependent")

    return {
        "pipeline": pipeline,
        "preprocessor": preprocessor,
        "model": lgb_model,
        "feature_names": feature_names,
        "explainer": explainer,
    }


def get_prediction_bundle(text: str, artifacts: dict) -> dict:
    """Return prediction, probability and cleaned text (if possible)."""
    pipeline = artifacts["pipeline"]

    pred_raw  = pipeline.predict([text])[0]
    pred_star = normalize_prediction(pred_raw)

    probabilities = None
    confidence    = None
    class_labels  = None

    if hasattr(pipeline, "predict_proba"):
        proba      = pipeline.predict_proba([text])[0]
        confidence = float(np.max(proba))

        if hasattr(pipeline, "classes_"):
            class_labels = [normalize_prediction(c) for c in pipeline.classes_]
        else:
            class_labels = list(range(len(proba)))

        probabilities = {
            f"{CLASS_LABELS.get(lbl, str(lbl))} (class {lbl})": float(p)
            for lbl, p in zip(class_labels, proba)
        }

    cleaned_text = None
    if hasattr(pipeline, "named_steps") and "prep" in pipeline.named_steps:
        try:
            X_clean      = pipeline.named_steps["prep"].transform([text])
            cleaned_text = str(X_clean)
        except Exception:
            cleaned_text = None

    return {
        "prediction":    pred_star,
        "confidence":    confidence,
        "probabilities": probabilities,
        "cleaned_text":  cleaned_text,
        "class_labels":  class_labels,
    }


def shap_top_words(
    text: str,
    artifacts: dict,
    predicted_class_idx: int,
    top_n: int = 10,
    word_display_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Compute SHAP word contributions.

    Parameters
    ----------
    text : str
        The (English) text that was actually fed to the model.
    word_display_map : dict, optional
        Mapping from English word → display word (e.g. Indonesian original).
        When provided, the "Word" column will show the display word.
    """
    preprocessor  = artifacts["preprocessor"]
    explainer     = artifacts["explainer"]
    feature_names = artifacts["feature_names"]

    if preprocessor is None:
        raise RuntimeError("Preprocessor is required for SHAP explanation.")

    X = preprocessor.transform([text])
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.array(X, dtype=np.float64)

    shap_values = explainer.shap_values(X)
    sv = np.array(shap_values)

    if sv.ndim == 3:
        sv = sv[0, :, predicted_class_idx]
    elif sv.ndim == 2:
        sv = sv[0]
    else:
        sv = sv.ravel()

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(sv))]

    import re
    text_tokens = set(re.findall(r"[a-z]+", text.lower()))

    NON_WORD_PREFIXES = ("char_", "review_length", "length_")

    word_mask = np.array([
        (not any(fn.startswith(p) for p in NON_WORD_PREFIXES))
        and fn in text_tokens
        and sv[i] != 0
        for i, fn in enumerate(feature_names)
    ])

    filtered_sv    = sv[word_mask]
    filtered_names = np.array(feature_names)[word_mask]

    if len(filtered_sv) == 0:
        return pd.DataFrame(columns=["Word", "SHAP Value"])

    pos_idx = np.where(filtered_sv > 0)[0]
    neg_idx = np.where(filtered_sv < 0)[0]

    top_pos = pos_idx[np.argsort(filtered_sv[pos_idx])[::-1]][:5]
    top_neg = neg_idx[np.argsort(filtered_sv[neg_idx])][:5]

    top_idx = np.concatenate([top_pos, top_neg])

    # Build display names: use the Indonesian original when available
    display_names = []
    for en_word in filtered_names[top_idx]:
        if word_display_map and en_word in word_display_map:
            display_names.append(word_display_map[en_word])
        else:
            display_names.append(en_word)

    top_df = pd.DataFrame({
        "Word":       display_names,
        "SHAP Value": filtered_sv[top_idx],
    })
    return top_df


# =============================================================================
# UI
# =============================================================================
def main() -> None:
    inject_styles()

    # ── App header ────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="app-header">
            <div class="app-title">Restaurant Rating Classification</div>
            <div class="app-subtitle">Predict star rating (1–5) from restaurant review text using LightGBM + SHAP</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Load model ────────────────────────────────────────────────────────────
    try:
        artifacts = load_artifacts()
    except Exception as exc:
        st.error(f"Model could not be loaded: {exc}")
        st.info(
            "Make sure `models/lgb_pipeline.pkl`, `models/lgb_preprocessor.pkl`, "
            "`models/lgb_model.pkl`, and `models/lgb_features.pkl` are present."
        )
        st.stop()

    # -------------------------------------------------------------------------
    # Single review
    # -------------------------------------------------------------------------

    # Input card
    with st.container(border=True):
        st.markdown('<div class="section-label">Restaurant review</div>', unsafe_allow_html=True)

        # Language selector
        lang = st.radio(
            "Input language",
            options=["English", "Indonesian"],
            horizontal=True,
            key="lang_select",
        )
        is_indonesian = lang == "Indonesian"

        placeholder = (
            'contoh: "Makanannya enak dan pelayanannya ramah, meski agak lama menunggu."'
            if is_indonesian else
            'e.g. "The food was excellent and the service was friendly, though the wait was a bit long."'
        )

        review_text = st.text_area(
            "Restaurant review",
            height=160,
            placeholder=placeholder,
            label_visibility="collapsed",
            key="review_input",
        )
        wc = word_count(review_text)
        st.markdown(
            f'<div class="word-count">{wc} word{"s" if wc != 1 else ""}</div>',
            unsafe_allow_html=True,
        )
        analyze_clicked = st.button(
            "Predict rating",
            type="primary",
            use_container_width=True,
            key="analyze_btn",
        )

    # ── Result ────────────────────────────────────────────────────────────────
    if analyze_clicked:
        if not review_text.strip():
            st.warning("Enter a review before running the prediction.")
        else:
            text_for_model = review_text
            translated_text = None
            if is_indonesian:
                with st.spinner("Translating to English…"):
                    text_for_model, ok = translate_to_english(review_text)
                    if ok:
                        translated_text = text_for_model
                    else:
                        st.warning("Translation failed — running prediction on original text.")

            with st.spinner("Running prediction…"):
                result = get_prediction_bundle(text_for_model, artifacts)

            pred       = result["prediction"]
            conf       = result["confidence"]
            num, stars = star_label(pred)

            # Primary result card
            with st.container(border=True):
                r_col1, r_col2 = st.columns([1, 1])
                with r_col1:
                    st.markdown(
                        f"""
                        <div style="padding-bottom: 8px;">
                            <div style="margin-bottom: 6px;">
                                <span class="app-subtitle" style="font-size:12px;">Predicted rating</span>
                            </div>
                            <div class="rating-hero">{num}/5</div>
                            <div class="rating-stars" style="font-size:22px;letter-spacing:3px;">{stars}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with r_col2:
                    if conf is not None:
                        pct = f"{conf:.1%}"
                        fill_w = f"{conf * 100:.1f}%"
                        st.markdown(
                            f"""
                            <div class="rating-meta" style="margin-bottom:6px;">Model confidence</div>
                            <div class="custom-progress-wrap">
                                <div class="custom-progress-fill" style="width:{fill_w};"></div>
                            </div>
                            <div class="custom-progress-label">{pct}</div>
                            """,
                            unsafe_allow_html=True,
                        )

            # Probability distribution
            if result.get("probabilities"):
                with st.expander("Rating probability distribution", expanded=False):
                    prob_df = (
                        pd.DataFrame({
                            "Sentiment": list(result["probabilities"].keys()),
                            "Probability": list(result["probabilities"].values()),
                        })
                        .sort_values("Probability", ascending=False)
                        .reset_index(drop=True)
                    )
                    st.dataframe(prob_df, use_container_width=True, hide_index=True)

                    chart_df = prob_df.set_index("Sentiment")
                    st.bar_chart(chart_df, color="#2F5D50")

            # Reviewed text / translation
            if translated_text:
                with st.expander("Translated text (sent to model)", expanded=False):
                    st.caption("Original (Indonesian):")
                    st.code(review_text, language=None)
                    st.caption("Translated (English):")
                    st.code(translated_text, language=None)
            else:
                with st.expander("Reviewed text", expanded=False):
                    st.code(review_text, language=None)

            # SHAP explanation
            st.markdown(
                '<div class="section-label" style="margin-top:8px;">Top contributing words</div>',
                unsafe_allow_html=True,
            )
            try:
                import re
                import matplotlib.pyplot as plt
                from matplotlib.patches import Patch

                predicted_class_idx = int(pred)

                # Build English→Indonesian word mapping when input is Indonesian
                word_display_map = None
                if is_indonesian and translated_text:
                    id_tokens = re.findall(r"[a-zA-Z]+", review_text.lower())
                    en_tokens = re.findall(r"[a-zA-Z]+", translated_text.lower())
                    # Align by position: each Indonesian word maps to the
                    # corresponding English word at the same index.
                    word_display_map = {}
                    for en_w, id_w in zip(en_tokens, id_tokens):
                        if en_w != id_w:          # only map when actually different
                            word_display_map[en_w] = id_w
                        else:
                            word_display_map[en_w] = id_w

                shap_df = shap_top_words(
                    text_for_model, artifacts,
                    predicted_class_idx=predicted_class_idx,
                    top_n=10,
                    word_display_map=word_display_map,
                )
                shap_df = shap_df.sort_values("SHAP Value", ascending=True).reset_index(drop=True)

                if shap_df.empty:
                    st.caption("No word-level SHAP contributions found for this input.")
                else:
                    is_low_rating = pred <= 1  # 1–2 stars

                    def bar_color(v: float) -> str:
                        if is_low_rating:
                            return "#C0392B" if v >= 0 else "#2E7D32"
                        return "#2E7D32" if v >= 0 else "#C0392B"

                    values    = shap_df["SHAP Value"].values
                    labels    = shap_df["Word"].values
                    positions = np.arange(len(labels))
                    colors    = [bar_color(v) for v in values]

                    fig, ax = plt.subplots(figsize=(6, max(3, len(labels) * 0.45)))
                    fig.patch.set_facecolor("#FAFAFA")
                    ax.set_facecolor("#FAFAFA")

                    ax.barh(positions, values, color=colors, height=0.6)
                    ax.axvline(0, color="#D1D5DB", linewidth=0.8)
                    ax.set_yticks(positions)
                    ax.set_yticklabels(labels, fontsize=11)
                    ax.set_xlabel("SHAP value", fontsize=11, color="#5F6368")
                    ax.set_title(
                        f"Words influencing '{CLASS_LABELS.get(pred, str(pred))}' prediction",
                        fontsize=12, fontweight="semibold", color="#1A1A1A", pad=10,
                    )
                    ax.tick_params(colors="#5F6368", labelsize=10)
                    ax.grid(axis="x", linestyle="--", alpha=0.3, color="#D1D5DB")
                    for spine in ["top", "right", "left"]:
                        ax.spines[spine].set_visible(False)
                    ax.spines["bottom"].set_color("#E5E5E5")

                    ax.legend(
                        handles=[
                            Patch(color="#2E7D32", label="Raises rating"),
                            Patch(color="#C0392B", label="Lowers rating"),
                        ],
                        fontsize=9, loc="lower right", framealpha=0.8,
                    )
                    plt.tight_layout()
                    st.pyplot(fig, clear_figure=True)

                    with st.expander("Feature contribution details", expanded=False):
                        display_df = shap_df.copy()
                        display_df["SHAP Value"] = display_df["SHAP Value"].round(4)
                        st.dataframe(
                            display_df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Word": st.column_config.TextColumn(
                                    "Word",
                                    width="medium",
                                ),
                                "SHAP Value": st.column_config.NumberColumn(
                                    "SHAP Value",
                                    format="%.4f",
                                    width="medium",
                                ),
                            },
                        )

            except Exception as exc:
                st.caption(f"SHAP explanation not available for this input: {exc}")

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="app-footer">Pieter Leviano · LightGBM · SHAP · Streamlit</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
