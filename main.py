from pathlib import Path

import numpy as np
import streamlit as st

MAX_SEQUENCE_LENGTH = 500
MAX_REVIEW_CHARACTERS = 20_000
MODEL_PATH = Path(__file__).resolve().parent / "simple_rnn_imdb.h5"


st.set_page_config(
    page_title="IMDB Movie Review Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_word_index() -> dict[str, int]:
    from tensorflow.keras.datasets import imdb

    return imdb.get_word_index()


@st.cache_resource(show_spinner="Loading the trained SimpleRNN model...")
def load_sentiment_model():
    from tensorflow.keras.models import load_model

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    return load_model(str(MODEL_PATH))


def preprocess_text(text: str, word_index: dict[str, int]) -> np.ndarray:
    from tensorflow.keras.preprocessing import sequence

    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    return sequence.pad_sequences([encoded_review], maxlen=MAX_SEQUENCE_LENGTH)


def confidence_label(confidence: float) -> str:
    if confidence >= 0.90:
        return "Very High Confidence"
    if confidence >= 0.75:
        return "High Confidence"
    if confidence >= 0.60:
        return "Moderate Confidence"
    return "Low Confidence"


def predict_sentiment(review: str) -> dict[str, float | str]:
    cleaned_review = review.strip()
    if not cleaned_review:
        raise ValueError("Enter a review before analyzing it.")
    if len(cleaned_review) > MAX_REVIEW_CHARACTERS:
        raise ValueError(
            f"Reviews must be no longer than {MAX_REVIEW_CHARACTERS:,} characters."
        )

    word_index = load_word_index()
    model = load_sentiment_model()
    padded_review = preprocess_text(cleaned_review, word_index)
    prediction = model.predict(padded_review, verbose=0)

    positive_probability = float(np.asarray(prediction).reshape(-1)[0])
    if not np.isfinite(positive_probability) or not 0 <= positive_probability <= 1:
        raise ValueError("The model returned an invalid prediction.")

    sentiment = "Positive" if positive_probability > 0.5 else "Negative"
    confidence = (
        positive_probability
        if sentiment == "Positive"
        else 1 - positive_probability
    )
    return {
        "sentiment": sentiment,
        "positive_probability": positive_probability,
        "confidence": confidence,
        "confidence_label": confidence_label(confidence),
    }


def set_example_review(review: str) -> None:
    st.session_state["review_input"] = review
    st.session_state["last_result"] = None


def clear_review() -> None:
    st.session_state["review_input"] = ""
    st.session_state["last_result"] = None


def clear_last_result() -> None:
    st.session_state["last_result"] = None


st.markdown(
    """
    <style>
    :root {
        color-scheme: dark;
        --ink: #f3f4f8;
        --muted: #a4a8b8;
        --panel: rgba(22, 25, 38, 0.88);
        --line: rgba(255, 255, 255, 0.09);
        --accent: #f2b45b;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 8% 0%, rgba(116, 54, 86, 0.23), transparent 40%),
            radial-gradient(ellipse at 92% 8%, rgba(45, 89, 121, 0.20), transparent 36%),
            #0d1018;
        color: var(--ink);
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] {
        background: rgba(15, 18, 27, 0.96);
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #d4d6df;
    }
    .main .block-container {
        max-width: 1120px;
        padding-top: 2.4rem;
        padding-bottom: 4rem;
    }
    h1, h2, h3 { color: var(--ink); letter-spacing: -0.035em; }
    h2 { margin-top: 1.8rem; }
    p, li { color: #c2c5d0; }
    .hero {
        padding: clamp(1.5rem, 4vw, 3rem);
        margin: 0 0 1.8rem;
        border: 1px solid var(--line);
        border-radius: 24px;
        background: linear-gradient(130deg, rgba(35, 39, 56, .94), rgba(22, 25, 38, .82));
        box-shadow: 0 24px 70px rgba(0, 0, 0, .24);
    }
    .hero-kicker {
        color: var(--accent);
        font-size: .76rem;
        font-weight: 750;
        letter-spacing: .16em;
        text-transform: uppercase;
    }
    .hero h1 {
        margin: .6rem 0 .7rem;
        font-size: clamp(2rem, 5vw, 3.3rem);
        line-height: 1.08;
    }
    .hero p {
        max-width: 720px;
        margin: 0;
        color: #b9bdca;
        font-size: 1.05rem;
        line-height: 1.7;
    }
    .section-note { color: var(--muted); margin-top: -.5rem; }
    .stTextArea textarea {
        min-height: 220px;
        border: 1px solid rgba(255, 255, 255, .13);
        border-radius: 14px;
        background: rgba(12, 15, 23, .76);
        color: var(--ink);
        line-height: 1.65;
    }
    .stTextArea textarea:focus {
        border-color: rgba(242, 180, 91, .75);
        box-shadow: 0 0 0 1px rgba(242, 180, 91, .25);
    }
    .stButton > button {
        min-height: 2.8rem;
        border: 1px solid var(--line);
        border-radius: 11px;
        font-weight: 650;
        transition: transform .15s ease, border-color .15s ease;
    }
    .stButton > button:hover {
        border-color: rgba(242, 180, 91, .7);
        transform: translateY(-1px);
    }
    div.stButton > button[kind="primary"] {
        border: 0;
        background: linear-gradient(100deg, #e7a347, #f0c77c);
        color: #21170a;
    }
    .result-card {
        margin: 1.2rem 0;
        padding: 1.5rem 1.7rem;
        border: 1px solid var(--line);
        border-radius: 18px;
        background: var(--panel);
    }
    .result-positive { border-left: 4px solid #51c99a; }
    .result-negative { border-left: 4px solid #fa7580; }
    .result-sentiment { margin: 0 0 .4rem; font-size: 1.8rem; font-weight: 760; }
    .result-positive .result-sentiment { color: #72e0b2; }
    .result-negative .result-sentiment { color: #ff929b; }
    .result-copy { margin: 0; color: #c1c5d1; line-height: 1.65; }
    .step-card {
        min-height: 120px;
        margin-bottom: .8rem;
        padding: 1rem;
        border: 1px solid var(--line);
        border-radius: 14px;
        background: rgba(22, 25, 38, .72);
    }
    .step-card strong { color: var(--accent); display: block; margin-bottom: .4rem; }
    .step-card span { color: #b8bdca; font-size: .92rem; line-height: 1.5; }
    .pipeline {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: .55rem;
        margin: 1rem 0 1.4rem;
    }
    .pipeline span {
        padding: .5rem .7rem;
        border: 1px solid var(--line);
        border-radius: 999px;
        background: rgba(22, 25, 38, .78);
        color: #e3e5eb;
        font-size: .86rem;
    }
    .pipeline b { color: var(--accent); }
    [data-testid="stProgressBar"] > div > div { background-color: #e9ae58; }
    @media (max-width: 700px) {
        .main .block-container { padding: 1.2rem 1rem 3rem; }
        .hero { border-radius: 18px; }
        .stTextArea textarea { min-height: 190px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("## 🎞️ Model details")
st.sidebar.caption("Inference configuration")
st.sidebar.markdown(
    """
    **Model:** SimpleRNN  
    **Dataset:** IMDB Movie Reviews  
    **Vocabulary Size:** 10,000  
    **Embedding Dimension:** 128  
    **Maximum Sequence Length:** 500  
    **Output:** Binary Sentiment  
    **SimpleRNN Activation:** tanh  
    **Output Activation:** sigmoid
    """
)
st.sidebar.divider()
st.sidebar.caption(
    "Predictions are generated by the existing trained model. "
    "Confidence describes the model's output, not a guarantee of correctness."
)

st.markdown(
    """
    <section class="hero">
      <div class="hero-kicker">A closer look at every review</div>
      <h1>🎬 IMDB Movie Review Sentiment Analyzer</h1>
      <p>Explore whether a movie review reads as positive or negative, using a trained
      SimpleRNN model and the original IMDB word-index encoding.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

st.subheader("Try an example")
st.markdown(
    '<p class="section-note">Choose a sample to fill the review box and analyze it.</p>',
    unsafe_allow_html=True,
)
examples = [
    (
        "😊 Clearly positive",
        "An absolutely wonderful film. The performances were compelling, the story "
        "was beautifully told, and I loved every minute of it.",
    ),
    (
        "😞 Clearly negative",
        "A dull and frustrating movie with wooden acting, a predictable plot, and "
        "a painfully slow pace. I would not recommend it.",
    ),
    (
        "🤔 Mixed / neutral",
        "The cinematography was impressive and a few scenes were entertaining, but "
        "the characters felt thin and the ending did not quite work.",
    ),
    ("⚡ Short review", "Surprisingly good."),
]
example_columns = st.columns(4)
for column, (label, example) in zip(example_columns, examples):
    with column:
        st.button(
            label,
            key=f"example_{label}",
            width="stretch",
            on_click=set_example_review,
            args=(example,),
        )

st.subheader("Your review")
st.markdown(
    '<p class="section-note">The model analyzes the words it recognizes and maps unknown words to its existing unknown-token ID.</p>',
    unsafe_allow_html=True,
)
if "review_input" not in st.session_state:
    st.session_state["review_input"] = ""
if "prediction_history" not in st.session_state:
    st.session_state["prediction_history"] = []
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

review = st.text_area(
    "Movie review",
    key="review_input",
    height=230,
    max_chars=MAX_REVIEW_CHARACTERS,
    placeholder=(
        "Example: The movie was absolutely amazing. The acting, story and direction "
        "were excellent..."
    ),
    label_visibility="collapsed",
    on_change=clear_last_result,
)
trimmed_review = review.strip()
word_count = len(trimmed_review.split())
st.caption(
    f"{len(review):,} / {MAX_REVIEW_CHARACTERS:,} characters · {word_count:,} words"
)

analyze_column, clear_column = st.columns([3, 1])
with analyze_column:
    analyze_clicked = st.button(
        "Analyze Review",
        type="primary",
        width="stretch",
        disabled=not bool(trimmed_review),
    )
with clear_column:
    st.button(
        "Clear",
        width="stretch",
        disabled=not bool(review),
        on_click=clear_review,
    )

if analyze_clicked:
    try:
        result = predict_sentiment(trimmed_review)
        st.session_state["last_result"] = result
        st.session_state["prediction_history"].insert(
            0,
            {
                "Review": trimmed_review,
                "Sentiment": result["sentiment"],
                "Positive probability": f"{result['positive_probability']:.1%}",
            },
        )
    except FileNotFoundError as error:
        st.session_state["last_result"] = None
        st.error(
            "The trained model file could not be found. Place "
            "`simple_rnn_imdb.h5` beside `main.py` and try again."
        )
        with st.expander("Technical details"):
            st.code(str(error))
    except Exception as error:
        st.session_state["last_result"] = None
        st.error(
            "We couldn't analyze this review. Please try again; if the problem "
            "continues, check that the model and TensorFlow installation are available."
        )
        with st.expander("Technical details"):
            st.code(str(error))

result = st.session_state["last_result"]
if result is not None:
    positive = result["sentiment"] == "Positive"
    icon = "😊" if positive else "😞"
    result_class = "result-positive" if positive else "result-negative"
    probability_label = "positive" if positive else "negative"
    st.markdown(
        f"""
        <div class="result-card {result_class}">
          <p class="result-sentiment">{icon} {result["sentiment"]}</p>
          <p class="result-copy">
            The model predicts this review is {result["sentiment"].lower()} with
            {result["confidence"]:.1%} model confidence.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"**Prediction confidence · {result['confidence_label']}**  \n"
        f"Model probability for {probability_label}: "
        f"**{result['confidence']:.1%}** · Positive probability: "
        f"**{result['positive_probability']:.1%}**"
    )
    st.progress(
        result["confidence"],
        text=f"Model confidence: {result['confidence']:.1%}",
    )

st.divider()
st.subheader("How it works")
st.markdown(
    """
    <div class="pipeline">
      <span>Review Text</span><b>→</b><span>Tokenization</span><b>→</b>
      <span>IMDB Word Index</span><b>→</b><span>Integer Encoding</span><b>→</b>
      <span>Padding</span><b>→</b><span>Embedding Layer</span><b>→</b>
      <span>SimpleRNN</span><b>→</b><span>Sigmoid</span><b>→</b>
      <span>Positive / Negative</span>
    </div>
    """,
    unsafe_allow_html=True,
)
pipeline_steps = [
    ("1 · Review text", "You enter a movie review. Leading and trailing whitespace is removed."),
    ("2 · Tokenization", "The text is lowercased and split into words, matching this app's existing preprocessing."),
    ("3 · IMDB word index", "Each recognized word is looked up in the IMDB dataset's vocabulary."),
    ("4 · Integer encoding", "Words become the same integer IDs used by the trained model; unknown words use its unknown-token ID."),
    ("5 · Padding", "The encoded review is padded or truncated to exactly 500 tokens."),
    ("6 · Embedding layer", "The model maps token IDs to learned 128-dimensional word representations."),
    ("7 · SimpleRNN", "The recurrent layer processes the sequence using its trained tanh activation."),
    ("8 · Sigmoid", "The output layer returns a score between 0 and 1 for the positive class."),
    ("9 · Sentiment", "A score above 0.5 is classified as positive; otherwise it is classified as negative."),
]
for row_start in range(0, len(pipeline_steps), 3):
    columns = st.columns(3)
    for column, (title, description) in zip(
        columns, pipeline_steps[row_start : row_start + 3]
    ):
        with column:
            st.markdown(
                f'<div class="step-card"><strong>{title}</strong>'
                f"<span>{description}</span></div>",
                unsafe_allow_html=True,
            )

st.divider()
history_column, history_action_column = st.columns([3, 1])
with history_column:
    st.subheader("Prediction history")
    st.markdown(
        '<p class="section-note">Stored only in this browser session and cleared when the session ends.</p>',
        unsafe_allow_html=True,
    )
with history_action_column:
    st.write("")
    if st.button(
        "Clear History",
        width="stretch",
        disabled=not st.session_state["prediction_history"],
    ):
        st.session_state["prediction_history"] = []
        st.rerun()

if st.session_state["prediction_history"]:
    st.dataframe(
        st.session_state["prediction_history"],
        hide_index=True,
        width="stretch",
    )
else:
    st.caption("Your analyzed reviews will appear here.")
