from pathlib import Path
import pickle
from typing import Any

import streamlit as st
from keras.models import load_model
from keras.utils import pad_sequences


MODEL_PATH = Path(__file__).parent / "gru_model.h5"
TOKENIZER_PATH = Path(__file__).parent / "tokenizer.pkl"
MAX_LEN_PATH = Path(__file__).parent / "max_len.pkl"


@st.cache_resource(show_spinner="Loading the GRU model and vocabulary...")
def load_prediction_assets() -> tuple[Any, Any, int]:
    model = load_model(MODEL_PATH, compile=False)
    with TOKENIZER_PATH.open("rb") as tokenizer_file:
        tokenizer = pickle.load(tokenizer_file)
    with MAX_LEN_PATH.open("rb") as max_len_file:
        max_length = int(pickle.load(max_len_file))
    return model, tokenizer, max_length


def predict_words(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_length: int,
    word_count: int,
) -> list[str]:
    token_ids = tokenizer.texts_to_sequences([prompt.lower()])[0]
    generated_words: list[str] = []

    for _ in range(word_count):
        padded_ids = pad_sequences(
            [token_ids],
            maxlen=max_length,
            padding="pre",
            truncating="pre",
        )
        probabilities = model.predict(padded_ids, verbose=0)
        predicted_id = int(probabilities[0].argmax())
        predicted_word = tokenizer.index_word.get(predicted_id)
        if predicted_word is None:
            break

        generated_words.append(predicted_word)
        token_ids.append(predicted_id)

    return generated_words


st.set_page_config(
    page_title="Next Word Predictor",
    layout="centered",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(160deg, #f7f8ff 0%, #f8fbff 55%, #f5faff 100%);
    }
    .block-container {
        max-width: 900px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }
    .hero {
        padding: 2rem 2.2rem;
        border: 1px solid #e5e9ff;
        border-radius: 24px;
        background: linear-gradient(120deg, #ffffff 0%, #f0f2ff 100%);
        box-shadow: 0 14px 38px rgba(43, 55, 110, 0.08);
        margin-bottom: 1.5rem;
    }
    .eyebrow {
        color: #5b62c8;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .hero h1 {
        color: #202446;
        font-size: clamp(2rem, 5vw, 3rem);
        line-height: 1.1;
        margin: 0;
    }
    .hero p {
        color: #606782;
        font-size: 1.05rem;
        margin: 0.8rem 0 0;
    }
    div.stButton > button[kind="primary"] {
        border: 0;
        border-radius: 12px;
        background: linear-gradient(100deg, #625cf5, #8274fa);
        font-weight: 700;
        min-height: 3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">GRU | Language model</div>
        <h1>What comes next?</h1>
        <p>Give the model a starting phrase and let it continue, one word at a time.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

missing_files = [
    path.name for path in (MODEL_PATH, TOKENIZER_PATH, MAX_LEN_PATH) if not path.is_file()
]
if missing_files:
    st.error(
        "The following model files are missing from the DL folder: "
        + ", ".join(missing_files)
    )
    st.stop()

with st.sidebar:
    st.subheader("About this demo")
    st.write(
        "This app uses your saved GRU model and tokenizer to predict the most likely "
        "next word for the text you enter."
    )
    st.caption("Model files: gru_model.h5 · tokenizer.pkl · max_len.pkl")

prompt = st.text_area(
    "Starting text",
    placeholder="For example: the world as we have",
    height=120,
    help="Enter one or more words for the model to continue.",
)
word_count = st.slider("Words to generate", min_value=1, max_value=20, value=10)

if st.button("Predict next word", type="primary", use_container_width=True):
    if not prompt.strip():
        st.warning("Enter a starting phrase to get a prediction.")
    else:
        model, tokenizer, max_length = load_prediction_assets()
        input_tokens = tokenizer.texts_to_sequences([prompt.lower()])[0]
        if not input_tokens:
            st.warning(
                "The tokenizer could not find any known words in that phrase. "
                "Try different words."
            )
        else:
            generated_words = predict_words(
                model,
                tokenizer,
                prompt,
                max_length,
                word_count,
            )
            if not generated_words:
                st.info("The model did not return a word for this phrase.")
            else:
                st.markdown("#### Prediction")
                st.write(f"{prompt.rstrip()} {' '.join(generated_words)}")
