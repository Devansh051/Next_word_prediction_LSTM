# import streamlit as st
# import numpy as np
# import pickle

# from tensorflow.keras.models import load_model
# from tensorflow.keras.utils import pad_sequences

# ## load the model
# model = load_model('next_word_lstm.keras')

# ## load the tokenizer
# with open('tokenizer.pickle', 'rb') as file:
#     tokenizer = pickle.load(file)


# ## function to predict the next word
# def predict_next_word(model, tokenizer, text, max_sequence_len):
#     token_list = tokenizer.texts_to_sequences([text])[0]

#     ## this make sure if new sentence > max then it selects max_len_words from last 
#     ## since to predict next_word context from last word play more role.
#     if len(token_list) > max_sequence_len:
#         token_list = token_list[-(max_sequence_len - 1):]

#     token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
#     predicted = model.predict(token_list, verbose=0)

#     predicted_word_index = np.argmax(predicted, axis=1)
#     for word, index in tokenizer.word_index.items():
#         if index == predicted_word_index:
#             return word

#     return None


# ## streamlit app 
# st.title("Next Word Prediction with LSTM")
# input_text = st.text_input("Enter the sequence of words: ", value="To be or not to be")

# if st.button("Predict Next Word"):
#     max_sequence_len=model.input_shape[1]+1
#     next_word=predict_next_word(model, tokenizer, input_text, max_sequence_len)
#     st.write(f"Next word Prediction: {next_word}")



import streamlit as st
import numpy as np
import pickle
import time

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import pad_sequences

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Next Word Prediction | LSTM",
    page_icon="🔮",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Custom CSS — modern, card-based, subtle gradient theme
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%);
        }
        .block-container {
            padding-top: 2.5rem;
            max-width: 780px;
        }
        h1, h2, h3, p, label, span, div {
            color: #e2e8f0 !important;
        }
        .hero-title {
            font-size: 2.4rem;
            font-weight: 800;
            background: linear-gradient(90deg, #818cf8, #c084fc, #f472b6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0;
        }
        .hero-subtitle {
            color: #94a3b8 !important;
            font-size: 1rem;
            margin-top: 0.2rem;
            margin-bottom: 1.8rem;
        }
        .card {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 1.4rem 1.6rem;
            margin-bottom: 1.2rem;
            backdrop-filter: blur(6px);
        }
        .prediction-word {
            font-size: 2.1rem;
            font-weight: 800;
            color: #a5b4fc !important;
        }
        .bar-row {
            display: flex;
            align-items: center;
            gap: 0.7rem;
            margin: 0.35rem 0;
        }
        .bar-label {
            width: 90px;
            font-weight: 600;
            color: #e2e8f0 !important;
        }
        .bar-track {
            flex: 1;
            background: rgba(255,255,255,0.08);
            border-radius: 8px;
            height: 10px;
            overflow: hidden;
        }
        .bar-fill {
            height: 100%;
            border-radius: 8px;
            background: linear-gradient(90deg, #6366f1, #ec4899);
        }
        .bar-pct {
            width: 55px;
            text-align: right;
            font-size: 0.85rem;
            color: #94a3b8 !important;
        }
        .stTextInput input {
            background: rgba(255,255,255,0.06) !important;
            color: #f1f5f9 !important;
            border: 1px solid rgba(255,255,255,0.15) !important;
            border-radius: 10px !important;
            padding: 0.7rem !important;
        }
        .stButton > button {
            background: linear-gradient(90deg, #6366f1, #d946ef);
            color: white !important;
            border: none;
            border-radius: 10px;
            padding: 0.6rem 1.4rem;
            font-weight: 700;
            transition: transform 0.15s ease;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
        }
        .footer-note {
            text-align: center;
            color: #64748b !important;
            font-size: 0.8rem;
            margin-top: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Model / tokenizer loading (cached so it only loads once per session)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading LSTM model...")
def load_artifacts():
    model = load_model("next_word_lstm.keras")
    with open("tokenizer.pickle", "rb") as file:
        tokenizer = pickle.load(file)
    return model, tokenizer


# ----------------------------------------------------------------------------
# Prediction logic
# ----------------------------------------------------------------------------
def predict_top_k(model, tokenizer, text, max_sequence_len, k=5):
    """Return list of (word, probability) tuples for the top-k predicted next words."""
    token_list = tokenizer.texts_to_sequences([text])[0]

    # Keep only the most recent context — recency matters most for next-word prediction
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]

    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding="pre")
    predicted = model.predict(token_list, verbose=0)[0]

    top_indices = np.argsort(predicted)[-k:][::-1]
    index_to_word = {idx: word for word, idx in tokenizer.word_index.items()}

    results = []
    for idx in top_indices:
        word = index_to_word.get(idx)
        if word:
            results.append((word, float(predicted[idx])))
    return results


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    top_k = st.slider("Number of suggestions", min_value=1, max_value=10, value=5)
    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown(
        "This app uses a **Long Short-Term Memory (LSTM)** neural network, "
        "trained on text, to predict the most likely next word given a "
        "sequence of input words."
    )
    st.markdown("**Stack:** TensorFlow / Keras · Streamlit")
    st.markdown("---")
    if "history" not in st.session_state:
        st.session_state.history = []
    if st.session_state.history:
        st.markdown("### 🕓 Recent Predictions")
        for h in reversed(st.session_state.history[-5:]):
            st.markdown(f"- *\"{h['input']}\"* → **{h['prediction']}**")
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown('<p class="hero-title">🔮 Next Word Prediction</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">An LSTM-powered language model that predicts the next word in your sentence.</p>',
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Load model (with a friendly error if files are missing)
# ----------------------------------------------------------------------------
try:
    model, tokenizer = load_artifacts()
    model_ready = True
except Exception as e:
    import traceback

    model_ready = False
    st.exception(e)
    st.code(traceback.format_exc())

# ----------------------------------------------------------------------------
# Main input card
# ----------------------------------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
input_text = st.text_input(
    "Enter a sequence of words",
    value="To be or not to be",
    placeholder="Start typing a sentence...",
)
predict_clicked = st.button("✨ Predict Next Word", disabled=not model_ready)
st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Prediction output
# ----------------------------------------------------------------------------
if predict_clicked and model_ready:
    if not input_text.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Thinking..."):
            max_sequence_len = model.input_shape[1] + 1
            start = time.time()
            predictions = predict_top_k(model, tokenizer, input_text, max_sequence_len, k=top_k)
            elapsed = time.time() - start

        if predictions:
            best_word, best_prob = predictions[0]

            st.markdown('<div class="card">', unsafe_allow_html=True)
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown("**Predicted next word**")
                st.markdown(f'<p class="prediction-word">{best_word}</p>', unsafe_allow_html=True)
            with col2:
                st.metric("Confidence", f"{best_prob*100:.1f}%")
            st.caption(f"Inference time: {elapsed*1000:.0f} ms")
            st.markdown("</div>", unsafe_allow_html=True)

            # Top-k breakdown
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"**Top {len(predictions)} candidates**")
            for word, prob in predictions:
                pct = prob * 100
                st.markdown(
                    f"""
                    <div class="bar-row">
                        <div class="bar-label">{word}</div>
                        <div class="bar-track">
                            <div class="bar-fill" style="width:{pct}%;"></div>
                        </div>
                        <div class="bar-pct">{pct:.1f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

            # Save to history
            st.session_state.history.append({"input": input_text, "prediction": best_word})
        else:
            st.warning("Couldn't generate a prediction for that input — try a different phrase.")

st.markdown(
    '<p class="footer-note">Built with TensorFlow/Keras + Streamlit</p>',
    unsafe_allow_html=True,
)