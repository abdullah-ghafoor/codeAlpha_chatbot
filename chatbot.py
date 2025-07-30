import nltk
import streamlit as st
from streamlit_chat import message
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

#STEP 1: Load Dataset 
data = pd.read_csv('chatbot_dataset.csv')

corpus_pairs = list(zip(data['pattern'], data['response']))  # Removed .lower() for proper preprocessing

# ===================== STEP 2: NLP Setup ==========================
# Download required NLTK resources

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Setup tokenizer, stopwords, and lemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Preprocessing function (NEW ✅)
def preprocess(text):
    tokens = word_tokenize(text.lower())
    filtered = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(filtered)

# Preprocess all questions (NEW ✅)
user_inputs = [preprocess(pair[0]) for pair in corpus_pairs]
responses = [pair[1] for pair in corpus_pairs]

# ===================== STEP 3: Vectorization ==========================
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(user_inputs)

# ===================== STEP 4: Matching Logic ================== 
import wikipedia

# Wikipedia fallback function
def search_wikipedia(query):
    try:
        summary = wikipedia.summary(query, sentences=2)
        return f"📚 *From Wikipedia:*\n{summary}"
    except Exception:
        return None  # If no result found


def chatbot_response(user_input):
    user_input_processed = preprocess(user_input)
    user_vector = vectorizer.transform([user_input_processed])
    similarity_scores = cosine_similarity(user_vector, vectors)
    most_similar_index = similarity_scores.argmax()
    max_similarity = similarity_scores[0, most_similar_index]
    threshold = 0.3

    if max_similarity < threshold:
        # Step 1: Try Wikipedia
        wiki_answer = search_wikipedia(user_input)
        if wiki_answer:
            return wiki_answer
        else:
            return "🤖 I couldn't find a direct answer. Please try rephrasing or ask a different way."
    else:
        return responses[most_similar_index]

# ===================== STEP 5: Streamlit UI ==========================

st.set_page_config(
    page_title="AI ChatBot",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
    <style>
        /* Light background and modern UI */
        body {
            background-color: #f5f7fa;
            color: #1c1e21;
        }

        .block-container {
            padding: 2rem;
        }

        .stTextInput > div > div > input {
            background-color: #ffffff;
            color: #000000;
            border: 1px solid #d1d1d1;
            border-radius: 12px;
            padding: 10px;
        }

        .stButton > button {
            background-color: #007BFF;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 10px 20px;
            font-weight: bold;
        }

        .stButton > button:hover {
            background-color: #0056b3;
        }

        .header {
            color: #2c3e50;
            font-size: 30px;
            margin-bottom: 1rem;
            text-align: center;
        }

        .sidebar .sidebar-content {
            background-color: #e9ecef;
            padding: 1.5rem;
            border-radius: 10px;
        }

        .sidebar-header {
            font-size: 24px;
            color: #007BFF;
            font-weight: bold;
            margin-bottom: 1rem;
        }

        .sidebar-subtext {
            font-size: 15px;
            color: #333;
            margin-bottom: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# ===== Session history =====

if 'history' not in st.session_state:
    st.session_state['history'] = []

with st.sidebar:
    st.markdown("<div class='sidebar-header'>💬 FAQ Chat Assistant</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-subtext'>Ask me anything from the FAQ database. I’ll try to find the most relevant answer.</div>", unsafe_allow_html=True)
    
    st.markdown("### 📌 Try asking:")
    for q in data['pattern'][:3]:
        st.markdown(f"• *{q.capitalize()}*")
    
    st.markdown("---")
    st.markdown("Built with dedication")

st.markdown("<h1 class='header'>💡 Smart FAQ Chatbot</h1>", unsafe_allow_html=True)

for i, (sender, message_text) in enumerate(st.session_state['history']):
    if sender == "user":
        message(message_text, is_user=True, key=f"user_{i}")
    else:
        message(message_text, key=f"bot_{i}")

user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state['history'].append(("user", user_input))
    bot_response = chatbot_response(user_input)
    st.session_state['history'].append(("bot", bot_response))
    st.rerun()

