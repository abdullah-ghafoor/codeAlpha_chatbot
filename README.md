# CodeAlpha Internship
## Task02 FAQ Chatbot
This is a  chatbot that answers frequently asked questions from your own local dataset using TF-IDF and cosine similarity. If no relevant answer is found, it fetches information from Wikipedia as a fallback source.
## Features
- TF-IDF based question matching
- Wikipedia fallback for unmatched queries
- Preprocessing using NLTK
- Streamlit-based clean chat UI
## Requirements
- streamlit
- pandas
- scikit-learn
- nltk
- streamlit-chat
- wikipedia
## Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/abdullah-ghafoor/codeAlpha_chatbot.git
cd codeAlpha_chatbot
pip install -r requirements.txt
 
streamlit run chatbot.py
