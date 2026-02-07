
import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

MODEL_PATH = "respondrAI_model.joblib"
VECTORIZER_PATH = "vectorizer.joblib"

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

model, vectorizer = load_model()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)

    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]

    return " ".join(words)

def assess_severity(tweet):

    high_keywords = ['explosion','massive','collapsed','destroyed','urgent']
    medium_keywords = ['fire','flood','damage','storm']

    tweet = tweet.lower()

    if any(word in tweet for word in high_keywords):
        return "HIGH"
    elif any(word in tweet for word in medium_keywords):
        return "MEDIUM"
    else:
        return "LOW"

def dispatch_agent(incident):

    mapping = {
        'Wildfire': 'Fire Department',
        'Floods': 'Disaster Response Team',
        'Hurricanes': 'Disaster Response Team',
        'Drought': 'Environmental Agency'
    }

    return mapping.get(incident, "General Emergency Services")

st.title("🚨 RespondrAI Disaster Detection")

tweet = st.text_area("Enter Tweet")

if st.button("Analyze"):

    if tweet:

        cleaned = clean_text(tweet)
        vect = vectorizer.transform([cleaned])

        incident = model.predict(vect)[0]
        severity = assess_severity(tweet)
        dispatch = dispatch_agent(incident)

        st.success(f"Incident Type: {incident}")
        st.warning(f"Severity: {severity}")
        st.info(f"Dispatch Unit: {dispatch}")
