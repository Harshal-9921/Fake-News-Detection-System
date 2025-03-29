import streamlit as st
import joblib
import requests
import random
from newspaper import Article  # Web scraping for news articles

# Load trained model and vectorizer
vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("lr_model.jb")

# News API Key (Replace with your own API Key)
NEWS_API_KEY = "840ee63886a44da2a6062e49c75e2b6c"
NEWS_API_URL = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={NEWS_API_KEY}"
NEWS_SEARCH_URL = "https://newsapi.org/v2/everything?q={}&apiKey={}"

# List of news websites for web scraping
NEWS_SITES = [
    "https://www.bbc.com/news",
    "https://edition.cnn.com/world",
    "https://www.nytimes.com/section/world",
    "https://www.aljazeera.com/news/",
    "https://www.reuters.com/news/world/"
]

# Initialize session state
if "article_index" not in st.session_state:
    st.session_state.article_index = 0
if "balanced_articles" not in st.session_state:
    st.session_state.balanced_articles = []
if "search_results" not in st.session_state:
    st.session_state.search_results = []
if "search_index" not in st.session_state:
    st.session_state.search_index = 0

# Navbar (Sidebar)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Search News", "About"])

# Function to fetch and analyze news
def fetch_news(query=None):
    """Fetches latest news from API or searches for a specific query."""
    url = NEWS_SEARCH_URL.format(query, NEWS_API_KEY) if query else NEWS_API_URL
    response = requests.get(url).json()
    articles = response.get("articles", [])

    if articles:
        mixed_articles = []  # Store a mix of real & fake news
        for article in articles:
            if article["description"]:
                text = article["title"] + "\n\n" + article["description"]
                transformed_text = vectorizer.transform([text])
                prediction = model.predict(transformed_text)

                status = "✅ Real News" if prediction[0] == 1 else "❌ Fake News"
                mixed_articles.append((article, status))

        return mixed_articles
    return []

# Home Page
if page == "Home":
    st.title("📰 Fake News Detector")
    st.write("Paste a news article OR fetch live news to check whether it's Fake or Real.")

    # User Input (Manual Pasting)
    input_text = st.text_area("Paste Your News Article Here:", "")

    # Fetch News
    if st.button("Fetch Latest News"):
        st.session_state.balanced_articles = fetch_news()
        st.session_state.article_index = 0  # Reset index

    # Display News
    if st.session_state.balanced_articles:
        article, status = st.session_state.balanced_articles[st.session_state.article_index]
        st.text_area("Fetched News Article:", article["title"] + "\n\n" + article["description"], height=150)
        st.write(f"**Model Prediction:** {status}")

        # Navigation Buttons
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.session_state.article_index > 0:
                if st.button("⬅ Previous"):
                    st.session_state.article_index -= 1
        with col2:
            if st.session_state.article_index < len(st.session_state.balanced_articles) - 1:
                if st.button("Next ➡"):
                    st.session_state.article_index += 1

    # Fake News Detection for User Input
    if st.button("Check News"):
        if input_text.strip():
            transformed_input = vectorizer.transform([input_text])
            prediction = model.predict(transformed_input)

            if prediction[0] == 1:
                st.success("✅ This News is Real!")
            else:
                st.error("❌ This News is Fake!")
        else:
            st.warning("⚠️ Please enter or fetch an article first.")

# News Search Page
elif page == "Search News":
    st.title("🔎 Search News Articles")
    search_query = st.text_input("Enter a topic to search for news:")

    if st.button("Search"):
        st.session_state.search_results = fetch_news(query=search_query)
        st.session_state.search_index = 0

    # Display Search Results
    if st.session_state.search_results:
        article, status = st.session_state.search_results[st.session_state.search_index]
        st.text_area("Search Result:", article["title"] + "\n\n" + article["description"], height=150)
        st.write(f"**Model Prediction:** {status}")

        # Navigation Buttons for Search Results
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.session_state.search_index > 0:
                if st.button("⬅ Previous Search Result"):
                    st.session_state.search_index -= 1
        with col2:
            if st.session_state.search_index < len(st.session_state.search_results) - 1:
                if st.button("Next Search Result ➡"):
                    st.session_state.search_index += 1

# About Page
elif page == "About":
    st.title("📌 About the Fake News Detector")
    st.write(
        """
        This app detects whether a news article is **Fake or Real** using **Machine Learning**.
        
        **How it Works:**
        - Fetch live news or search for a specific topic.
        - The system will automatically analyze if it's **Fake or Real**.
        - The model is trained on real and fake news datasets using **Logistic Regression**.

        **Features:**
        - Fetch live news dynamically from an API.
        - Search for any topic using NewsAPI.
        - Automatic fake news detection.
        - User-friendly interface.

        **Future Enhancements:**
        - Advanced AI models like BERT for better accuracy.
        - More real-time news sources.

        **Developed by:** Harshal
        """
    )
