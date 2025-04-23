import streamlit as st
# import joblib
import numpy as np
import pandas as pd
import streamlit.components.v1 as components

# Set page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for orange color scheme
def add_custom_css():
    st.markdown("""
    <style>
    .main {
        background-color: #FFF8F0;
        color: #994D00;
    }
    .stButton>button {
        background-color: #FF8C00;
        color: white;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #E67300;
        color: white;
    }
    .stTextArea textarea {
        border-color: #FF8C00;
        border-radius: 6px;
    }
    h1, h2, h3 {
        color: #994D00;
    }
    .sidebar .sidebar-content {
        background-color: #994D00;
        color: white;
    }
    .st-eb {
        background-color: #FF8C00;
    }
    .css-1d391kg {
        background-color: #994D00;
    }
    div.stAlert > div {
        background-color: #FFD6AD;
        color: #994D00;
        border: none;
        border-radius: 6px;
    }
    footer {display: none;}
    </style>
    """, unsafe_allow_html=True)

add_custom_css()

# App header with custom design
def header():
    st.title("📰 Fake News Detection")
    st.markdown("<p style='font-size: 20px; color: #FF8C00;'>Identify misinformation with machine learning</p>", unsafe_allow_html=True)
    st.markdown("---")

# Sidebar content
def sidebar():
    st.sidebar.markdown("## About")
    st.sidebar.info(
        "This application uses machine learning to analyze and " 
        "classify news articles as potentially real or fake. "
        "The model has been trained on thousands of labeled news articles."
    )
    
    st.sidebar.markdown("## How to use")
    st.sidebar.info(
        "1. Paste your news article in the text area\n"
        "2. Click the 'Analyze' button\n" 
        "3. View the prediction and confidence score"
    )
    
    st.sidebar.markdown("## Tips for best results")
    st.sidebar.info(
        "- Include the full article text\n"
        "- Make sure text is in English\n"
        "- Include headline if available"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Developed by Kevin Rozario")
    st.sidebar.markdown("© 2025 - All Rights Reserved")

# Main application function    
def main():
    header()
    sidebar()
    
    # Creating columns for better layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("News Article Analysis")
        st.write("Enter the complete text of the news article you want to analyze:")
        news_article = st.text_area("", height=250, placeholder="Paste article text here...")
        
        analyze_button = st.button("Analyze Article")
        
    with col2:
        st.subheader("Results")
        if analyze_button:
            if news_article:
                # Display a spinner during processing
                with st.spinner("Analyzing article..."):
                    try:
                        # Load the model
                        model = joblib.load("model.pkl")
                        
                        # Load the vectorizer (uncomment when files are available)
                        vectorizer = joblib.load("vectorizer.pkl")
                        
                        # Transform the input text
                        transformed_text = vectorizer.transform([news_article])
                        
                        # Make prediction
                        prediction = model.predict(transformed_text)
                        prediction_proba = model.predict_proba(transformed_text)
                        
                        # Display the result with custom styling
                        if prediction[0] == 1:
                            st.error("#### Verdict: Likely FAKE NEWS")
                            confidence = prediction_proba[0][1] * 100
                        else:
                            st.success("#### Verdict: Likely REAL NEWS")
                            confidence = prediction_proba[0][0] * 100
                        
                        st.markdown(f"**Confidence**: {confidence:.2f}%")
                        
                        # Display gauge meter for confidence
                        st.progress(confidence/100)
                        
                        # Show factors that influenced the decision
                        st.subheader("Key Factors")
                        st.info("This analysis is based on patterns found in thousands of news articles. The system analyzes writing style, emotional tone, and content patterns.")
                        
                    except Exception as e:
                        st.error(f"An error occurred during analysis. Please try again.")
            else:
                st.warning("Please enter a news article to analyze.")
        else:
            st.info("Enter an article and click 'Analyze' to get results.")
    
    # Features section below the main content
    st.markdown("---")
    st.subheader("How it works")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📊 Analysis")
        st.write("Our model analyzes article text using natural language processing techniques.")
    
    with col2:
        st.markdown("### 🤖 Machine Learning")
        st.write("Trained on thousands of verified real and fake news articles to identify patterns.")
    
    with col3:
        st.markdown("### 🔍 Results")
        st.write("Get instant feedback on the credibility of news articles with confidence scores.")

# Run the application
if __name__ == "__main__":
    main()