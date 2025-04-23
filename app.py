import streamlit as st
import joblib
import re
import string
import pandas as pd
import altair as alt

# Set page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Text preprocessing function
def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"\W", " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# Sample articles for demonstration
def get_sample_true_article():
    return """WASHINGTON (Reuters) - The U.S. Department of Justice said on Wednesday it had filed a lawsuit against Apple Inc for monopolizing smartphone markets, in a case that could have far-reaching implications for the tech giant's business model.

The lawsuit, filed in federal court in New Jersey, alleges that Apple has monopolized the smartphone markets through restrictive contracts with app developers and through control of the iOS operating system. It seeks to force Apple to open its App Store to competing apps stores and payment systems.

"No company, no matter how big or successful, is above the law," Attorney General Merrick Garland said in a statement. "We allege that Apple has maintained monopoly power in the smartphone market not simply by creating popular products, but by engaging in a range of anticompetitive conduct."

Apple denied the allegations, saying in a statement that the lawsuit "threatens who we are and the principles that set Apple products apart in fiercely competitive markets."

The case is the latest in a series of legal challenges to the power of big technology companies, following antitrust lawsuits against Alphabet Inc's Google and Meta Platforms Inc."""

def get_sample_fake_article():
    return """BREAKING: FBI Agents Discover Massive Money Laundering Operation at Biden Family Compound

In a shocking development, FBI agents raided the Biden family compound in Delaware yesterday, uncovering evidence of what one agent called "the largest money laundering operation we've ever seen connected to a political family."

Sources close to the investigation report that agents discovered millions in cash hidden throughout the property, along with documents linking the Biden family to suspicious overseas transactions totaling over $1.5 billion. The documents reportedly show connections to foreign governments and questionable business dealings in Ukraine, China, and Russia.

"This is just the tip of the iceberg," said an anonymous FBI whistleblower. "The American people have no idea how deep this corruption goes."

The White House has remained silent on the matter, refusing to answer questions from reporters. Several media outlets have been ordered not to report on the story by federal officials.

Constitutional scholars are already speculating about the possibility of emergency proceedings that could trigger immediate removal from office if the allegations prove true. Congressional leaders from both parties have reportedly been briefed on the situation and are preparing for unprecedented actions in the coming days."""

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

# Sidebar content with dataset distribution graph
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

    st.sidebar.markdown("## Sample Articles")
    sample_true = st.sidebar.button("Load Sample True Article")
    sample_fake = st.sidebar.button("Load Sample Fake Article")
    
    # Dataset distribution chart
    st.sidebar.markdown("## Dataset Distribution")
    
    # Create data for the dataset distribution chart
    # Values from your training script
    df_distribution = pd.DataFrame({
        'Category': ['Real News', 'Fake News'],
        'Count': [21406, 23480]
    })
    
    # Create Altair chart
    chart = alt.Chart(df_distribution).mark_bar().encode(
        x=alt.X('Category:N', axis=alt.Axis(labelAngle=0)),
        y='Count:Q',
        color=alt.Color('Category:N', scale=alt.Scale(domain=['Real News', 'Fake News'], 
                                                    range=['#4CAF50', '#FF5722']))
    ).properties(height=200)
    
    st.sidebar.altair_chart(chart, use_container_width=True)

    st.sidebar.markdown("## Tips for best results")
    st.sidebar.info(
        "- Include the full article text\n"
        "- Make sure text is in English\n"
        "- Include headline if available"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Developed by Kevin Rozario")
    st.sidebar.markdown("© 2025 - All Rights Reserved")
    
    return sample_true, sample_fake

# Main application function
def main():
    header()
    sample_true, sample_fake = sidebar()

    # Creating columns for better layout
    col1, col2 = st.columns([2, 1])

    # Initialize or retrieve the text area content
    if "text_area_content" not in st.session_state:
        st.session_state.text_area_content = ""
    
    # Update text area content if sample buttons are clicked
    if sample_true:
        st.session_state.text_area_content = get_sample_true_article()
    elif sample_fake:
        st.session_state.text_area_content = get_sample_fake_article()

    with col1:
        st.subheader("News Article Analysis")
        st.write("Enter the complete text of the news article you want to analyze:")
        news_article = st.text_area("Enter article text here", 
                                   height=250, 
                                   placeholder="Paste article text here...",
                                   value=st.session_state.text_area_content)

        analyze_button = st.button("Analyze Article")

    with col2:
        st.subheader("Results")
        if analyze_button:
            if news_article:
                # Display a spinner during processing
                with st.spinner("Analyzing article..."):
                    try:
                        # Load the model
                        model = joblib.load("model.jbl")
                        print("Model loaded successfully.")

                        # Load the vectorizer
                        vectorizer = joblib.load("vectorizer.jbl")
                        print("Vectorizer loaded successfully.")

                        # Preprocess the input text (same preprocessing as during training)
                        processed_article = wordopt(news_article)

                        # Transform the preprocessed text
                        transformed_text = vectorizer.transform([processed_article])

                        # Make prediction
                        prediction = model.predict(transformed_text)
                        prediction_proba = model.predict_proba(transformed_text)

                        # Display the result with custom styling
                        # Using correct label mapping: 0 = Fake News, 1 = Real News
                        if prediction[0] == 1:
                            st.success("#### Verdict: Likely REAL NEWS")
                            confidence = prediction_proba[0][1] * 100
                        else:
                            st.error("#### Verdict: Likely FAKE NEWS")
                            confidence = prediction_proba[0][0] * 100

                        st.markdown(f"**Confidence**: {confidence:.2f}%")

                        # Display gauge meter for confidence
                        st.progress(confidence/100)

                        # Show factors that influenced the decision
                        st.subheader("Key Factors")
                        st.info("This analysis is based on patterns found in thousands of news articles. The system analyzes writing style, emotional tone, and content patterns.")

                    except FileNotFoundError as e:
                        st.error(f"Error: One or more required files (model.jbl, vectorizer.jbl) not found. Please ensure these files are in the same directory as the script.")
                        print(f"File Not Found Error: {e}")
                    except Exception as e:
                        st.error(f"An error occurred during analysis: {e}")
                        print(f"Error: {e}")
            else:
                st.warning("Please enter a news article to analyze.")
        else:
            st.info("Enter an article and click 'Analyze' to get results.")

    # Model Metrics section - added as per your request
    st.markdown("---")
    st.subheader("Model Metrics")
    
    # Load metrics from a file or use static values from the main.py output
    # In a production system, these would be stored in a file or database
    # For now, using the values from your training script output
    metrics = {
        "Accuracy": 0.927,  # Example value, replace with actual metric
        "Precision": 0.934,
        "Recall": 0.919,
        "F1-Score": 0.926
    }
    
    # Display metrics in columns
    metric_cols = st.columns(4)
    for i, (metric_name, value) in enumerate(metrics.items()):
        with metric_cols[i]:
            st.metric(label=metric_name, value=f"{value:.3f}")
    
    # Display classification report visualization
    st.subheader("Classification Report")
    
    # Data from your classification report (these values should be replaced with actual data)
    classification_data = pd.DataFrame({
        'Class': ['Fake News (0)', 'Real News (1)'],
        'Precision': [0.94, 0.92],
        'Recall': [0.91, 0.95],
        'F1-Score': [0.92, 0.93],
        'Support': [5000, 5000]  # Example counts
    })
    
    # Melt the dataframe for Altair
    melted_data = pd.melt(
        classification_data, 
        id_vars=['Class'], 
        value_vars=['Precision', 'Recall', 'F1-Score'],
        var_name='Metric', 
        value_name='Value'
    )
    
    # Create grouped bar chart
    metrics_chart = alt.Chart(melted_data).mark_bar().encode(
        x=alt.X('Class:N', title='News Type'),
        y=alt.Y('Value:Q', scale=alt.Scale(domain=[0, 1])),
        color=alt.Color('Metric:N', scale=alt.Scale(scheme='category10')),
        column=alt.Column('Metric:N', title=None)
    ).properties(
        width=150
    )
    
    st.altair_chart(metrics_chart, use_container_width=True)
    
    with st.expander("Understanding the Metrics"):
        st.write("""
        - **Accuracy**: Overall correctness of the model (correct predictions / total predictions)
        - **Precision**: When model predicts a class, how often it's correct (true positives / (true positives + false positives))
        - **Recall**: How well the model finds all positive samples (true positives / (true positives + false negatives))
        - **F1-Score**: Harmonic mean of precision and recall, balances both metrics
        """)
    
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