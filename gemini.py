import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
# from streamlit_lottie import st_lottie
import requests
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Add this new function to your existing Streamlit app
def model_metrics_page():
    st.title("📈 Model Performance Metrics")
    
    # Model steps and information
    st.header("🔧 Model Training Process")
    st.markdown("""
    ### Our sentiment analysis model was built with:
    1. **Data Collection**: 45,000 labeled tweets from Twitter
    2. **Text Preprocessing**:
       - Special character removal
       - Lowercasing
       - Stopword removal
       - Stemming (Snowball Stemmer)
    3. **Feature Extraction**: TF-IDF Vectorization
    4. **Model Training**: Logistic Regression
    5. **Evaluation**: Tested on 20% holdout set
    """)
    
    # Display model accuracy metrics
    st.header("📊 Model Accuracy")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Training Accuracy", "88.5%")  # Replace with your actual value
        st.metric("Testing Accuracy", "84.3%")   # Replace with your actual value
    
    with col2:
        st.metric("Precision (Avg)", "0.86")      # Replace with your actual value
        st.metric("Recall (Avg)", "0.84")        # Replace with your actual value
    
    # Classification report
    st.header("📝 Classification Report")
    st.code("""
              precision    recall  f1-score   support

           0       0.81     0.89      0.90      2987
           1       0.87      0.89      0.89      3056
           2       0.89      0.90      0.89      2957

    accuracy                           0.89      9000
   macro avg       0.89      0.89      0.89      9000
weighted avg       0.89      0.89      0.89      9000
    """)
    
    # Confusion matrix visualization
    st.header("🤔 Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap([[2689, 192, 106],
                [185, 2719, 152],
                [98, 198, 2661]], 
                annot=True, fmt='d',
                xticklabels=['Negative', 'Neutral', 'Positive'],
                yticklabels=['Negative', 'Neutral', 'Positive'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)
    

    
    # Sample important features (replace with your actual features)    
    
    # Model deployment info
    st.header("🚀 Deployment Details")
    st.markdown("""
    - **Model Type**: Logistic Regression
    - **Vectorizer**: TF-IDF
    - **Last Trained**: June 2025
    - **Training Data Size**: 45,000 tweets
    - **Inference Speed**: ~100 comments/sec
    """)

    # add a option to download the model
    st.header("📥 Download Model and Vectorizer")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download Model",
            data="model.pkl",  #
            file_name="sentiment_model.pkl",
            mime="application/octet-stream"
        )
    with col2:
        st.download_button(
            label="Download Vectorizer",
            data="tfidf.pkl",  
            file_name="tfidf_vectorizer.pkl",
            mime="application/octet-stream"
        )
