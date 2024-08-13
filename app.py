import streamlit as st
import pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np

# Load the trained model
def load_model():
    with open('reddit_sentiment.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return model

# Predict sentiment of a given text and return probabilities
def predict_sentiment(model, text):
    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]
    return prediction, probabilities

# Generate word cloud from input text
def generate_wordcloud(text):
    wordcloud = WordCloud(background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

# Main function for the Streamlit app
def main():
    st.title("Reddit Sentiment Analysis")

    # Load the trained model
    model = load_model()

    # Sidebar options for input
    st.sidebar.header("Input Options")
    input_mode = st.sidebar.selectbox("Select input mode", ("Type Text", "Upload File"))

    if input_mode == "Type Text":
        # User input for text
        user_input = st.text_area("Enter the text for sentiment analysis:", "")
        
        if st.button("Analyze Sentiment"):
            if user_input:
                # Predict sentiment
                prediction, probabilities = predict_sentiment(model, user_input)
                st.write(f"**Predicted Sentiment:** {prediction}")
                
                # Display sentiment probabilities
                sentiment_labels = model.classes_
                sentiment_probabilities = {label: prob for label, prob in zip(sentiment_labels, probabilities)}
                st.write("**Sentiment Probabilities:**")
                st.write(sentiment_probabilities)

                # Generate and display word cloud
                st.write("**Word Cloud of the Input Text:**")
                generate_wordcloud(user_input)
            else:
                st.error("Please enter text for sentiment analysis.")
    
    elif input_mode == "Upload File":
        uploaded_file = st.file_uploader("Choose a text file", type="txt")
        
        if uploaded_file is not None:
            text = uploaded_file.read().decode("utf-8")
            st.write("**Uploaded Text:**")
            st.write(text)

            if st.button("Analyze Sentiment from File"):
                # Predict sentiment for each line in the file
                predictions = []
                probabilities_list = []

                for line in text.splitlines():
                    if line.strip():  # Skip empty lines
                        prediction, probabilities = predict_sentiment(model, line.strip())
                        predictions.append(prediction)
                        probabilities_list.append(probabilities)

                st.write(f"**Predicted Sentiments:** {predictions}")

                # Show sentiment distribution
                sentiment_counts = np.unique(predictions, return_counts=True)
                sentiment_distribution = dict(zip(sentiment_counts[0], sentiment_counts[1]))
                st.write("**Sentiment Distribution:**")
                st.bar_chart(sentiment_distribution)

                # Generate and display word cloud for the entire text
                st.write("**Word Cloud of the Uploaded Text:**")
                generate_wordcloud(text)

if __name__ == "__main__":
    main()
