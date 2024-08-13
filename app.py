import streamlit as st
import pickle

# Load the trained model
def load_model():
    with open('reddit_sentiment.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return model

# Predict sentiment of a given text
def predict_sentiment(model, text):
    return model.predict([text])[0]

# Main function for the Streamlit app
def main():
    st.title("Reddit Sentiment Analysis")
    
    # Load the trained model
    model = load_model()

    # User input for text
    user_input = st.text_area("Enter the text for sentiment analysis:", "")
    
    if st.button("Analyze Sentiment"):
        if user_input:
            # Predict sentiment
            prediction = predict_sentiment(model, user_input)
            st.write(f"**Predicted Sentiment:** {prediction}")
        else:
            st.error("Please enter text for sentiment analysis.")

if __name__ == "__main__":
    main()
