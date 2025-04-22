import pickle
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

# Global variables
project_name = "Sentiment Analysis with Insights"
pickle_model = None
vocab = None
df_clean = None

# Load model and dataset
def load_model():
    global pickle_model, vocab, df_clean
    
    # Load all CSV parts into one dataframe
    all_files = ["solo_development/balanced_reviews_part_1.csv", 
                 "solo_development/balanced_reviews_part_2.csv", 
                 "solo_development/balanced_reviews_part_3.csv", 
                 "solo_development/balanced_reviews_part_4.csv", 
                 "solo_development/balanced_reviews_part_5.csv", 
                 "solo_development/balanced_reviews_part_6.csv", 
                 "solo_development/balanced_reviews_part_7.csv", 
                 "solo_development/balanced_reviews_part_8.csv"]
    df_list = [pd.read_csv(file) for file in all_files]
    df_clean = pd.concat(df_list, ignore_index=True)

    # Load the model and vocab
    with open("pickle files/pickle_model.pkl", 'rb') as file:
        pickle_model = pickle.load(file)
    with open("pickle files/feature.pkl", 'rb') as voc:
        vocab = pickle.load(voc)

# Check sentiment of a given review
def check_review(review_text):
    transformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error="replace", vocabulary=vocab)
    review_text = transformer.fit_transform(loaded_vec.fit_transform([review_text]))
    return pickle_model.predict(review_text)

# Streamlit main function
def main():

    global df_clean
    # Set page configuration
    st.set_page_config(page_title=project_name, layout="centered")
    st.title(project_name)
    
    # Load model and data
    load_model()
    
    # Clean data
    df_clean = df_clean.dropna()
    df_clean = df_clean[df_clean['overall'] != 3]
    df_clean['Positivity'] = np.where(df_clean['overall'] > 3, 1, 0)

    # Pie chart
    st.subheader("Overall Sentiment Distribution")
    labels = ['Positive Reviews', 'Negative Reviews']
    values = [len(df_clean[df_clean.Positivity == 1]), len(df_clean[df_clean.Positivity == 0])]
    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
    st.plotly_chart(fig, use_container_width=True)

    st.write("---")

    # Text area input for user review
    st.subheader("Check Sentiment of Your Own Review")
    user_input = st.text_area("Enter your review:", "My daughter loves these shoes")
    
    if st.button("Submit Review Text"):
        result = check_review(user_input)
        if result[0] == 1:
            st.success("Positive Review ✅")
        elif result[0] == 0:
            st.error("Negative Review ❌")
        else:
            st.warning("Unknown Sentiment ⚠️")

    st.write("---")
    
    # Dropdown to select existing reviews
    st.subheader("Or Select a Sample Review from Data")
    selected_review = st.selectbox("Choose a Review:", df_clean['reviewText'].dropna().tolist())
    
    if st.button("Submit Selected Review"):
        result = check_review(selected_review)
        if result[0] == 1:
            st.success("Positive Review ✅")
        elif result[0] == 0:
            st.error("Negative Review ❌")
        else:
            st.warning("Unknown Sentiment ⚠️")

# Call the main function to run the app
if __name__ == "__main__":
    main()
