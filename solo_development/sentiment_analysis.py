# Importing Libraries

import pickle
import pandas as pd
import numpy as np
import webbrowser
import dash
from dash import Dash, callback, html, dcc, dash_table, Input, Output, State, MATCH, ALL

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import plotly.graph_objects as go

# DECLARING GLOBAL VARIABLES 

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
project_name = "Sentiment Analysis with Insights"

# DECLARING MY FUNCTIONS 

def open_browser():
    return webbrowser.open("http://127.0.0.1:8050/")
import streamlit as st
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

# DECLARING GLOBAL VARIABLES
project_name = "Sentiment Analysis with Insights"

# DECLARING MY FUNCTIONS

def load_model():
    global pickle_model
    global vocab
    global df, dfs
    
    df = pd.read_csv("balanced_reviews.csv")    
    dfs = pd.read_csv("scrappedReviews.csv")   
    
    
    with open("pickle_model.pkl", 'rb') as file:
        pickle_model = pickle.load(file)
    with open("feature.pkl", 'rb') as voc:
        vocab = pickle.load(voc)
        
def check_review(reviewText):
    transformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error="replace",vocabulary=vocab)
    reviewText = transformer.fit_transform(loaded_vec.fit_transform([reviewText]))
    return pickle_model.predict(reviewText)

def create_app_ui():
    global project_name
    global df
    df = df.dropna()
    df = df[df['overall'] != 3]
    df['Positivity'] = np.where(df['overall'] > 3, 1, 0)
    labels = ['Positive Reviews', 'Negative Reviews']
    values = [len(df[df.Positivity == 1]), len(df[df.Positivity == 0])]
    main_layout = dbc.Container(
        dbc.Jumbotron(
                [
                    html.H1(id = 'heading', children = project_name, className = 'display-3 mb-4'),
                    dbc.Container(
                        dcc.Loading(
                        dcc.Graph(
                            figure = {'data' : [go.Pie(labels=labels, values=values)],
                                      'layout': go.Layout(height = 600, width = 1000, autosize = False)
                                      }
                            )
                        ),
                        className = 'd-flex justify-content-center'
                    ),
                    
                    html.Hr(),
                    dbc.Textarea(id = 'textarea', className="mb-3", placeholder="Enter the Review", value = 'My daughter loves these shoes', style = {'height': '150px'}),
                    html.Hr(),
                    dbc.Container([
                        dcc.Dropdown(
                    id='dropdown',
                    placeholder = 'Select a Review',
                    options=[{'label': i[:50] + "...", 'value': i} for i in dfs.reviews],
                    value = df.reviewText[0],
                    style = {'margin-bottom': '30px'}
                    
                )
                       ],
                        style = {'padding-left': '50px', 'padding-right': '50px'}
                        ),
                    dbc.Button("Submit", color="dark", className="mt-2 mb-3", id = 'button', style = {'width': '100px'}),
                    html.Div(id = 'result'),
                    html.Div(id = 'result1')
                    ],
                className = 'text-center'
                ),
        className = 'mt-4'
        )
    return main_layout

@app.callback(
    Output('result', 'children'),
    [
    Input('button', 'n_clicks')
    ],
    [
    State('textarea', 'value')
    ]
    )    
def update_app_ui(n_clicks, textarea):
    result_list = check_review(textarea)
    
    if (result_list[0] == 0 ):
        return dbc.Alert("Negative", color="danger")
    elif (result_list[0] == 1 ):
        return dbc.Alert("Positive", color="success")
    else:
        return dbc.Alert("Unknown", color="dark")

@app.callback(
    Output('result1', 'children'),
    [
    Input('button', 'n_clicks')
    ],
    [
     State('dropdown', 'value')
     ]
    )
def update_dropdown(n_clicks, value):
    result_list = check_review(value)
    
    if (result_list[0] == 0 ):
        return dbc.Alert("Negative", color="danger")
    elif (result_list[0] == 1 ):
        return dbc.Alert("Positive", color="success")
    else:
        return dbc.Alert("Unknown", color="dark")

# DECLARING MAIN FUNCTION 
        
def main():
    global app
    global project_name
    load_model()
    open_browser()
    app.layout = create_app_ui()
    app.title = project_name
    app.run_server()
    app = None
    project_name = None

# CALLING MAIN FUNCTION 
    
if __name__ == '__main__':
    main()

    global df

    df = pd.read_csv("balanced_reviews.csv")
    
    with open("../pickle_model.pkl", 'rb') as file:
        pickle_model = pickle.load(file)
    with open("../feature.pkl", 'rb') as voc:
        vocab = pickle.load(voc)

def check_review(review_text):
    transformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error="replace", vocabulary=vocab)
    review_text = transformer.fit_transform(loaded_vec.fit_transform([review_text]))
    return pickle_model.predict(review_text)

def main():
    st.set_page_config(page_title=project_name, layout="centered")
    st.title(project_name)
    
    load_model()
    
    # Cleaning Data
    df_clean = df.dropna()
    df_clean = df_clean[df_clean['overall'] != 3]
    df_clean['Positivity'] = np.where(df_clean['overall'] > 3, 1, 0)
    
    # Pie Chart
    st.subheader("Overall Sentiment Distribution")
    labels = ['Positive Reviews', 'Negative Reviews']
    values = [len(df_clean[df_clean.Positivity == 1]), len(df_clean[df_clean.Positivity == 0])]
    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
    st.plotly_chart(fig, use_container_width=True)
    
    st.write("---")
    
    # Textarea input
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
    
if __name__ == "__main__":
    main()

