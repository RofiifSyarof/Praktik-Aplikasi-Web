# import library
import streamlit as st
import pickle
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# load model
feature_bow = pickle.load(open('model/feature-bow.p', 'rb'))
model_nb = pickle.load(open('model/model-nb.p', 'rb'))
model_nn = pickle.load(open('model/model-nn.p', 'rb'))

# load data
data = pd.read_csv('data/data_tweet_prabowo_clean.csv')
documents = data['Tweet'].tolist()

# vectorize
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

# clustering
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

# PCA
pca = PCA(n_components=2, random_state=0)
reduced_feature = pca.fit_transform(X.toarray())
reduced_cluster_centers = pca.transform(kmeans.cluster_centers_)

# cleansing data
def cleansing(data):
    data = re.sub(r'http\S+', '', data)
    data = re.sub(r'@\S+', '', data)
    data = re.sub(r'#', '', data)
    data = re.sub(r'[^A-Za-z0-9]+', ' ', data)
    return data

# predict sentiment
def predict_sentiment(data):
    data = cleansing(data)
    data_feature = feature_bow.transform([data])
    sentiment_nb = model_nb.predict(data_feature)[0]
    sentiment_nn = model_nn.predict(data_feature)[0]
    return sentiment_nb, sentiment_nn
    
# Tampilan streamlit
st.title('Sentiment Analysis')
tweet = st.text_area('Input Tweet')

if st.button('Predict'):
    sentiment_nb, sentiment_nn = predict_sentiment(tweet)
    st.write('Naive Bayes:', sentiment_nb)
    st.write('Neural Network:', sentiment_nn)

    # plot
    plt.figure(figsize=(10, 10))
    plt.scatter(reduced_feature[:, 0], reduced_feature[:, 1], c=kmeans.labels_, s=50, cmap='viridis')
    plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:, 1], c='red', s=300, alpha=0.5)
    plt.title('KMeans Clustering')
    st.pyplot(plt)