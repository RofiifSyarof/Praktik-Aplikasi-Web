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

# predict sentiment
def predict_sentiment(data):
    data_feature = feature_bow.transform([data])
    sentiment_nb = model_nb.predict(data_feature)[0]
    sentiment_nn = model_nn.predict(data_feature)[0]
    return sentiment_nb, sentiment_nn
    
# Tampilan streamlit

# Membuat sidebar
st.sidebar.title('Web Analisis Sentiment')
menu = st.sidebar.selectbox('Pilihan Menu', ['Home', 'Sentiment per Kalimat', 'Sentiment per kata'])

if menu == 'Home':
    st.title('Analisis Sentiment Komentar dari Sosial Media X')
    st.write('Selamat datang di aplikasi Sentiment Analysis')
    st.write('Aplikasi ini digunakan untuk menganalisis sentiment dari tweet')
    st.write('Silahkan pilih menu di sidebar untuk melihat hasil analisis')
    
elif menu == 'Sentiment per Kalimat':
    st.title('Sentiment per Kalimat')
    st.write('Halaman ini digunakan untuk menganalisis sentiment dari setiap komentar yang didapat dari tweet')
    
    # menampilkan tabel berisi tweet dan nilai sentiment
    if st.checkbox('Tampilkan Data'):
        data['Sentiment_NB'] = data['Tweet'].apply(lambda x: predict_sentiment(x)[0])
        data['Sentiment_NN'] = data['Tweet'].apply(lambda x: predict_sentiment(x)[1])
        st.write(data[['Tweet', 'Sentiment_NB', 'Sentiment_NN']])
        
    # menampilkan jumlah tweet positif, negatif dan netral dalam bentuk diagram batang
    if st.checkbox('Tampilkan Jumlah Sentiment berdasarkan Metode Naive Bayes'):
        data['Sentiment_NB'] = data['Tweet'].apply(lambda x: predict_sentiment(x)[0])
        sentiment_count = data['Sentiment_NB'].value_counts()
        st.bar_chart(sentiment_count)
    
    if st.checkbox('Tampilkan Jumlah Sentiment berdasarkan Metode Neural Network'):
        data['Sentiment_NN'] = data['Tweet'].apply(lambda x: predict_sentiment(x)[1])
        sentiment_count = data['Sentiment_NN'].value_counts()
        st.bar_chart(sentiment_count)
        
elif menu == 'Sentiment per kata':
    st.title('Sentiment per Kata')
    st.write('Halaman ini digunakan untuk menganalisis sentiment dari setiap kata yang ada pada tweet')
        
    # menampilkan nilai sentiment setiap kata
    if st.checkbox('Tampilkan Nilai Sentiment Setiap Kata'):
        sentiment_data = []
        for word, idx in feature_bow.vocabulary_.items():
            sentiment_nb, sentiment_nn = predict_sentiment(word)
            sentiment_data.append({'Word': word, 'Sentiment_NB': sentiment_nb, 'Sentiment_NN': sentiment_nn})
        
        sentiment_df = pd.DataFrame(sentiment_data)
        st.write(sentiment_df)
    
    # menampilkan word cloud
    if st.checkbox('Tampilkan Word Cloud'):
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        
        wordcloud = WordCloud().generate(' '.join(data['Tweet']))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
        
    # menampilkan hasil clustering
    if st.checkbox('Tampilkan Hasil Clustering'):
        plt.figure(figsize=(10, 10))
        plt.scatter(reduced_feature[:, 0], reduced_feature[:, 1], c=kmeans.labels_, s=50, cmap='viridis')
        plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:, 1], c='red', s=300, alpha=0.5)
        plt.title('KMeans Clustering')
        st.pyplot(plt)