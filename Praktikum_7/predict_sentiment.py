# import library
import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from streamlit_lottie import st_lottie


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

# membuat function untuk load file lottie
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# memasukkan file lottie 
lottie_alert = load_lottiefile("lottiefolder/alert.json")
lottie_comment_reading = load_lottiefile("lottiefolder/comment_reading.json")
lottie_question = load_lottiefile("lottiefolder/question.json")
lottie_caution = load_lottiefile("lottiefolder/caution.json")
lottie_arrow_left = load_lottiefile("lottiefolder/arrow_left.json")
lottie_analyzing = load_lottiefile("lottiefolder/analyzing.json")
lottie_bar_chart = load_lottiefile("lottiefolder/bar_chart.json")
    
# Tampilan streamlit

# Membuat sidebar
st.sidebar.title('📊Web Analisis Sentiment')
menu = st.sidebar.selectbox('⬇️Pilihan Menu⬇️', ['🏠Home', '📃Sentiment per Kalimat', '📄Sentiment per kata'])

if menu == '🏠Home':
    # membuat 2 kolom untuk judul dan gambar
    col1, col2 = st.columns([3, 1.5], vertical_alignment='center')
    with col1:
        st.title('Analisis Sentiment Komentar dari Sosial Media📊')
    with col2:
        st_lottie(
            lottie_comment_reading,
            speed=1,
            width=200,
            height=200,
        )
        
    st.markdown('---')
    
    col1, col2 = st.columns([1, 2], vertical_alignment='center')
    with col1:
        st_lottie(
            lottie_question,
            speed=1,
            width=200,
            height=200,
        )
    with col2:
        st.header('What\'s this website about?')
        st.write('Website ini bertujuan untuk menilai sentiment dari komentar-komentar dari sosial media yang sudah dikumpulkan menjadi sebuah file csv.'
                 ' Komentar-komentar tersebut akan dianalisis menggunakan metode Naive Bayes dan Neural Network.'
                 ' Dengan menggunakan metode tersebut, kita dapat mengetahui apakah komentar tersebut bersifat positif, negatif atau netral.')
    
    st.write('\n\n')
    
    col1, col2, col3 = st.columns([1, 3, 1], vertical_alignment='center')
    with col1:
        st_lottie(
            lottie_caution,
            speed=1,
            width=100,
            height=100,
        )
    with col2:
        st.markdown(
        """
        <div style="text-align: center;">
            <h2><b>CATATAN PENTING!</h2>
            <p>Saat ini, website ini hanya dapat membaca analisis sentiment secara statis.
            Artinya, website ini hanya dapat membaca data yang sudah ada dan tidak dapat membaca data secara yang ingin dianalisis oleh pengguna.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    with col3:
        st_lottie(
            lottie_caution,
            speed=1,
            width=100,
            height=100,
        )
    
    col1, col2 = st.columns([1, 2], vertical_alignment='center')
    with col1:
        st_lottie(
            lottie_arrow_left,
            speed=1,
            width=180,
            height=180,
        )
    with col2:
        st.write('<h3>🚀Untuk mencoba fitur pada website ini, silakan pilih menu yang ada di sebelah kiri🚀', unsafe_allow_html=True)
    
    st.header('💡Our Future Plan💡')
    st.write('Rencana kedepannya, kami akan menambahkan fitur-fitur seperti pada list berikut:')
    st.markdown(
    """
    - Menambahkan fitur untuk mengupload file csv yang ingin dianalisis.
    - Menambahkan fitur untuk menginput data secara langsung.
    - Menambahkan dukungan untuk komentar-komentar dari sosial media lainnya.
    
    """)
    
elif menu == '📃Sentiment per Kalimat':
    col1, col2 = st.columns([3, 1.5], vertical_alignment='center')
    with col1:
        st.title('Analisis Sentiment per Kalimat📃')
    with col2:
        st_lottie(
            lottie_analyzing,
            speed=1,
            width=250,
            height=250,
        )
        
    st.markdown('---')
    
    st.header('❓What\'s this page about❓')
    st.write('Halaman ini digunakan untuk menganalisis nilai sentiment dari kalimat-kalimat komentar yang sudah dikumpulkan.'
             ' Komentar-komentar tersebut akan dianalisis menggunakan metode Naive Bayes dan Neural Network.')
    st.write('⬇️Untuk melihat hasil analisis, silakan pilih menu yang ada di bawah ini⬇️')
    
    # menampilkan tabel berisi tweet dan nilai sentiment
    if st.checkbox('Tampilkan Data'):
        st.write('Fitur ini akan menampilkan seluruh kalimat komentar yang berasal dari twitter beserta nilai sentimentnya.')
        st.markdown('<h4><b>⚠️Catatan!⚠️:</b></h4>', unsafe_allow_html=True)
        st.markdown(
        """
        - Sentiment_NB: Nilai sentiment yang didapat menggunakan metode Naive Bayes.
        - Sentiment_NN: Nilai sentiment yang didapat menggunakan metode Neural Network.
        """,
        unsafe_allow_html=True
        )
        data['Sentiment_NB'] = data['Tweet'].apply(lambda x: predict_sentiment(x)[0])
        data['Sentiment_NN'] = data['Tweet'].apply(lambda x: predict_sentiment(x)[1])
        st.write(data[['Tweet', 'Sentiment_NB', 'Sentiment_NN']])
        
    # menampilkan jumlah tweet positif, negatif dan netral dalam bentuk diagram batang
    if st.checkbox('Tampilkan Jumlah Sentiment berdasarkan Metode Naive Bayes'):
        st.write('Fitur ini akan menampilkan jumlah tweet yang memiliki sentiment positif, negatif dan netral berdasarkan metode Naive Bayes menggunakan diagaram batang.')
        data['Sentiment_NB'] = data['Tweet'].apply(lambda x: predict_sentiment(x)[0])
        sentiment_count = data['Sentiment_NB'].value_counts()
        st.bar_chart(sentiment_count)
        st.write('Dan berikut adalah jumlah tweet yang memiliki sentiment positif, negatif dan netral berdasarkan metode Naive Bayes:')
        st.write(sentiment_count)
    
    if st.checkbox('Tampilkan Jumlah Sentiment berdasarkan Metode Neural Network'):
        st.write('Fitur ini akan menampilkan jumlah tweet yang memiliki sentiment positif, negatif dan netral berdasarkan metode Neural Network menggunakan diagaram batang.')
        data['Sentiment_NN'] = data['Tweet'].apply(lambda x: predict_sentiment(x)[1])
        sentiment_count = data['Sentiment_NN'].value_counts()
        st.bar_chart(sentiment_count)
        st.write('Dan berikut adalah jumlah tweet yang memiliki sentiment positif, negatif dan netral berdasarkan metode Neural Network:')
        st.write(sentiment_count)
        
elif menu == '📄Sentiment per kata':
    col1, col2 = st.columns([3, 1.5], vertical_alignment='center')
    with col1:
        st.title('Analisis Sentiment per Kata📄')
    with col2:
        st_lottie(
            lottie_bar_chart,
            speed=1,
            width=250,
            height=250,
        )

    st.markdown('---')

    st.header('❓What\'s this page about❓')
    st.write('Halaman ini akan menampilkan nilai sentiment dari setiap kata dari kalimat-kalimat komentar yang terdapat di dataset.'
             ' Nilai sentiment tersebut didapat menggunakan metode Naive Bayes dan Neural Network.'
             ' Kalimat-kalimat komentar akan diubah  menjadi bentuk vector sehingga dapat diidentifikasi menjadi kata-kata yang memiliki sentiment positif, negatif atau netral.')
    st.write('⬇️Untuk melihat hasil analisis, silakan pilih menu yang ada di bawah ini⬇️')
        
    # menampilkan nilai sentiment setiap kata
    if st.checkbox('Tampilkan Nilai Sentiment Setiap Kata'):
        st.write('Fitur ini akan menampilkan nilai sentiment dari setiap kata yang terdapat di dataset.')
        st.markdown('<h4><b>⚠️Catatan!⚠️:</b></h4>', unsafe_allow_html=True)
        st.markdown(
        """
        - Sentiment_NB: Nilai sentiment yang didapat menggunakan metode Naive Bayes.
        - Sentiment_NN: Nilai sentiment yang didapat menggunakan metode Neural Network.
        """,
        unsafe_allow_html=True
        )
        sentiment_data = []
        for word, idx in feature_bow.vocabulary_.items():
            sentiment_nb, sentiment_nn = predict_sentiment(word)
            sentiment_data.append({'Word': word, 'Sentiment_NB': sentiment_nb, 'Sentiment_NN': sentiment_nn})
        
        sentiment_df = pd.DataFrame(sentiment_data)
        st.write(sentiment_df)
        
        # menampilkan banyak kata yang memiliki sentiment positif, negatif dan netral
        st.write('Dan berikut adalah jumlah kata yang memiliki sentiment positif, negatif dan netral:')
        sentiment_count = sentiment_df[['Sentiment_NB', 'Sentiment_NN']].apply(pd.Series.value_counts)
        st.write(sentiment_count)
    
    # menampilkan word cloud
    if st.checkbox('Tampilkan Word Cloud'):
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        
        st.write('☁️Fitur ini akan menampilkan word cloud dari seluruh kata yang terdapat di dataset☁️')
        st.write('Kata yang sering muncul akan ditampilkan lebih besar dibandingkan dengan kata yang jarang muncul.')
        wordcloud = WordCloud().generate(' '.join(data['Tweet']))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
        
    # menampilkan most frequent word dalam bentuk diagram batang
    if st.checkbox('Tampilkan Kata yang Paling Sering Muncul'):
        from sklearn.feature_extraction.text import CountVectorizer
        
        st.write('Fitur ini akan menampilkan kata yang paling sering muncul dalam dataset menggunakan diagram batang📊')
        count_vectorizer = CountVectorizer(stop_words='english')
        count_vectorizer.fit(data['Tweet'])
        count = count_vectorizer.transform(data['Tweet'])
        count = count.toarray().sum(axis=0)
        word_freq = pd.DataFrame({
            'Word': count_vectorizer.get_feature_names_out(),
            'Count': count
        })
        word_freq = word_freq.sort_values('Count', ascending=False).head(10)
        
        # menampilkan diagram batang
        st.bar_chart(word_freq.set_index('Word'))
        
        st.write('Dan berikut adalah kata yang paling sering muncul:')
        st.write(word_freq)
        
    # menampilkan hasil clustering
    if st.checkbox('Tampilkan Hasil Clustering'):
        st.write('Fitur ini akan menampilkan hasil clustering dari dataset menggunakan metode KMeans.'
                 ' Hasil clustering ini akan menampilkan data-data yang memiliki karakteristik yang sama dalam satu cluster.'
                 ' Clustering pada data ini akan dibedakan menjadi 2 cluster.')
        plt.figure(figsize=(10, 10))
        plt.scatter(reduced_feature[:, 0], reduced_feature[:, 1], c=kmeans.labels_, s=50, cmap='viridis')
        plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:, 1], c='red', s=300, alpha=0.5)
        plt.title('KMeans Clustering')
        st.pyplot(plt)