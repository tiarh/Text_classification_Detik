from numpy import vectorize
from sklearn.model_selection import train_test_split
import streamlit as st
from data_preparation import prepare_data, preprocess_data, extract_tfidf_features, preprocess_text
from text_classification import train_naive_bayes, train_knn, evaluate_model, classify_text
from sklearn.feature_extraction.text import TfidfVectorizer


st.title("Aplikasi Klasifikasi Berita")

# Menu Persiapan Data
st.header("1. Persiapan Data")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    df = prepare_data(uploaded_file)

    st.subheader("Data Sebelum Pra-Pemrosesan")
    st.write(df.head())

    # Pra-pemrosesan data
    df_processed = preprocess_data(df)
    st.subheader("Data Setelah Pra-Pemrosesan")
    st.write(df_processed.head())

    # Ekstraksi Fitur TF-IDF
    df_tfidf = extract_tfidf_features(df_processed)
    st.subheader("Hasil Ekstraksi Fitur TF-IDF")
    st.write(df_tfidf)

    # Bagi data menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(df_tfidf, df_processed['kategori'], test_size=0.2, random_state=42)

# Menu Model dan Evaluasi
st.header("2. Model dan Evaluasi")
model_option = st.selectbox("Pilih Model", ["", "Naive Bayes", "KNN"])

if model_option == "Naive Bayes":
    nb_model = train_naive_bayes(X_train, y_train)
    evaluate_model(nb_model, X_test, y_test)

elif model_option == "KNN":
    k_value = st.slider("Pilih Nilai K untuk KNN", min_value=1, max_value=10, value=3)
    knn_model = train_knn(X_train, y_train, k_value)
    evaluate_model(knn_model, X_test, y_test)

# Menu Test Teks
st.header("3. Klasifikasi Berita")
text_input = st.text_area("Masukkan Teks untuk Diklasifikasi", "")
preprocessed_text = preprocess_text(text_input)

if st.button("Klasifikasi"):
    # Assuming vectorizer is the fitted TF-IDF vectorizer used during model training
    vectorizer = TfidfVectorizer()
    vectorizer.fit(X_train)

    # Transform the preprocessed text using the fitted vectorizer
    text_vectorized = vectorizer.transform([preprocessed_text])

    # Prediksi menggunakan model terpilih
    if model_option == "Naive Bayes":
        prediction = classify_text(nb_model, text_vectorized)
    elif model_option == "KNN":
        prediction = classify_text(knn_model, text_vectorized)

    st.subheader("Hasil Klasifikasi:")
    st.write(f"Teks '{text_input}' diklasifikasikan sebagai: {prediction}")
