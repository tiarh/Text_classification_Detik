from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st

def train_naive_bayes(X_train, y_train):
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    return nb_model

def train_knn(X_train, y_train, k_value):
    knn_model = KNeighborsClassifier(n_neighbors=k_value)
    knn_model.fit(X_train, y_train)
    return knn_model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    st.subheader("Evaluasi Model")
    st.write("Akurasi:", accuracy_score(y_test, y_pred))
    st.write("Classification Report:\n", classification_report(y_test, y_pred))

def classify_text(model, text_vectorized):
    prediction = model.predict(text_vectorized)[0]
    return prediction
