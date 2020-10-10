import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Random seed for consistency
np.random.seed(500)

# Uncomment line below to download nltk requirements
# nltk.download('punkt')
factory = StemmerFactory()
stemmer = factory.create_stemmer()
Encoder = LabelEncoder()
Tfidf_vect = TfidfVectorizer()

# Configuration
DATA_LATIH = "./Data Latih/Data Latih BDC.csv"
DATA_UJI = "./Data Uji/Data Uji BDC.csv"


def train_model():
    """Training step for Support Vector Classifier
    included data preprocessing using Sklearn module """

    # Open Datasets
    datasets = pd.read_csv(DATA_LATIH)
    print(datasets["label"].value_counts())

    # Text Normalization using
    # PySastrawi(Word Stemmer for Bahasa Indonesia)
    lower = [stemmer.stem(row.lower()) for row in datasets["narasi"]]
    vectors = [word_tokenize(element) for element in lower]
    labels = datasets["label"]

    # Splitting Datasets for feeding to Machine Learning
    Train_X, Test_X, Train_Y, Test_Y = train_test_split(
        vectors, labels, test_size=0.3, stratify=labels)

    # Encoder for Data Label
    Train_Y = Encoder.fit_transform(Train_Y)
    Test_Y = Encoder.fit_transform(Test_Y)

    # Create Tfidf Vector
    Tfidf_vect.fit(["".join(row) for row in lower])

    # Applying Tfidf for Training and Testing Features
    Train_X_Tfidf = Tfidf_vect.transform([" ".join(row) for row in Train_X])
    Test_X_Tfidf = Tfidf_vect.transform([" ".join(row) for row in Test_X])

    # Classifier - Algorithm - SVM
    # fit the training dataset on the classifier
    SVM = svm.SVC(C=1.0, kernel='linear', degree=1, gamma="auto", verbose=True)
    SVM.fit(Train_X_Tfidf, Train_Y)  # predict the labels on validation dataset
    # Use accuracy_score function to get the accuracy
    predictions_SVM = SVM.predict(Test_X_Tfidf)
    print("SVM Accuracy Score -> ", accuracy_score(predictions_SVM, Test_Y)*100)
    return SVM


def test_model(SVM):
    """Testing Machine Learning model using test 
    datasets using the same method as training"""

    datasets = pd.read_csv(DATA_UJI)

    lower = [stemmer.stem(row.lower()) for row in datasets["narasi"]]
    vectors = [word_tokenize(element) for element in lower]

    Test_X_Tfidf = Tfidf_vect.transform([" ".join(row) for row in vectors])

    predictions_SVM = SVM.predict(Test_X_Tfidf)

    data = {"ID": list(datasets["ID"]), "prediksi": predictions_SVM}
    hasil = pd.DataFrame(data, columns=["ID", "prediksi"])
    hasil.to_csv("./Hasil Uji Model.csv", index=False)


# Train Machine Learning model
SVM = train_model()

# Only uncomment line below if you want to generate a file on test
# test_model(SVM)
