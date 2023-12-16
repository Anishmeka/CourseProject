import pandas as pd
from DataProcessing import df
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm

def get_sentiment_svm(review: str):
    X = df['review_content'].tolist()
    X.append(review)
    y = pd.to_numeric(df['rating'], 'coerce').fillna(0).astype(float) >= 4
    # Create feature vectors
    vectorizer = TfidfVectorizer(min_df = 5,
                                max_df = 0.8,
                                sublinear_tf = True,
                                use_idf = True)
    vectors = vectorizer.fit_transform(X)
    tokenized_review = vectors[-1]
    vectors = vectors[:-1]
    model = svm.SVC(kernel='linear')
    model.fit(vectors, y)
    prediction = model.predict(tokenized_review)
    if prediction[0] == True:
        return 1
    else:
        return 0