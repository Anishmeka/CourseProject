import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from DataProcessing import df

def get_sentiment_naive_bayes(review: str):
    X = df['review_content'].tolist()
    X.append(review)
    y = pd.to_numeric(df['rating'], 'coerce').fillna(0).astype(float) >= 4
    vec = CountVectorizer(stop_words='english')
    X = vec.fit_transform(X).toarray()
    tokenized_review = X[-1]
    X = X[:-1]

    model = MultinomialNB()
    model.fit(X, y)
    prediction = model.predict(tokenized_review.reshape(1, -1))
    if prediction[0] == True:
        return 1
    else:
        return 0