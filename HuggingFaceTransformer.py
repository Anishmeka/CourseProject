from transformers import pipeline


def get_sentiment_hugging_face(review: str):
    sent_pipeline = pipeline("sentiment-analysis")
    result = sent_pipeline(review)[0]
    if result['label'] == 'POSITIVE':
        return 1
    else:
        return 0
