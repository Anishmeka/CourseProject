from transformers import pipeline

'''Takes in a review and uses a hugging face transformer 
model to determines whether a review 
is positive, neutral or negative.
'''
def get_sentiment_hugging_face(review: str):
    sent_pipeline = pipeline("sentiment-analysis")
    result = sent_pipeline(review)[0]
    if result['label'] == 'POSITIVE':
        return 1
    elif result['label'] == 'NEUTRAL':
        return 0
    else:
        return -1
