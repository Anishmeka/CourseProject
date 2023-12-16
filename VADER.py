import nltk
from DataProcessing import df
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

#download the VADER sentiment analyzer
nltk.download('vader_lexicon')

'''Takes in a review and uses VADER to determine 
whether a review is positive, neutral or negative.
'''
def get_sentiment_VADER(review: str):
    analyzer_obj = SentimentIntensityAnalyzer()
    result_dict = analyzer_obj.polarity_scores(review)
    if result_dict['compound'] >=0.05:
        return 1
    elif result_dict['compound'] <= 0.05:
        return -1
    else:
        return 0
    
    