import HuggingFaceTransformer
import VADER


'''Takes in a review and determines
whether the review is positive, neutral or negative.
'''
def get_sentiment(review: str):
    scores = []
    scores.append(HuggingFaceTransformer.get_sentiment_hugging_face(review))
    scores.append(VADER.get_sentiment_VADER(review))
    if sum(scores) > 0:
        print("Your Review was positive!")
    elif sum(scores) == 0:
        print("Your Review was neutral!")
    else:
        print("Your Review was negative!")



if __name__ == '__main__':
    review = input("Please input a review:\n")
    get_sentiment(review)
