import HuggingFaceTransformer
import naiveBayes
import SVMTFIDF
import VADER

def get_sentiment(review: str):
    scores = []
    scores.append(HuggingFaceTransformer.get_sentiment_hugging_face(review))
    scores.append(naiveBayes.get_sentiment_naive_bayes(review))
    scores.append(SVMTFIDF.get_sentiment_svm(review))
    scores.append(VADER.get_sentiment_VADER(review))
    if sum(scores) >= 3:
        print("Your Review was positive!")
    elif sum(scores) == 2:
        del scores[1]
        if sum(scores) >=1:
            print("Your Review was positive!")
        else:
            print("Your Review was negative!")
    else:
        print("Your Review was negative!")



if __name__ == '__main__':
    review = input("Please input a review:\n")
    get_sentiment(review)
