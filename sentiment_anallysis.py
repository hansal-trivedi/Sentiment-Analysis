import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC


from nltk.classify import ClassifierI
from statistics import mode

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes=[]
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)


    def confidence(self,features):
        votes=[]
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

    
documents = [(list(movie_reviews.words(file)),category)
             for category in movie_reviews.categories()
             for file in movie_reviews.fileids(category)]

random.shuffle(documents)
##print(movie_reviews.fileids("pos"))
##  print(movie_reviews.words())

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
##print(all_words.most_common(15))


word_features = list(all_words.keys())[:3000]

def find_features(document):
    words = set(document)
    features={}
    for w in word_features:
        features[w] = (w in words)
    return features

##print(find_features(movie_reviews.words("neg/cv000_29416.txt")))

featureset = [(find_features(rev),category) for (rev,category) in documents]

training_set = featureset[:1900]
test_set = featureset[1900:]
## Naive Bayes classifier

classifier = nltk.NaiveBayesClassifier.train(training_set)
print("accuracy: ",(nltk.classify.accuracy(classifier,test_set)))


## USE OF PICKLE

##save_classifier = open("naiveBayes.pickle","wb")
##pickle.dump(classifier,save_classifier)
##save_classifier.close()
##
##
##classifier_f = open("naiveBayes.pickle","rb")
##classifier = pickle.load(classifier_f)
##classifier_f.close()


MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("accuracy for multinomial NB: ",(nltk.classify.accuracy(MNB_classifier,test_set)))

##GaussianNB_classifier = SklearnClassifier(GaussianNB())
##GaussianNB_classifier.train(training_set)
##print("accuracy for Gaussian NB: ",(nltk.classify.accuracy(GaussianNB_classifier,test_set)))

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("accuracy for Bernoulli NB: ",(nltk.classify.accuracy(BernoulliNB_classifier,test_set)))

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("accuracy for LogisticRegression: ",(nltk.classify.accuracy(LogisticRegression_classifier,test_set)))

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("accuracy for SGDClassifier: ",(nltk.classify.accuracy(SGDClassifier_classifier,test_set)))

##SVC_classifier = SklearnClassifier(SVC())
##SVC_classifier.train(training_set)
##print("accuracy for SVC: ",(nltk.classify.accuracy(SVC_classifier,test_set)))

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("accuracy for LinearSVC: ",(nltk.classify.accuracy(LinearSVC_classifier,test_set)))

NuSVC_classifier = SklearnClassifier(NuSVC(nu=0.5))
NuSVC_classifier.train(training_set)
print("accuracy for NuSVC: ",(nltk.classify.accuracy(NuSVC_classifier,test_set)))





voted_classifier = VoteClassifier(classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier,
                                  SGDClassifier_classifier,
                                  LinearSVC_classifier,
                                  NuSVC_classifier)

print("accuracy for voted_classifier: ",(nltk.classify.accuracy(voted_classifier,test_set)))
print("Classification:",voted_classifier.classify(test_set[0][0]), "confidence %:",voted_classifier.confidence(test_set[0][0]))
