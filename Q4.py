from nltk.corpus import webtext

# use to find bigrams, which are pairs of words
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

# Loading the data
words = [w.lower() for w in webtext.words('/Users/itn.rohith.kakarla/Downloads/B180039CS_NLP_LabAssignment/file3.txt')]

biagram_collocation = BigramCollocationFinder.from_words(words)
biagram_collocation.nbest(BigramAssocMeasures.likelihood_ratio, 15)

from nltk.corpus import stopwords

stopset = set(stopwords.words('english'))
filter_stops = lambda w: len(w) < 3 or w in stopset

biagram_collocation.apply_word_filter(filter_stops)
print(biagram_collocation.nbest(BigramAssocMeasures.likelihood_ratio, 15))
