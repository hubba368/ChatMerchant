import nltk
import os
import pickle
import numpy as np
import sys
import warnings

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('universal_tagset', quiet=True)
#nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from ChatbotUtils import GetVectorSimilarity

np.set_printoptions(threshold=sys.maxsize)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

class Classifier:
    def __init__(self):
        self.trainLabels = None
        self.testLabels = None
        self.trainData = None
        self.testData = None

        self.classifier = None
        self.vectorizer = None
        self.transformer = None

        self.labels = []
        self.data = []
        self.tagset = []
        self.responses = {}
        self.docsTfIdf = None
        self.encoder = None
        self.encLabels = None

    # load corpora thru NLTK
    def LoadData(self, nltkPreProc, path, removeStopWords):
        for file in os.listdir(path):
            p = os.path.join(path, file)
            content, label = nltkPreProc.GetCorporaFromFile(p)
            tokens = nltkPreProc.Tokenize(content, removeStopWords, True)
            sents = nltkPreProc.Tokenize(content, removeStopWords, False)
            self.tagset += nltkPreProc.GetPOSTags(sents)
            self.data += nltkPreProc.LemmatizeTokens(tokens, True)
            # setup data labels
            # bigram
            #for word in tokens:
            #    self.labels += ([label for t in word])
            self.labels += ([label for word in tokens])

    def LoadResponses(self, path):
        labels = set(self.labels)
        for lab in labels:
            self.responses[lab] = None

        for file in os.listdir(path):
            p = os.path.join(path, file)
            name = os.path.basename(p)
            name = os.path.splitext(name)[0]
            with open(p, encoding='utf-8', errors='ignore', mode='r') as f:
                trustVal = f.readline()
                responses = f.read()
                self.responses[name] = (trustVal.strip(), responses.splitlines())

    # vectorize and weight a set of preprocessed tokens
    def Vectorize(self, tokens, useStopwords=True):
        if useStopwords:
            self.vectorizer = CountVectorizer(stop_words=stopwords.words('english'), ngram_range=(1, 2))
        else:
            self.vectorizer = CountVectorizer(ngram_range=(1, 2))
        countVect = self.vectorizer.fit_transform(tokens)

        self.transformer = TfidfTransformer(use_idf=True, sublinear_tf=False)
        Xtf = self.transformer.fit_transform(countVect)

        return Xtf

    # predict intent by cosine similarity and/or classification
    def Predict(self, query=None, predictByCosine=False):
        if not query:
            print("ERROR: Nothing inputted.\n")
            return None

        # assemble query vector by transforming against trained
        # also handle inputs that would be removed by stopwords
        testdoc = query
        newCount = self.vectorizer.transform(testdoc)
        newtfidf = None
        try:
            newtfidf = self.transformer.transform(newCount)
        except:
            print("ERROR: Unable to transform vector. Most likely a single character or stopword.")
            return None

        # get decoded labels
        intents = []
        for i in set(self.encoder.inverse_transform(self.encLabels)):
            intents.append(i)
            intents = sorted(intents)

        # get cosine sim if needed
        cosines = []

        if(predictByCosine==True):
            # calc cosine per word in input, then store for later
            for i in range(0, len(testdoc)):
                currWord = [testdoc[i]]
                cosCount = self.vectorizer.transform(currWord)
                cosTfIdf = None
                try:
                    cosTfIdf = self.transformer.transform(cosCount)
                except:
                    print("Unable to transform cosine vector. Most likely a single character or stopword.")

                # get cosine and flatten because sklearn returns a matrix
                csm = GetVectorSimilarity(cosTfIdf, self.docsTfIdf)
                csm = csm.flatten()

                # collate closest numerical matches (cosine sim)
                temp = np.partition(csm, -len(intents))
                closestSim = temp[-len(intents):]
                closestSim = list(closestSim)

                # collect all highest cosines to store for classification comparison
                # only correct for BOW style models because 'document' = 1 word
                if 1.0 in closestSim:
                    cosines.append(1.0)
                    continue
                elif 0.0 in closestSim and 1.0 not in closestSim:
                    cosines.append(0.0)
                    continue
                else:
                    cosines.append(max(closestSim))

        # using predict_proba because predict returns binary matches even on unseen words
        # however unseen words will still return probabilities but at some arbitrary minimum value for all labels
        # because of smoothing
        # to get around this, I will be summing the probabilities of predicted
        # and then will ignore unseen words based on their cosine sim
        # as without an "unseen word" check, with a small enough corpus
        # you will get large predicted vals for unseen words, which may ruin classification
        # NOTE: could probs do this with just predict, as cosine can still be used to check for unseens
        predictedIntents = []
        predicted = self.classifier.predict_proba(newtfidf)

        for d, c in zip(testdoc, predicted):
            predictedIntents.append([d, c])

        # get closest matches by classification
        # OR classify AND match against cosines
        useCosine = False
        if len(cosines) > 0:
            useCosine = True

        if len(testdoc) > 1:
            curr = predictedIntents[0][1]
            final = None
            for i in range(0, len(predictedIntents)):
                # if no cosine match, ignore the prediction of that word
                if useCosine and self.InputWordToCosineSim(cosines, i) == 0.0:
                    final = curr
                    continue
                # sum intent probabs
                sums = curr + predictedIntents[i][1]
                final = sums
            # choose the highest probability sum
            final = list(final)
            index = final.index(max(final))
            chosenProb = max(final)
            chosen = intents[index]

            # not sure how to handle this part well, min predict val scales with corpus size
            if chosenProb < 0.3:
                return 'NoMatch'
            else:
                return chosen
        else:
            # get the "best" intent match for single word inputs - usually binary for 1 word input
            chosenProb = max(predictedIntents[0][1])
            index = list(predictedIntents[0][1]).index(chosenProb)
            chosen = intents[index]

            if useCosine and self.InputWordToCosineSim(cosines, 0) == 0.0:
                return 'NoMatch'
            else:
                return chosen

    def InputWordToCosineSim(self, cosines, index):
        try:
            return cosines[index]
        except:
            return 1.0

    def AssembleClassifier(self, useStopwords, loadFromPickle=False):
        self.encoder = LabelEncoder()
        self.encLabels = self.encoder.fit_transform(self.labels)

        if not loadFromPickle:
            self.docsTfIdf = self.Vectorize(self.data, useStopwords)
            self.classifier = ComplementNB(alpha=0.0001).fit(self.docsTfIdf, self.encLabels)


    # for evaluating model with a test set+labels
    def Evaluate(self, fromPickle=False):
        if not fromPickle:
            encoder = LabelEncoder()
            encLabels = encoder.fit_transform(self.labels)
            self.trainData, self.testData, self.trainLabels, self.testLabels = train_test_split(self.data, encLabels,
                                                                                                test_size=0.25,
                                                                                                random_state=1)
            trainTf = self.Vectorize(self.trainData)
            temp = self.vectorizer.transform(self.testData)
            testTf = self.transformer.transform(temp)
            self.classifier = ComplementNB(alpha=1).fit(trainTf, self.trainLabels)

            prediction = self.classifier.predict(testTf)
            print('\nAccuracy:')
            print(accuracy_score(self.testLabels, prediction))
            print(set("Labels: "))
            print("NOTE: these labels may not be in order in regards to the Confusion Matrix.")
            print(set(encoder.inverse_transform(encLabels)))
            print('\nConfusion matrix:')
            print(confusion_matrix(self.testLabels, prediction))
            print('\nF1 Score:')
            print(f1_score(self.testLabels, prediction, average='micro'))

    # save stuff
    def SaveModel(self, modelName, vectName, tfidfName):
        with open(os.getcwd() + '\\Pickles\\Classifiers\\' + modelName + ".pickle", "wb") as f:
            pickle.dump(self.classifier, f)

        with open(os.getcwd() + '\\Pickles\\Vectorizers\\' + vectName + ".pickle", "wb") as f2:
            pickle.dump(self.vectorizer, f2)

        with open(os.getcwd() + '\\Pickles\\Vectorizers\\' + tfidfName + ".pickle", "wb") as f3:
            pickle.dump(self.transformer, f3)

        with open(os.getcwd() + '\\Pickles\\Vectorizers\\' + "docsTfIdf.pickle", "wb") as f4:
            pickle.dump(self.docsTfIdf, f4)


