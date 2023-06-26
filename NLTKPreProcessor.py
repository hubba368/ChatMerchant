import nltk
import numpy as np
import random
import os
import string

from nltk.corpus import stopwords
from nltk.tree import Tree
from nltk.chunk import RegexpParser
from ChatbotUtils import ExtractProductAttribs, ExtractProductNameToString

class NLTKPreProcessor():

    def GetCorporaFromFile(self, path):
        with open(path, encoding='utf-8', errors='ignore', mode='r') as f:
            label = f.readline().strip()
            content = f.read()
        return content, label

    def Tokenize(self, raw, removeStopWords, tokenizeByWord=True, bigram=False):
        if bigram:
            tokens = nltk.word_tokenize(raw)
            tokens = self.RemovePunctuation(tokens, removeStopWords)
            bg = list(nltk.bigrams(tokens))
            return bg
        if tokenizeByWord:
            tokens = nltk.word_tokenize(raw)
            tokens = self.RemovePunctuation(tokens, removeStopWords)
            return tokens
        else:
            tokens = nltk.sent_tokenize(raw)
            tokens = [nltk.word_tokenize(t) for t in tokens]
            tokens = self.RemovePunctuation(tokens, removeStopWords)
            return tokens

    def LemmatizeTokens(self, tokens, isLemmatizer):
        newTokens = []

        if not isLemmatizer:
            stemmer = nltk.stem.SnowballStemmer('english')
            
            for token in tokens:
                newTokens.append(stemmer.stem(token))
            return newTokens
        else:
            lemmatiser = nltk.stem.WordNetLemmatizer()

            for token in tokens:
                newTokens.append(lemmatiser.lemmatize(token.lower()))
                #newTokens.append(lemmatiser.lemmatize(token[1].lower()))
            return newTokens

    def GetPOSTags(self, tokens):
        tags = []
        for words in tokens:
            tags = nltk.pos_tag(words, tagset='universal')

        return tags

    # get Noun Phrase Chunks to list structure instead of default Tree
    def GetNPChunks(self, posTaggedTokens, grammars, grammarType):
        test = []
        #print(posTaggedTokens)
        for i in range(0, len(grammars)):
            chunker = nltk.RegexpParser(grammars[i])
            result = chunker.parse(posTaggedTokens)

            for tree in result.subtrees(filter=lambda t: t.label() == grammarType):
                test.append(list(zip([x[0]for x in tree], [x[1]for x in tree])))

        test2 = list(test)
        # extract based on intended output - chunker outputs as list of tuples
        if grammarType == 'NAME' and len(test2) > 0:
            final = self.TraverseChunks(test2)
            final = final[0][0]
            return final
        elif grammarType == 'ITEMSTART' and len(test2) > 0:
            final = ExtractProductNameToString(self.TraverseChunks(test2, True))
            return final

    def TraverseChunks(self, sortedChunks, noSort=False):
        # loop backwards because that's where our needed words usually go
        # get a sublist of required entities/words
        result = []
       # print(sortedChunks)
        for t in reversed(sortedChunks):
            for t2 in reversed(t):
                if noSort:
                    result.append([t2[0], t2[1]])
                    continue
                if t2[1] == 'NOUN' or 'ADJ' or 'ADV' or 'ADP' or 'NUM':
                    result.append([t2[0], t2[1]])
        if len(result) > 0:
           # print(result)
            return result

    # at minimum remove punct so they don't get extracted as features
    def RemovePunctuation(self, text, removeStopWords):
        stops = None
        if removeStopWords:
            stops = set(stopwords.words('english') + list(string.punctuation))
        else:
            stops = list(string.punctuation)

        final = [word for word in text if word not in stops]
        return final









