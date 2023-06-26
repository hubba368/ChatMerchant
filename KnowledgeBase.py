import random
from collections import defaultdict
import os
import nltk

# its not 'pythonic' i know :(
class KnowledgeBase:
    def __init__(self):
        self.PlayerName = ""
        self.CurrentCustomerName = None
        self.CurrentProductName = ""
        self.CurrentProductAttributes = []
        self.KnowsPlayerName = False
        self.CurrentCustomerPreferredAttribs = defaultdict(list)

        self.ProductVect = None
        self.ProductTfIdfs = None
        self.ProductTrans = None

    def AddPlayerNameToKB(self, playerName):
        self.PlayerName = playerName
        self.KnowsPlayerName = True

    def AddCustomerName(self):
        with open(os.getcwd() + "\\Corpora\\CustomerNames.txt", "r") as f:
            n = f.read()
            names = nltk.word_tokenize(n)
            self.CurrentCustomerName = names[random.randint(0, len(names)-1)]

    def AddNewProductName(self, productName):
        self.CurrentProductName = productName

    # collected from player input
    def AddNewProductAttribs(self, attribs):
        for i in attribs:
            if i in self.CurrentProductAttributes:
                return True
            else:
                self.CurrentProductAttributes += attribs
                return False

    # generated per NewDay state
    def AddCustomerPrefAttribs(self, key, attribs):
        self.CurrentCustomerPreferredAttribs[key].append(attribs)

    def AddCurrDayLM(self, vect, trans, tfidfs):
        self.ProductVect = vect
        self.ProductTrans = trans
        self.ProductTfIdfs = tfidfs

    def ClearCurrentKnowledge(self):
        self.PlayerName = ""
        self.CurrentCustomerName = None
        self.CurrentProductName = ""
        self.CurrentProductAttributes.clear()
        self.KnowsPlayerName = False
        self.CurrentCustomerPreferredAttribs.clear()
        self.ProductVect = None
        self.ProductTrans = None
        self.ProductTfIdfs = None
