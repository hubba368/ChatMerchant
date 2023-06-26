import numpy as np
import random
import KnowledgeBase
from sklearn.metrics.pairwise import cosine_similarity

# Static Responses
LossOutputs = ["CUSTOMER: Yeah, I think I'm gonna pass.", "CUSTOMER: I think I'm going to leave. Thanks.",
               "CUSTOMER: I don't trust what you've said to me. I'm leaving."]
WinOutputs = ["CUSTOMER: I think I'll take it!", "CUSTOMER: You have such a way with words. I'll take it.",
              "CUSTOMER: Sounds amazing! I'll buy it right away!"]

NewDayLines = ["It is a new day, and every new day brings a new customer.",
               "It is a new day, and you set up shop, ready for whatever."]

EndDayLossLines = ["You close your shop, a feeling of defeat washing over you.",
                   "The day is over. You feel disappointed in your lack of a sale."]

EndDayWinLines = ["You close your shop, feeling good for making a sale.",
                  "The day comes to a close, with another happy customer."]

EndDayEarlyLines = ["You decide to close your shop early. Your customer leaves in a huff.",
                    "You decide to pack up early. You feel as if that wasn't the right choice."]

# nltk.regexparser grammars
NameGrammars = ["NAME:{<NOUN>}", "NAME:{<DET><NOUN><NOUN>}", "NAME:{<PRON><NOUN><VERB><ADJ>}",
                "NAME:{<PRON><NOUN><VERB><ADV|DET|ADJ>+}"]
StartGameGrammars = ["ITEMSTART:{<PRON|VERB><ADV|ADJ|VERB|DET|NOUN|ADP|NUM>+}"]
ItemGrammars = ["ITEM:{<PRON|VERB><ADV|ADJ|VERB|DET|NOUN|ADP|CONJ>+}"]

# FUNCTIONS
def OutputBotResponse(response, botName=None):
    if botName is not None:
        return botName + ": " + response
    else:
        return "CUSTOMER: " + response

# soft cosine - cosine angle with semantic meanings output (words of similar cosine score)
def GetVectorSimilarity(query, document):
    csm = cosine_similarity(query, document)
    return csm

# regex entity extraction
def ExtractProductNameToString(words):
    # extract words till reach DET
    # if CONJ exist then keep going until the next word or a DET
    result = ""
    temp = []
    for t in words:
        word = t[0]
        tag = t[1]
        if tag == 'DET':
             break
        if tag == 'NOUN' or 'ADJ':
            temp.append(word)
    if len(temp) > 0:
        for i in reversed(temp):
            result += i + " "
        return result
    else:
        return None

# 'Main' cosine info extraction
def ExtractProductAttribs(words, vect, trans, docsTfIdf, knowledgeBase):
    # return string, then add attribs to knowledgebase
    totalWords = len(words)
    if totalWords == 0:
        print("Item Descriptor Bag of Words is empty.")
        return None
    matches = set()
    # have to check cosine sim per word in input
    for i in range(0, totalWords):
        currWord = [words[i]]
        cosCount = vect.transform(currWord)
        cosTfIdf = trans.transform(cosCount)
        csm = GetVectorSimilarity(cosTfIdf, docsTfIdf)
        csm = csm.flatten()

        temp = np.partition(csm, -totalWords)
        closestSim = temp[-totalWords:]
        closestSim = list(closestSim)
        highestMatch = max(closestSim)

        if highestMatch == 0.0:
            continue
        # get all the cosine matches to the input and store them for the 'Day'
        matches.add(words[i])

    # check whether it is positive or negative according to current BoW
    if matches is not None and len(matches) > 0:
        posTotal = GetProductInputLabel(matches, knowledgeBase, 'Pos')
        negTotal = GetProductInputLabel(matches, knowledgeBase, 'Neg')
        label = 'Pos' if posTotal > negTotal else 'Neg'
        return list(matches), label
    else:
        # if nothing just coin flip response
        flip = random.randint(0, 1)
        label = 'Pos' if flip == 1 else 'Neg'
        return [], label

def GetProductInputLabel(words, kb, label):
    total = 0
    for i in kb.CurrentCustomerPreferredAttribs[label]:
        for j in words:
            if j == i:
                total += 1
    return total

