import os
import pickle

import ChatbotUtils
import Classifier
import NLTKPreProcessor
import FiniteStateMachine
from FiniteStateMachine import State
import KnowledgeBase
from ChatbotUtils import ExtractProductAttribs
import random
import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer



# init main game stuff
fsm = FiniteStateMachine.StateMachine()
posTrust, negTrust, totalPosTrust, totalNegTrust = 0, 0, 0, 0
currMaxPostTrust, currMaxNegTrust = random.randint(5, 10), -random.randint(5, 10)
budget = 500
currentDays = 1
knowledgeBase = KnowledgeBase.KnowledgeBase()

def InitGame():
    # init NLP stuff - check if pickles exist
    pickleCFPaths = []
    pickleVectPaths = []
    pp = NLTKPreProcessor.NLTKPreProcessor()
    cf = None

    for fileName in os.listdir(os.getcwd() + '\\Pickles\\Classifiers\\'):
        if fileName.endswith('.pickle'):
            pickleCFPaths.append(os.getcwd() + '\\Pickles\\Classifiers\\' + fileName)

    for fileName in os.listdir(os.getcwd() + '\\Pickles\\Vectorizers\\'):
        if fileName.endswith('.pickle'):
            pickleVectPaths.append(os.getcwd() + '\\Pickles\\Vectorizers\\' + fileName)
    # try pickle loading of stuff
    if len(pickleCFPaths) > 0:
        with open(pickleCFPaths[0], "rb") as f:
            smallTalkerCF = pickle.load(f)
            cf = Classifier.Classifier()
            cf.classifier = smallTalkerCF

            with open(pickleVectPaths[0], "rb") as f2:
                cf.docsTfIdf = pickle.load(f2)
            with open(pickleVectPaths[1], "rb") as f3:
                cf.transformer = pickle.load(f3)
            with open(pickleVectPaths[2], "rb") as f4:
                cf.vectorizer = pickle.load(f4)

            cf.LoadData(pp, os.getcwd() + '\\Corpora\\Smalltalker', False)
            cf.LoadResponses(os.getcwd() + '\\Responses\\Smalltalker')
            cf.AssembleClassifier(True, True)
            print("pickles Loaded.")
    else:
        # rebuild
        print("Couldn't find ChatMerchant pickles. Training models from scratch.\n")
        cf = RebuildModels(pp)

    print("Welcome to ChatMerchant. A game where you play as a Travelling merchant,")
    print("trying to convince a new customer to buy your latest product.")
    print(
        "The goal of the game is to raise the Positive Trust value to max, convincing the customer to buy your product.")
    print("If Negative Trust falls too low, the customer will leave, and you will lose that day.")
    print("Each day, your expenses will fall as you need to pay the running costs.")
    print("If your expenses fall to zero, you lose the game!")
    print("type '/evaluateNew' to evaluate new models with train_test_split(random_state=1).")
    print("Type '/stop' to quit the game.")
    print("Type '/start' to begin!")

    return cf, pp

def RebuildModels(preproc):
    cf = Classifier.Classifier()
    # load train data + specific responses
    cf.LoadData(preproc, os.getcwd() + '\\Corpora\\Smalltalker', False)
    cf.LoadResponses(os.getcwd() + '\\Responses\\Smalltalker')

    cf.AssembleClassifier(False)
    cf.SaveModel("SmallTalker", "SmallTalkerVectorizer", "SmalltalkerTfIdfs")

    return cf

def StartNewDay():
    knowledgeBase.ClearCurrentKnowledge()
    # assemble a new 'Item attributes' BOW model per day
    itemCorp, label = pp.GetCorporaFromFile(os.getcwd() + '\\Corpora\\ItemDescriptors.txt')
    tokens = pp.Tokenize(itemCorp, False)
    # 'randomize' the corpus so theres some variation with each customer
    # not perfect but is functional at least
    # would bring potential perf issues with much larger corpus
    random.shuffle(tokens)
    totalWords = len(tokens)-1
    seenPosWords = set()
    seenNegWords = set()

    for i in range(0, totalWords):
        num = random.randint(i, totalWords)
        flip = random.randint(0, 1)
        if flip == 0 and tokens[num] not in seenNegWords:
            seenPosWords.add(tokens[num])
            knowledgeBase.AddCustomerPrefAttribs(tokens[num], "Pos")
        elif flip == 1 and tokens[num] not in seenPosWords:
            seenNegWords.add(tokens[num])
            knowledgeBase.AddCustomerPrefAttribs(tokens[num], "Neg")

    # assemble bag of words
    itemVect = CountVectorizer()
    itemCounts = itemVect.fit_transform(tokens)
    itemXtf = TfidfTransformer(use_idf=True, sublinear_tf=False)
    itemTfIdf = itemXtf.fit_transform(itemCounts)
    fsm.SwitchState(State.Intro)

    # store model for later use in current Day
    knowledgeBase.AddCurrDayLM(itemVect, itemXtf, itemTfIdf)

    print(ChatbotUtils.NewDayLines[random.randint(0, len(ChatbotUtils.NewDayLines)-1)])
    print("A new customer has approached you.")

# huge disgusting main below
if __name__ == "__main__":
    smallTalker, pp = InitGame()
    itemVect, itemTrans, docsTfIdf = None, None, None
    stop = False
    running = False
    hasLost = False

    while not stop:
        query = input("Say: ")

        if query == "/start" and not running:
            StartNewDay()
            running = True
            continue

        elif (query == "/stop" and running) or hasLost:
            # output game stats and quit
            running = False
            print("Game Over!!")
            print(f"Days Completed: {currentDays}")
            print(f"Total Positive Gains: {totalPosTrust}")
            print(f"Total Negative Gains: {totalNegTrust}")
            time.sleep(10)
            stop = True

        elif query == "/evaluateNew":
            print("Evaluating Smalltalker CNB Classifier:")
            smallTalkEval = Classifier.Classifier()
            smallTalkEval.LoadData(pp, os.getcwd() + '\\Corpora\\Smalltalker', False)
            smallTalkEval.Evaluate()
        else:
            if fsm.currentState == State.NewDay:
                StartNewDay()

            elif fsm.currentState == State.Intro:
                # tokenize->posTag->Phrase Chunk->lemmatize->Intent Match->Response/State Change->Game Stuff
                query = query.lower()
                query = pp.Tokenize(query, False, True)
                query = pp.LemmatizeTokens(query, True)

                output, secret = fsm.ExecuteState(query, pp, smallTalker, knowledgeBase)
                print(output)
                if secret and secret is not None:
                    print(ChatbotUtils.EndDayEarlyLines[random.randint(0, len(ChatbotUtils.EndDayEarlyLines) - 1)])
                    fsm.SwitchState(State.EndDay)

            elif fsm.currentState == State.Main:
                # this will use cosine sim to get phrase entities (because they can be anything)
                queryActual = query
                query = query.lower()
                query = pp.Tokenize(query, False, True)
                query = pp.LemmatizeTokens(query, True)

                output, trustVal = fsm.ExecuteState(query, pp, smallTalker, knowledgeBase, queryActual)
                print(output)

                # handle sneaky game ending secret
                if trustVal < currMaxNegTrust and trustVal is not None:
                    negTrust -= -trustVal
                    print(ChatbotUtils.EndDayEarlyLines[random.randint(0, len(ChatbotUtils.EndDayEarlyLines) - 1)])
                    print("Enter anything to continue.")
                    fsm.SwitchState(State.EndDay)
                    continue

                # handle trust changes/day ends
                if trustVal is not None and trustVal > 0:
                    print(ChatbotUtils.OutputBotResponse("What else can it do?", knowledgeBase.CurrentCustomerName))
                    posTrust += trustVal
                    print("TRUST++")
                    print(f"{posTrust}/{currMaxPostTrust}")
                if trustVal is not None and trustVal < 0:
                    print(ChatbotUtils.OutputBotResponse("What else can it do?", knowledgeBase.CurrentCustomerName))
                    negTrust -= -trustVal
                    print("TRUST--")
                    print(f"{negTrust}/{currMaxNegTrust}")

                if posTrust >= currMaxPostTrust:
                    print(ChatbotUtils.OutputBotResponse(
                        ChatbotUtils.WinOutputs[random.randint(0, len(ChatbotUtils.WinOutputs) - 1)]))
                    fsm.SwitchState(State.EndDay)
                elif negTrust <= currMaxNegTrust:
                    print(ChatbotUtils.OutputBotResponse(
                        ChatbotUtils.LossOutputs[random.randint(0, len(ChatbotUtils.LossOutputs) - 1)]))
                    fsm.SwitchState(State.EndDay)
                continue

            elif fsm.currentState == State.EndDay:
                print("Day Finished...")

                if posTrust == currMaxPostTrust:
                    print(ChatbotUtils.EndDayWinLines[random.randint(0, len(ChatbotUtils.EndDayWinLines)-1)] + '\n')
                    print("Daily Profit:")
                    print("\tBudget++ " + ' 150!')
                    budget += 150
                else:
                    print(ChatbotUtils.EndDayLossLines[random.randint(0, len(ChatbotUtils.EndDayLossLines) - 1)] + '\n')

                print("Daily Expenses:")
                print("\tBudget--" + ' 100')
                budget -= 100
                currentDays += 1
                print("\nCurrent Stats:")
                print(f"\tCurrent Budget: {budget}")
                print(f"\tDays Completed: {currentDays}")
                print(f"\tTotal Positive Gains: {posTrust}")
                print(f"\tTotal Negative Gains: {negTrust}")

                totalPosTrust += posTrust
                totalNegTrust -= negTrust
                posTrust, negTrust = 0, 0
                time.sleep(3)

                if budget <= 0:
                    hasLost = True

                fsm.SwitchState(State.NewDay)
                print("Enter anything to continue.")
                continue
