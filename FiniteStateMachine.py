from enum import Enum
import random
import ChatbotUtils
import KnowledgeBase

# basic switch states to handle response 'types'
# e.g. if not in intro state a 'Greeting' intent will output with something different
class State(Enum):
    Intro = 1
    Outro = 2
    Main = 3
    NewDay = 4
    EndDay = 5
    NPCAskQuestion = 6
    NoState = 7

class StateMachine():

    def __init__(self):
        self.currentState = State.NoState

    def SwitchState(self, newState):
        self.currentState = newState

    def ExecuteState(self, query, preproc, cf, kb, inputStr=None):
        intent = cf.Predict(query, True)
        queryActual = inputStr
        customerName = None

        if kb.CurrentCustomerName != "":
            customerName = kb.CurrentCustomerName

        if self.currentState == State.Intro:
            output, secret = self.IntroState(query, preproc, cf, intent, customerName, kb)
            return output, secret

        if self.currentState == State.Main:
            output, trustVal = self.MainState(query, preproc, cf, intent, customerName, queryActual, kb)
            return output, trustVal

    def IntroState(self, query, preproc, cf, intent, customerName, kb):
        # tokenize->posTag->Phrase Chunk->lemmatize->Intent Match->Response/State Change->Game Stuff
        # handle errors/unwanted intents
        if intent is None or intent == "GiveProductInfo":
            return ChatbotUtils.OutputBotResponse("Uh... what?", customerName), None

        # get responses
        matchedResponses = cf.responses[intent]
        responses = matchedResponses[1]
        response = str(responses[random.randint(0, len(responses) - 1)])

        # handle entity extraction related intents
        if intent == "GiveName":
            if kb.KnowsPlayerName:
                return ChatbotUtils.OutputBotResponse("Um... " + kb.PlayerName + ", you already told me your name.",
                                                      customerName), None
            elif not kb.KnowsPlayerName:
                temp = [query]
                post = preproc.GetPOSTags(temp)
                output = preproc.GetNPChunks(post, ChatbotUtils.NameGrammars, 'NAME')

                if output is None:
                    return ChatbotUtils.OutputBotResponse("Uh... what?", customerName), None

                kb.AddPlayerNameToKB(output.capitalize())
                return ChatbotUtils.OutputBotResponse(response + " " + kb.PlayerName + ".", customerName), None

        if intent == "StartGame":
            temp = [query]
            post = preproc.GetPOSTags(temp)
            output = preproc.GetNPChunks(post, ChatbotUtils.StartGameGrammars, 'ITEMSTART')

            if intent == "StartGame" and output is None:
                return "HINT: try presenting your item like you would to someone in real life! e.g. 'Take a look at this torch'", None

            self.SwitchState(State.Main)
            return ChatbotUtils.OutputBotResponse("A " + output + "? " + response, customerName), None

        # handle other intents/ones with certain criteria
        if intent == "GreetingsIntro" and kb.KnowsPlayerName:
            return ChatbotUtils.OutputBotResponse("Uh... hello, " + kb.PlayerName + ".", customerName), None
        if intent == "GoodbyeOutro":
            return ChatbotUtils.OutputBotResponse(response, customerName), True

        if intent == "AskName" and customerName is None:
            kb.AddCustomerName()
            return ChatbotUtils.OutputBotResponse(response + kb.CurrentCustomerName + ".", customerName), None
        elif intent == "AskName" and customerName is not None:
            return ChatbotUtils.OutputBotResponse("Uh... " + kb.PlayerName + ", I already told you my name.",
                                                  customerName), None
        # if no entities/match give no match
        return ChatbotUtils.OutputBotResponse(response, customerName), None


    def MainState(self, query, preproc, cf, intent, customerName, inputQuery, kb):
        # extract product attributes intents
        if intent == 'GiveProductInfo':
            output, label = ChatbotUtils.ExtractProductAttribs(query, kb.ProductVect, kb.ProductTrans,
                                                                   kb.ProductTfIdfs, kb)
            matchedResponses = cf.responses[intent + label]
            trustVal = int(matchedResponses[0])
            trustVal = -trustVal if label == "Neg" else trustVal
            responses = matchedResponses[1]
            response = str(responses[random.randint(0, len(responses) - 1)])
            isKnown = False
            queryActual = inputQuery # redo this !!!
            queryActual = queryActual.replace("your", "my").replace("you", "me")

            if len(output) == 0:
                # HINT: try describing your item of 'what it can do'! e.g. 'it can make you look younger'
                # output random
                return ChatbotUtils.OutputBotResponse(queryActual + "? " + response, customerName), trustVal
            else:
                isKnown = kb.AddNewProductAttribs(output)
                if isKnown:
                    return ChatbotUtils.OutputBotResponse("You already told me that.", customerName), 0
                else:
                    return ChatbotUtils.OutputBotResponse(queryActual + "? " + response, customerName), trustVal

        # no matches/any other thing
        matchedResponses = cf.responses[intent]
        responses = matchedResponses[1]
        response = str(responses[random.randint(0, len(responses) - 1)])

        if intent == "GreetingsIntro" and kb.KnowsPlayerName:
            return ChatbotUtils.OutputBotResponse("Yes.. Hi, " + kb.PlayerName + ".", customerName), 0
        if intent == "GiveName":
            if kb.KnowsPlayerName:
                return ChatbotUtils.OutputBotResponse("I know that, " + kb.PlayerName + ".", customerName), 0
            else:
                return ChatbotUtils.OutputBotResponse("Is that really relevant now?", customerName), 0
        if intent == "GoodbyeOutro":
            return ChatbotUtils.OutputBotResponse(response, customerName), -99

        return ChatbotUtils.OutputBotResponse(response, customerName), 0
