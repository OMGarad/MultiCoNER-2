
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel, RobertaModel
import torch
import pandas as pd
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from torch.utils import data
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import recall_score, precision_score
import tqdm
from joblib import dump, load
from transformers import RobertaModel, BertModel
from transformers import RobertaTokenizer, RobertaForSequenceClassification, XLMRobertaTokenizer, BertTokenizer, AutoModel
import numpy as np
import pandas as pd
from sklearn.svm import SVC
import xgboost
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import random
import os
import matplotlib.pyplot as plt


def generateEmbeddings(modelname:str, sentencelist:list, embeddingtype:str):
    '''
    Generates word or sentence embeddings using a pretrained Roberta language model
    input:
        modelname: text string with name of the pretrained model and corresponding tokenizer, example: "pdelobelle/robbert-v2-dutch-base".
        sentencelist: a list of text strings for which we will generate embeddings.
        embeddingtype: either "word" or "sentence". Sentence embeddings are generated using the CLS token (first of each sequence).
    '''
    if "xlm" in modelname.lower():
        tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base", add_prefix_space=True, pad_to_max_length=True, max_length=128)
        pretrainedmodel = AutoModel.from_pretrained(modelname)
    elif "roberta" in modelname.lower() or "robbert" in modelname.lower() :
        tokenizer = RobertaTokenizer.from_pretrained(modelname, add_prefix_space=True, pad_to_max_length=True, 
        max_length=128)
        pretrainedmodel = AutoModel.from_pretrained(modelname)
    else:
        tokenizer = BertTokenizer.from_pretrained(modelname, add_prefix_space=True, pad_to_max_length=True, 
        max_length=128)
        pretrainedmodel = BertModel.from_pretrained(modelname)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encodedcorpus = PranayTokenizer(sentencelist, tokenizer)

    neuralmodel = Net(model=pretrainedmodel, embeddingtype=embeddingtype)
    neuralmodel.to(device)
    neuralmodel = nn.DataParallel(neuralmodel)

    outputlist = []
    for i, batch in tqdm.tqdm(enumerate(encodedcorpus)):
        words = batch
        bert_embeds = neuralmodel(words)
        bert_embeds = bert_embeds.cpu().numpy()
        outputlist.append(bert_embeds)

    return outputlist


def PranayTokenizer(corpus, pret_tokenizer):
    tokenizedcorpus = []
    for sent in corpus:
            sent_tokens = pret_tokenizer.encode(sent, padding='max_length', truncation=True, max_length=128)
            tokenizedcorpus.append(sent_tokens)
    return tokenizedcorpus    


#Function to remove any line-breaks at the end of strings or line-breaks that separate strings
def RemoveLineBreak(contents):
    cleanedContents = []
    
    for i in range(len(contents)):
            if contents[i] == '\n':
                    continue
            cleanedContents.append(contents[i][:-1])

    return cleanedContents


#Function to extract the tokens where the format is:        [TOKEN _ _ TAG]
def ObtainTokens(contents):
    dashes = " _ _ "

    for _ in range(len(contents)):
            if dashes in contents[_]:
                    index = contents[_].find(dashes)
                    contents[_] = contents[_][:index]

    return contents


#Function to convert the tokens in each line into a sentence (each sentence is separated by the '# id' phrase) 
def CreateSentences(contents):
  sentences = []
  i = 1

  while i != len(contents)-1:
    sntnce = ""
    while(contents[i+1][:4]!="# id"):
      if sntnce == "":
        sntnce = contents[i]
      else: 
        sntnce = sntnce + " " + contents[i]
      i = i + 1
      if i == len(contents)-2:
        break
    sntnce = sntnce + " " + contents[i]
    sentences.append(sntnce)
    if i == len(contents)-2:
        break
    i = i + 2

  return sentences

#Given a file this function converts all the tokens into proper sentences and returns it
def SentenceGenerator(training_file):

    file = open(training_file)
    contents= file.readlines()
    
    contents = RemoveLineBreak(contents)
    contents = ObtainTokens(contents)
    sentences = CreateSentences(contents)

    return sentences

#Function to remove the '# id ..... domain=lang" line
def RemoveIDs(contentsFine):
    substr = "domain="
    cleanedContentsFine = []

    for _ in range(len(contentsFine)):
        if substr in contentsFine[_]:
            continue
        cleanedContentsFine.append(contentsFine[_])

    return cleanedContentsFine


#Function to extract the B,I tags from the line [TOKEN _ _ TAG]
#Input is the tag: B-, I- or O
def ExtractOnlyTags(contentsFine):

    for i in range(len(contentsFine)):
        if contentsFine[i]=='O':
            continue
        else:
            contentsFine[i] = contentsFine[i][2:]
    
    return contentsFine

#Function to get the fine tags from each row (used with ExtractOnlyTags)
def ObtainFineTags(contentsFine):
    dashes = " _ _ "

    for _ in range(len(contentsFine)):
            if dashes in contentsFine[_]:
                    index = contentsFine[_].find(dashes)
                    contentsFine[_] = contentsFine[_][index + 5:]   #because the " _ _ " string is 5 chars long
    
    contentsFine = ExtractOnlyTags(contentsFine)
    return contentsFine


#Given a file, returns a list of only the fine-grained tags from each row (not including O)
def FineGrainedTags(originalFile):

    file = open(originalFile)
    contentsFine= file.readlines()

    contentsFine = RemoveLineBreak(contentsFine)
    contentsFine = RemoveIDs(contentsFine)
    contentsFine = ObtainFineTags(contentsFine)

    return contentsFine

#Rmoving Tabs from the tsv file
def RemoveTabs(contentsPred):

    cleanedContentsPred = []

    for i in range(len(contentsPred)):
        if contentsPred[i] == '\t':
            continue
        cleanedContentsPred.append(contentsPred[i])

    return cleanedContentsPred     


#Tags that are incorrectly deemed to be of the given domain are given the Wrong tag
def WrongTags(domain, fineGrainedDomainTags):
    if domain == "Medical":
        for k in range(len(fineGrainedDomainTags)):
            if fineGrainedDomainTags[k]== 'Medication/Vaccine' or fineGrainedDomainTags[k]=='MedicalProcedure' or fineGrainedDomainTags[k]== 'AnatomicalStructure' or fineGrainedDomainTags[k]=='Symptom' or fineGrainedDomainTags[k]=='Disease':
                continue
            else:
                fineGrainedDomainTags[k]= 'Wrong'
    elif domain == "Location":
        for k in range(len(fineGrainedDomainTags)):
            if fineGrainedDomainTags[k]== 'Facility' or fineGrainedDomainTags[k]=='OtherLOC' or fineGrainedDomainTags[k]=='Station' or fineGrainedDomainTags[k]=='HumanSettlement':
                continue
            else:
                fineGrainedDomainTags[k]= 'Wrong'
    elif domain == "CreativeWorks":
        for k in range(len(fineGrainedDomainTags)):
            if fineGrainedDomainTags[k]== 'VisualWork' or fineGrainedDomainTags[k]=='MusicalWork' or fineGrainedDomainTags[k]=='WrittenWork' or fineGrainedDomainTags[k]=='ArtWork' or fineGrainedDomainTags[k]=='Software' or fineGrainedDomainTags[k]=='OtherCW':
                continue
            else:
                fineGrainedDomainTags[k]= 'Wrong'
    elif domain == "Group":
        for k in range(len(fineGrainedDomainTags)):
            if fineGrainedDomainTags[k]== 'MusicalGRP' or fineGrainedDomainTags[k]=='PublicCORP' or fineGrainedDomainTags[k]=='PrivateCORP' or fineGrainedDomainTags[k]=='OtherCORP' or fineGrainedDomainTags[k]=='SportsGRP' or fineGrainedDomainTags[k]=='TechCORP' or fineGrainedDomainTags[k]=='ORG' or fineGrainedDomainTags[k]=='CarManufacturer' or fineGrainedDomainTags[k]=='AerospaceManufacturer':
                continue
            else:
                fineGrainedDomainTags[k]= 'Wrong'
    elif domain == "Person":
        for k in range(len(fineGrainedDomainTags)):
            if fineGrainedDomainTags[k]== 'OtherPER' or fineGrainedDomainTags[k]=='SportsManager' or fineGrainedDomainTags[k]=='Cleric' or fineGrainedDomainTags[k]=='Politician' or fineGrainedDomainTags[k]=='Athlete' or fineGrainedDomainTags[k]=='Artist' or fineGrainedDomainTags[k]=='Scientist':
                continue
            else:
                fineGrainedDomainTags[k]= 'Wrong'
    elif domain == "Product":
        for k in range(len(fineGrainedDomainTags)):
            if fineGrainedDomainTags[k]== 'OtherPROD' or fineGrainedDomainTags[k]=='Drink' or fineGrainedDomainTags[k]=='Food' or fineGrainedDomainTags[k]=='Vehicle' or fineGrainedDomainTags[k]=='Clothing':
                continue
            else:
                fineGrainedDomainTags[k]= 'Wrong'
    
    return fineGrainedDomainTags 
     
                
#Getting the training data for the joint system  
def DomainCorpusGenerator(PredictionFile, domain, test_or_train):

    domainWordsArray= [] #Will contain all the domain words to train (these words are words predicted to be domain by the coarse-grained model)
    domainWordsIndex = [] #Will contain the indices where the domain words are located (from the same predicted tags file as above)
    fineGrainedDomainTags = [] #Will contain the tags of the domain words (these are the actual tags of the words in the fine-grained training data set)


    file = open(PredictionFile)
    contentsPred= file.readlines()
    contentsPred = RemoveLineBreak(contentsPred)
    contentsPred = RemoveTabs(contentsPred)
    
    for _ in range(len(contentsPred)):
        if domain in contentsPred[_]:
            index = contentsPred[_].find(domain)
            domWord = contentsPred[_][:index-3]
            domainWordsArray.append(domWord)
            domainWordsIndex.append(_)
    
    print("_________THIS IS FOR DOMAIN WORD__________________")
    print(len(domainWordsArray))

    
    #Depending on whether we want to get the fine-grained tags for the train or dev set, we specify
    if test_or_train == 'train':
        trainingTags = FineGrainedTags('/home/omkar/multiconer-baseline/en-train90.conll')
    elif test_or_train == 'test':
        trainingTags = FineGrainedTags('/home/omkar/multiconer-baseline/en-dev.conll')
        
    for i in range(len(domainWordsIndex)):
        fineGrainedDomainTags.append(trainingTags[domainWordsIndex[i]])

    print("___________________________THIS IS FOR TAGS OF DOMAIN WORDS_____________________")
    
    fineGrainedDomainTags = WrongTags(domain, fineGrainedDomainTags)
    print(len(fineGrainedDomainTags))

    return domainWordsArray, fineGrainedDomainTags


#Getting the data for the upper bound model
def OriginalDomainCorpusGenerator(CoarseTagFile, domain, test_or_train):

    #First get all words from coarse grained
    #Then take all tags from fine grained

    domainWordsArray= [] #Will contain all the domain words to train (these words are words predicted to be domain by the coarse-grained model)
    domainWordsIndex = [] #Will contain the indices where the domain words are located (from the same predicted tags file as above)
    fineGrainedDomainTags = [] #Will contain the tags of the domain words (these are the actual tags of the words in the fine-grained training data set)


    file = open(CoarseTagFile)
    contentsCoarse= file.readlines()
    contentsCoarse = RemoveLineBreak(contentsCoarse)
    contentsCoarse = RemoveIDs(contentsCoarse)
    
    dashes = " _ _ "

    for _ in range(len(contentsCoarse)):
        if domain in contentsCoarse[_]:
            index = contentsCoarse[_].find(dashes)
            medWord = contentsCoarse[_][:index]
            domainWordsArray.append(medWord)
            domainWordsIndex.append(_)
    
    print(len(domainWordsArray))

    #Depending on whether we want to get the fine-grained tags for the train or dev set, we specify
    if test_or_train == 'train':
        trainingTags = FineGrainedTags('/home/omkar/multiconer-baseline/en-train90.conll')
    elif test_or_train == 'test':
        trainingTags = FineGrainedTags('/home/omkar/multiconer-baseline/en-dev.conll')
    print(len(trainingTags))


    for i in range(len(domainWordsIndex)):
        fineGrainedDomainTags.append(trainingTags[domainWordsIndex[i]])
        
    print(len(fineGrainedDomainTags))


    return domainWordsArray, fineGrainedDomainTags
    
    
class Net(nn.Module):
    
    def __init__(self, model, embeddingtype):
        '''
        model should be a pretrained huggingface transformer model
        embeddingtype can be "word" or "sentence"
        '''
        super().__init__()
        self.model = model
        self.embeddingtype = embeddingtype

    def forward(self, sent):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        sent = torch.LongTensor(sent).to(device)
        input_ids = sent.unsqueeze(0)  # Batch size 1
        with torch.no_grad():
            outputs = self.model(input_ids)
            last_hidden_states = outputs.last_hidden_state[0] # The last hidden-state is the first element of the output tuple
        if self.embeddingtype == "word":
            last_hidden_states = torch.mean(last_hidden_states[1:], axis=0)
            return last_hidden_states
        elif self.embeddingtype == "sentence":
            return last_hidden_states[0]

def seed_everything(seed=42):
    """"
    Seed everything.
    """   
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
        
    

def main():

    modelname = "/home/omkar/coarse_ner_baseline_models/XLMR_Base/xlmr_base_eng_coarse_ner_e10/lightning_logs/version_0/checkpoints"

    seed_everything(42)

    print("\n\n----------------------------------------JOINT SYSTEM-----------------------------------------------------------------\n\n")
    print("\n\n----------------------------------------JOINT SYSTEM-----------------------------------------------------------------\n\n")

    print("---------------------------------------------------------------------------------------------------------------------------------------------\n")
    print("------------------------------------             PERSON DOMAIN             ------------------------------------------------------")
    
    personTrainCorpus, personTrainTags = DomainCorpusGenerator('/home/omkar/multiconer-baseline/eng_prediction.txt', domain = "Person", test_or_train= 'train')
    personTestCorpus, personTestTags = DomainCorpusGenerator('/home/omkar/multiconer-baseline/eng_test_prediction.txt', domain = "Person", test_or_train= 'test')
    
    
    #Testing Set
    textTestcorpus = personTestCorpus
    testResult = generateEmbeddings(modelname=modelname, sentencelist=textTestcorpus, embeddingtype="word")
    testResult = np.array(testResult)
    
    fineTesttags = personTestTags
    print(len(fineTesttags))
    print(type(fineTesttags[0]))
    
    #Training Set
    textcorpus = personTrainCorpus
    result = generateEmbeddings(modelname=modelname, sentencelist=textcorpus, embeddingtype="word")
    result = np.array(result)

    finetags = personTrainTags
    print(len(finetags))
    print(type(finetags[0]))

    totalTags = finetags + fineTesttags
    totalTags = np.array(totalTags)
    label_map = {cat:index for index,cat in enumerate(np.unique(totalTags))}
    y_prep = np.asarray([label_map[l] for l in finetags])
    y_prep_Test = np.asarray([label_map[l] for l in fineTesttags])

    print(label_map)

    x_train = result
    y_train = y_prep    
    x_test = testResult
    y_test = y_prep_Test
    
    # print("---------------------------------PERSON - JOINT SYSTEM -----------------------")


 
    X_axis = []
    accuracy = []
    F1 = []
    precision = []
    recall = []
    print("\n\n\n-------------------------------EVALUATION SCORES FOR THE JOINT SYSTEM---------------------------------\n\n\n")

    print("XGBoost\n")

    for i in range(100):
        XGB= XGBClassifier(n_estimators= i+1, random_state = 0)
        XGB.fit(x_train, y_train)

        X_axis.append(i+1)
        y_pred = XGB.predict(x_test)

        accuracy.append(accuracy_score(y_test,y_pred)*100)
        F1.append(f1_score(y_test, y_pred, average='macro', labels=np.unique(y_pred))*100)
        recall.append(recall_score(y_test, y_pred, average='macro', labels=np.unique(y_pred))*100)
        precision.append(precision_score(y_test, y_pred, average='macro', labels=np.unique(y_pred))*100)

        print(len(accuracy))
        print(len(F1))
        print(len(precision))
        print(len(recall))

    #ACCURACY 
    fig1 = plt.figure()
    plt.plot(X_axis, accuracy, marker='o')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Accuracy')
    plt.title('Accuracy v/s Number of Estimators')
    plt.show()
    fig1.savefig('eng-js-xgb-person-acc.png')

    #F1 SCORE
    fig2 = plt.figure()
    plt.plot(X_axis, F1, marker='o')
    plt.xlabel('Number of Estimators')
    plt.ylabel('F1 Score')
    plt.title('F1 Score v/s Number of Estimators')
    plt.show()
    fig2.savefig('eng-js-xgb-person-f1.png')

    #RECALL
    fig3 = plt.figure()
    plt.plot(X_axis, recall, marker='o')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Recall')
    plt.title('Recall v/s Number of Estimators')
    plt.show()
    fig3.savefig('eng-js-xgb-person-recall.png')

    #PRECISION
    fig4 = plt.figure()
    plt.plot(X_axis, precision, marker='o')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Precision')
    plt.title('Precision v/s Number of Estimators')
    plt.show()
    fig4.savefig('eng-js-xgb-person-prec.png')
    
    
    print("---------------------------------------------------------------------------------------------------------------------------------------------\n")
    print("------------------------------------             CREATIVEWORKS DOMAIN             ------------------------------------------------------")
    
    cwcorpus, cwtags = DomainCorpusGenerator('/home/omkar/multiconer-baseline/eng_prediction.txt', "CreativeWorks", "train")
    cwTestCorpus,cwTestTags = DomainCorpusGenerator('/home/omkar/multiconer-baseline/eng_test_prediction.txt', "CreativeWorks", "test")
    
    textcorpus = cwcorpus
    result = generateEmbeddings(modelname=modelname, sentencelist=textcorpus, embeddingtype="word")
    result = np.array(result)

    finetags = cwtags
    print(len(finetags))
    print(type(finetags[0]))
    
    textTestcorpus = cwTestCorpus
    testResult = generateEmbeddings(modelname=modelname, sentencelist=textTestcorpus, embeddingtype="word")
    testResult = np.array(testResult)
    
    fineTesttags = cwTestTags
    print(len(fineTesttags))
    print(type(fineTesttags[0]))

    totalTags = finetags + fineTesttags
    totalTags = np.array(totalTags)
    label_map = {cat:index for index,cat in enumerate(np.unique(totalTags))}
    y_prep = np.asarray([label_map[l] for l in finetags])
    y_prep_Test = np.asarray([label_map[l] for l in fineTesttags])
    
    print(label_map)

    x_train = result
    y_train = y_prep
    x_test = testResult
    y_test = y_prep_Test
    print("-------------------------------------THIS IS y_test--------------------------------------------")
    print(y_test)
    
    X_axis = []
    accuracy = []
    F1 = []
    precision = []
    recall = []
    print("\n\n\n-------------------------------EVALUATION SCORES FOR THE JOINT SYSTEM---------------------------------\n\n\n")

    print("---------------------------------CREATIVEWORKS - JOINT SYSTEM -----------------------")


 

    print("XGBoost\n")

    for i in range(100):
        XGB= XGBClassifier(n_estimators= i+1, random_state = 0)
        XGB.fit(x_train, y_train)

        X_axis.append(i+1)
        y_pred = XGB.predict(x_test)

        accuracy.append(accuracy_score(y_test,y_pred)*100)
        F1.append(f1_score(y_test, y_pred, average='macro', labels=np.unique(y_pred))*100)
        recall.append(recall_score(y_test, y_pred, average='macro', labels=np.unique(y_pred))*100)
        precision.append(precision_score(y_test, y_pred, average='macro', labels=np.unique(y_pred))*100)

        print(len(accuracy))
        print(len(F1))
        print(len(precision))
        print(len(recall))

    #ACCURACY 
    fig5 = plt.figure()
    plt.plot(X_axis, accuracy, marker='o')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Accuracy')
    plt.title('Accuracy v/s Number of Estimators')
    plt.show()
    fig5.savefig('eng-js-xgb-cw-acc.png')
    
    # #F1 SCORE
    fig6 = plt.figure()
    plt.plot(X_axis, F1, marker='o')
    plt.xlabel('Number of Estimators')
    plt.ylabel('F1 Score')
    plt.title('F1 Score v/s Number of Estimators')
    plt.show()
    fig6.savefig('eng-js-xgb-cw-f1.png')

    # #RECALL
    fig7 = plt.figure()
    plt.plot(X_axis, recall, marker='o')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Recall')
    plt.title('Recall v/s Number of Estimators')
    plt.show()
    fig7.savefig('eng-js-xgb-cw-recall.png')
    
    # #PRECISION
    fig8 = plt.figure()
    plt.plot(X_axis, precision, marker='o')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Precision')
    plt.title('Precision v/s Number of Estimators')
    plt.show()
    fig8.savefig('eng-js-xgb-cw-prec.png')
    
    print("---------------------------------------------------------------------------------------------------------------------------------------------\n")
    print("------------------------------------             GROUP DOMAIN             ------------------------------------------------------")
    
    groupcorpus, grouptags = DomainCorpusGenerator('/home/omkar/multiconer-baseline/eng_prediction.txt', "Group", "train")
    groupTestCorpus, groupTestTags = DomainCorpusGenerator('/home/omkar/multiconer-baseline/eng_test_prediction.txt', "Group", "test")
    
    
    textcorpus = groupcorpus
    result = generateEmbeddings(modelname=modelname, sentencelist=textcorpus, embeddingtype="word")
    result = np.array(result)

    finetags = grouptags
    
    textTestcorpus = groupTestCorpus
    testResult = generateEmbeddings(modelname=modelname, sentencelist=textTestcorpus, embeddingtype="word")
    testResult = np.array(testResult)
    
    fineTesttags = groupTestTags
    
    
    totalTags = finetags + fineTesttags
    totalTags = np.array(totalTags)    
    label_map = {cat:index for index,cat in enumerate(np.unique(totalTags))}
    y_prep = np.asarray([label_map[l] for l in finetags])
    y_prep_Test = np.asarray([label_map[l] for l in fineTesttags])
    
    print(label_map)

    x_train= result
    y_train = y_prep
    x_test = testResult
    y_test = y_prep_Test


    X_axis = []
    accuracy = []
    F1 = []
    precision = []
    recall = []
    print("\n\n\n-------------------------------EVALUATION SCORES FOR THE JOINT SYSTEM---------------------------------\n\n\n")

    print("---------------------------------GROUP - JOINT SYSTEM -----------------------")


 

    print("XGBoost\n")

    for i in range(100):
        XGB= XGBClassifier(n_estimators= i+1, random_state = 0)
        XGB.fit(x_train, y_train)

        X_axis.append(i+1)
        y_pred = XGB.predict(x_test)

        accuracy.append(accuracy_score(y_test,y_pred)*100)
        F1.append(f1_score(y_test, y_pred, average='macro', labels=np.unique(y_pred))*100)
        recall.append(recall_score(y_test, y_pred, average='macro', labels=np.unique(y_pred))*100)
        precision.append(precision_score(y_test, y_pred, average='macro', labels=np.unique(y_pred))*100)

        print(len(accuracy))
        print(len(F1))
        print(len(precision))
        print(len(recall))

    #ACCURACY 
    fig9 = plt.figure()
    plt.plot(X_axis, accuracy, marker='o')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Accuracy')
    plt.title('Accuracy v/s Number of Estimators')
    plt.show()
    fig9.savefig('eng-js-xgb-group-acc.png')

    #F1 SCORE
    fig10 = plt.figure()
    plt.plot(X_axis, F1, marker='o')
    plt.xlabel('Number of Estimators')
    plt.ylabel('F1 Score')
    plt.title('F1 Score v/s Number of Estimators')
    plt.show()
    fig10.savefig('eng-js-xgb-group-f1.png')
    
    #RECALL
    fig11 = plt.figure()
    plt.plot(X_axis, recall, marker='o')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Recall')
    plt.title('Recall v/s Number of Estimators')
    plt.show()
    fig11.savefig('eng-js-xgb-group-recall.png')

    #PRECISION
    fig12 = plt.figure()
    plt.plot(X_axis, precision, marker='o')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Precision')
    plt.title('Precision v/s Number of Estimators')
    plt.show()
    fig12.savefig('eng-js-xgb-group-prec.png')

    print("---------------------------------------------------------------------------------------------------------------------------------------------\n")
    print("------------------------------------             PRODUCT DOMAIN             ------------------------------------------------------")
    
    productcorpus, producttags = DomainCorpusGenerator('/home/omkar/multiconer-baseline/eng_prediction.txt', "Product", "train")
    productTestCorpus, productTestTags = DomainCorpusGenerator('/home/omkar/multiconer-baseline/eng_test_prediction.txt', "Product", "test")
    
    textcorpus = productcorpus
    result = generateEmbeddings(modelname=modelname, sentencelist=textcorpus, embeddingtype="word")
    result = np.array(result)

    finetags = producttags

    textTestcorpus = productTestCorpus
    testResult = generateEmbeddings(modelname=modelname, sentencelist=textTestcorpus, embeddingtype="word")
    testResult = np.array(testResult)
    
    fineTesttags = productTestTags
    
    totalTags = finetags + fineTesttags
    totalTags = np.array(totalTags)    
    label_map = {cat:index for index,cat in enumerate(np.unique(totalTags))}
    y_prep = np.asarray([label_map[l] for l in finetags])
    y_prep_Test = np.asarray([label_map[l] for l in fineTesttags])
    
    
    print(label_map)

    x_train = result
    y_train = y_prep
    x_test = testResult
    y_test = y_prep_Test

    X_axis = []
    accuracy = []
    F1 = []
    precision = []
    recall = []
    print("\n\n\n-------------------------------EVALUATION SCORES FOR THE JOINT SYSTEM---------------------------------\n\n\n")

    print("---------------------------------PRODUCT - JOINT SYSTEM -----------------------")


 
    print("XGBoost\n")

    for i in range(100):
        XGB= XGBClassifier(n_estimators= i+1, random_state = 0)
        XGB.fit(x_train, y_train)

        X_axis.append(i+1)
        y_pred = XGB.predict(x_test)

        accuracy.append(accuracy_score(y_test,y_pred)*100)
        F1.append(f1_score(y_test, y_pred, average='macro', labels=np.unique(y_pred))*100)
        recall.append(recall_score(y_test, y_pred, average='macro', labels=np.unique(y_pred))*100)
        precision.append(precision_score(y_test, y_pred, average='macro', labels=np.unique(y_pred))*100)

        print(len(accuracy))
        print(len(F1))
        print(len(precision))
        print(len(recall))

    #ACCURACY 
    fig13 = plt.figure()
    plt.plot(X_axis, accuracy, marker='o')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Accuracy')
    plt.title('Accuracy v/s Number of Estimators')
    plt.show()
    fig13.savefig('eng-js-xgb-product-acc.png')

    #F1 SCORE
    fig14 = plt.figure()
    plt.plot(X_axis, F1, marker='o')
    plt.xlabel('Number of Estimators')
    plt.ylabel('F1 Score')
    plt.title('F1 Score v/s Number of Estimators')
    plt.show()
    fig14.savefig('eng-js-xgb-product-f1.png')

    #RECALL
    fig15 = plt.figure()
    plt.plot(X_axis, recall, marker='o')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Recall')
    plt.title('Recall v/s Number of Estimators')
    plt.show()
    fig15.savefig('eng-js-xgb-product-recall.png')

    #PRECISION
    fig16 = plt.figure()
    plt.plot(X_axis, precision, marker='o')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Precision')
    plt.title('Precision v/s Number of Estimators')
    plt.show()
    fig16.savefig('eng-js-xgb-product-prec.png')

    print("---------------------------------------------------------------------------------------------------------------------------------------------\n")
    print("------------------------------------             LOCATION DOMAIN             ------------------------------------------------------")
    
    locationcorpus, locationtags = DomainCorpusGenerator('/home/omkar/multiconer-baseline/eng_prediction.txt', "Location", "train")

    textcorpus = locationcorpus
    result = generateEmbeddings(modelname=modelname, sentencelist=textcorpus, embeddingtype="word")
    result = np.array(result)

    finetags = locationtags    
    
    locationTestCorpus, locationTestTags = DomainCorpusGenerator('/home/omkar/multiconer-baseline/eng_test_prediction.txt', "Location", "test")
    
    textTestcorpus = locationTestCorpus
    testResult = generateEmbeddings(modelname=modelname, sentencelist=textTestcorpus, embeddingtype="word")
    testResult = np.array(testResult)
    
    fineTesttags = locationTestTags

    totalTags = finetags + fineTesttags 
    totalTags = np.array(totalTags)       
    label_map = {cat:index for index,cat in enumerate(np.unique(totalTags))}
    y_prep = np.asarray([label_map[l] for l in finetags])
    y_prep_Test = np.asarray([label_map[l] for l in fineTesttags])
    
    print(label_map)

    x_train = result
    y_train = y_prep
    x_test = testResult
    y_test = y_prep_Test

    X_axis = []
    accuracy = []
    F1 = []
    precision = []
    recall = []
    print("\n\n\n-------------------------------EVALUATION SCORES FOR THE JOINT SYSTEM---------------------------------\n\n\n")

    print("---------------------------------LOCATION - JOINT SYSTEM -----------------------")


 
    print("XGBoost\n")

    for i in range(100):
        XGB= XGBClassifier(n_estimators= i+1, random_state = 0)
        XGB.fit(x_train, y_train)

        X_axis.append(i+1)
        y_pred = XGB.predict(x_test)

        accuracy.append(accuracy_score(y_test,y_pred)*100)
        F1.append(f1_score(y_test, y_pred, average='macro', labels=np.unique(y_pred))*100)
        recall.append(recall_score(y_test, y_pred, average='macro', labels=np.unique(y_pred))*100)
        precision.append(precision_score(y_test, y_pred, average='macro', labels=np.unique(y_pred))*100)

        print(len(accuracy))
        print(len(F1))
        print(len(precision))
        print(len(recall))

    #ACCURACY 
    fig17 = plt.figure()
    plt.plot(X_axis, accuracy, marker='o')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Accuracy')
    plt.title('Accuracy v/s Number of Estimators')
    plt.show()
    fig17.savefig('eng-js-xgb-location-acc.png')

    # #F1 SCORE
    fig18 = plt.figure()
    plt.plot(X_axis, F1, marker='o')
    plt.xlabel('Number of Estimators')
    plt.ylabel('F1 Score')
    plt.title('F1 Score v/s Number of Estimators')
    plt.show()
    fig18.savefig('eng-js-xgb-location-f1.png')

    # #RECALL
    fig19 = plt.figure()
    plt.plot(X_axis, recall, marker='o')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Recall')
    plt.title('Recall v/s Number of Estimators')
    plt.show()
    fig19.savefig('eng-js-xgb-location-recall.png')

    # #PRECISION
    fig20 = plt.figure()
    plt.plot(X_axis, precision, marker='o')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Precision')
    plt.title('Precision v/s Number of Estimators')
    plt.show()
    fig20.savefig('eng-js-xgb-location-prec.png')


 
    print("---------------------------------------------------------------------------------------------------------------------------------------------\n")
    print("------------------------------------             MEDICAL DOMAIN             ------------------------------------------------------")
    
    medicalcorpus, medicaltags = DomainCorpusGenerator('/home/omkar/multiconer-baseline/eng_prediction.txt', "Medical", "train")
    medicalTestCorpus, medicalTestTags = DomainCorpusGenerator('/home/omkar/multiconer-baseline/eng_test_prediction.txt', "Medical", "test")

    textcorpus = medicalcorpus
    result = generateEmbeddings(modelname=modelname, sentencelist=textcorpus, embeddingtype="word")
    result = np.array(result)

    finetags = medicaltags

    textTestcorpus = medicalTestCorpus
    testResult = generateEmbeddings(modelname=modelname, sentencelist=textTestcorpus, embeddingtype="word")
    testResult = np.array(testResult)
    
    fineTesttags = medicalTestTags

    totalTags = finetags + fineTesttags
    totalTags = np.array(totalTags)           
    label_map = {cat:index for index,cat in enumerate(np.unique(totalTags))}
    y_prep = np.asarray([label_map[l] for l in finetags])
    y_prep_Test = np.asarray([label_map[l] for l in fineTesttags])
    
    print(label_map)

    x_train = result
    y_train = y_prep
    x_test = testResult
    y_test = y_prep_Test

    print("---------------------------------MEDICAL - JOINT SYSTEM -----------------------")


 
    X_axis = []
    accuracy = []
    F1 = []
    precision = []
    recall = []
    print("\n\n\n-------------------------------EVALUATION SCORES FOR THE JOINT SYSTEM---------------------------------\n\n\n")

    print("XGBoost\n")

    for i in range(100):
        XGB= XGBClassifier(n_estimators= i+1, random_state = 0)
        XGB.fit(x_train, y_train)

        X_axis.append(i+1)
        y_pred = XGB.predict(x_test)

        accuracy.append(accuracy_score(y_test,y_pred)*100)
        F1.append(f1_score(y_test, y_pred, average='macro', labels=np.unique(y_pred))*100)
        recall.append(recall_score(y_test, y_pred, average='macro', labels=np.unique(y_pred))*100)
        precision.append(precision_score(y_test, y_pred, average='macro', labels=np.unique(y_pred))*100)

        print(len(accuracy))
        print(len(F1))
        print(len(precision))
        print(len(recall))

    #ACCURACY 
    fig100 = plt.figure()
    plt.plot(X_axis, accuracy, marker='o')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Accuracy')
    plt.title('Accuracy v/s Number of Estimators')
    plt.show()
    fig100.savefig('eng-js-xgb-medical-acc.png')

    #F1 SCORE
    fig101 = plt.figure()
    plt.plot(X_axis, F1, marker='o')
    plt.xlabel('Number of Estimators')
    plt.ylabel('F1 Score')
    plt.title('F1 Score v/s Number of Estimators')
    plt.show()
    fig101.savefig('eng-js-xgb-medical-f1.png')

    #RECALL
    fig102 = plt.figure()
    plt.plot(X_axis, recall, marker='o')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Recall')
    plt.title('Recall v/s Number of Estimators')
    plt.show()
    fig102.savefig('eng-js-xgb-medical-recall.png')

    #PRECISION
    fig103 = plt.figure()
    plt.plot(X_axis, precision, marker='o')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Precision')
    plt.title('Precision v/s Number of Estimators')
    plt.show()
    fig103.savefig('eng-js-xgb-medical-prec.png')

        
    print("\n\n----------------------------------------UPPER BOUND MODEL-----------------------------------------------------------------\n\n")
    print("\n\n----------------------------------------UPPER BOUND MODEL-----------------------------------------------------------------\n\n")


    print("---------------------------------------------------------------------------------------------------------------------------------------------\n")
    print("------------------------------------             PERSON DOMAIN             ------------------------------------------------------")
    
    originalpersoncorpus, originalpersontags = OriginalDomainCorpusGenerator('/home/omkar/multiconer-baseline/coarse-en-train90.conll', "Person", "train")
    originalPersonTestCorpus, originalPersonTestTags = OriginalDomainCorpusGenerator('/home/omkar/multiconer-baseline/coarse-en-dev.conll', "Person", "test")
    
    
    textcorpus = originalpersoncorpus
    result = generateEmbeddings(modelname=modelname, sentencelist=textcorpus, embeddingtype="word")
    result = np.array(result)
    print(len(result))

    finetags = originalpersontags

    textTestcorpus = originalPersonTestCorpus
    testResult = generateEmbeddings(modelname=modelname, sentencelist=textTestcorpus, embeddingtype="word")
    testResult = np.array(testResult)
    
    fineTesttags = originalPersonTestTags

    totalTags = finetags + fineTesttags
    totalTags = np.array(totalTags)               
    label_map = {cat:index for index,cat in enumerate(np.unique(totalTags))}
    y_prep = np.asarray([label_map[l] for l in finetags])
    y_prep_Test = np.asarray([label_map[l] for l in fineTesttags])
    
    
    x_train = result
    y_train = y_prep
    x_test = testResult
    y_test = y_prep_Test

    X_axis_bm = []
    accuracy_bm = []
    F1_bm = []
    precision_bm = []
    recall_bm = []
    
    print("\n\n\n-------------------------------EVALUATION SCORES FOR THE UPPER BOUND MODELS -------------------------------\n\n\n")

    print("---------------------------------PERSON - UPPER BOUND -----------------------")


    # #Decision Trees
    # print("\n\nDecision Trees\n")
    # DT = DecisionTreeClassifier(criterion='entropy', random_state=0)
    # DT.fit(x_train, y_train)

    # score = DT.score(x_test,y_test)
    # print('Decision Trees Accuracy: ', "%.2f" % (score*100))

    # y_pred = DT.predict(x_test)
    # cm = confusion_matrix(y_test, y_pred)
    # print(cm)
    # f1 = f1_score(y_test, y_pred, average='weighted')
    # print('Decision Trees F1: ', "%.2f" % (f1*100))

    # print("\n\nRBF SVM\n")
    # rbfSVM = SVC(kernel = 'rbf', probability = True, random_state = 0)
    # rbfSVM.fit(x_train,y_train)

    # score = rbfSVM.score(x_test,y_test)
    # print('RBF SVM Accuracy: ', "%.2f" % (score*100))

    # y_pred = rbfSVM.predict(x_test)
    # cm = confusion_matrix(y_test, y_pred)
    # print(cm)
    # f1 = f1_score(y_test, y_pred, average='weighted')
    # print('RBF SVM F1: ', "%.2f" % (f1*100))

    # print("\n\nPolySVM\n")
    # polySVM = SVC( kernel = 'poly', probability = True, random_state = 0)
    # polySVM.fit(x_train,y_train)

    # score = polySVM.score(x_test,y_test)
    # print('PolySVM Accuracy: ', "%.2f" % (score*100))

    # y_pred = polySVM.predict(x_test)
    # cm = confusion_matrix(y_test, y_pred)
    # print(cm)
    # f1 = f1_score(y_test, y_pred, average='weighted')
    # print('PolySVM F1: ', "%.2f" % (f1*100))    


    print("\n\n\n-------------------------------EVALUATION SCORES FOR THE UPPER BOUND MODELS -------------------------------\n\n\n")

    print("XGBOOST\n")
    for i in range(100):
        XGB= XGBClassifier(n_estimators= i+1, random_state = 0)
        XGB.fit(x_train, y_train)

        X_axis_bm.append(i+1)
        y_pred = XGB.predict(x_test)

        accuracy_bm.append(accuracy_score(y_test,y_pred)*100)
        F1_bm.append(f1_score(y_test, y_pred, average='macro', labels=np.unique(y_pred))*100)
        recall_bm.append(recall_score(y_test, y_pred, average='macro', labels=np.unique(y_pred))*100)
        precision_bm.append(precision_score(y_test, y_pred, average='macro', labels=np.unique(y_pred))*100)

        print(len(accuracy_bm))
        print(len(F1_bm))
        print(len(precision_bm))
        print(len(recall_bm))

    
    #ACCURACY_bm 
    fig21 = plt.figure()
    plt.plot(X_axis_bm, accuracy_bm, marker='o')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Accuracy')
    plt.title('Accuracy v/s Number of Estimators (Upper Bound Models)')
    plt.show()
    fig21.savefig('eng-ub-xgb-person-acc.png')

    #F1_bm SCORE 
    fig22 = plt.figure()
    plt.plot(X_axis_bm, F1_bm, marker='o')
    plt.xlabel('Number of Estimators')
    plt.ylabel('F1 Score')
    plt.title('F1 Score v/s Number of Estimators (Upper Bound Models)')
    plt.show()
    fig22.savefig('eng-ub-xgb-person-f1.png')

    #RECALL 
    fig23 = plt.figure()
    plt.plot(X_axis_bm, recall_bm, marker='o')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Recall')
    plt.title('Recall v/s Number of Estimators (Upper Bound Models)')
    plt.show()
    fig23.savefig('eng-ub-xgb-person-recall.png')

    #PRECISION 
    fig24 = plt.figure()
    plt.plot(X_axis_bm, precision_bm, marker='o')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Precision')
    plt.title('Precision v/s Number of Estimators (Upper Bound Models)')
    plt.show()
    fig24.savefig('eng-ub-xgb-person-prec.png')

    print("---------------------------------------------------------------------------------------------------------------------------------------------\n")
    print("------------------------------------             GROUP DOMAIN             ------------------------------------------------------")
    
    originalgroupcorpus, originalgrouptags = OriginalDomainCorpusGenerator('/home/omkar/multiconer-baseline/coarse-en-train90.conll', "Group", "train")
    originalGroupTestCorpus, originalGroupTestTags = OriginalDomainCorpusGenerator('/home/omkar/multiconer-baseline/coarse-en-dev.conll', "Group", "test")
    
    textcorpus = originalgroupcorpus
    result = generateEmbeddings(modelname=modelname, sentencelist=textcorpus, embeddingtype="word")
    result = np.array(result)
    print(len(result))

    finetags = originalgrouptags
    
    textTestcorpus = originalGroupTestCorpus
    testResult = generateEmbeddings(modelname=modelname, sentencelist=textTestcorpus, embeddingtype="word")
    testResult = np.array(testResult)
    
    fineTesttags = originalGroupTestTags

    totalTags = finetags + fineTesttags
    totalTags = np.array(totalTags)                   
    label_map = {cat:index for index,cat in enumerate(np.unique(totalTags))}
    y_prep = np.asarray([label_map[l] for l in finetags])
    y_prep_Test = np.asarray([label_map[l] for l in fineTesttags])
    
    
    x_train = result
    y_train = y_prep
    x_test = testResult
    y_test = y_prep_Test

    X_axis_bm = []
    accuracy_bm = []
    F1_bm = []
    precision_bm = []
    recall_bm = []




    print("\n\n\n-------------------------------EVALUATION SCORES FOR THE UPPER BOUND MODELS -------------------------------\n\n\n")

    print("---------------------------------GROUP - UPPER BOUND -----------------------")


     
    print("XGBOOST\n")
    for i in range(100):
        XGB= XGBClassifier(n_estimators= i+1, random_state = 0)
        XGB.fit(x_train, y_train)

        X_axis_bm.append(i+1)
        y_pred = XGB.predict(x_test)

        accuracy_bm.append(accuracy_score(y_test,y_pred)*100)
        F1_bm.append(f1_score(y_test, y_pred, average='macro', labels=np.unique(y_pred))*100)
        recall_bm.append(recall_score(y_test, y_pred, average='macro', labels=np.unique(y_pred))*100)
        precision_bm.append(precision_score(y_test, y_pred, average='macro', labels=np.unique(y_pred))*100)

        print(len(accuracy_bm))
        print(len(F1_bm))
        print(len(precision_bm))
        print(len(recall_bm))

    
    #ACCURACY_bm 
    fig25 = plt.figure()
    plt.plot(X_axis_bm, accuracy_bm, marker='o')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Accuracy')
    plt.title('Accuracy v/s Number of Estimators (Upper Bound Models)')
    plt.show()
    fig25.savefig('eng-ub-xgb-group-acc.png')

    #F1_bm SCORE 
    fig26 = plt.figure()
    plt.plot(X_axis_bm, F1_bm, marker='o')
    plt.xlabel('Number of Estimators')
    plt.ylabel('F1 Score')
    plt.title('F1 Score v/s Number of Estimators (Upper Bound Models)')
    plt.show()
    fig26.savefig('eng-ub-xgb-group-f1.png')

    #RECALL 
    fig27 = plt.figure()
    plt.plot(X_axis_bm, recall_bm, marker='o')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Recall')
    plt.title('Recall v/s Number of Estimators (Upper Bound Models)')
    plt.show()
    fig27.savefig('eng-ub-xgb-group-recall.png')

    #PRECISION 
    fig28 = plt.figure()
    plt.plot(X_axis_bm, precision_bm, marker='o')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Precision')
    plt.title('Precision v/s Number of Estimators (Upper Bound Models)')
    plt.show()
    fig28.savefig('eng-ub-xgb-group-prec.png')

    print("---------------------------------------------------------------------------------------------------------------------------------------------\n")
    print("------------------------------------             CREATIVE WORKS DOMAIN             ------------------------------------------------------")
    
    originalcwcorpus, originalcwtags = OriginalDomainCorpusGenerator('/home/omkar/multiconer-baseline/coarse-en-train90.conll', "CreativeWorks", "train")
    originalcwTestCorpus, originalcwTestTags = OriginalDomainCorpusGenerator('/home/omkar/multiconer-baseline/coarse-en-dev.conll', "CreativeWorks", "test")    
    
    textcorpus = originalcwcorpus
    result = generateEmbeddings(modelname=modelname, sentencelist=textcorpus, embeddingtype="word")
    result = np.array(result)
    print(len(result))

    finetags = originalcwtags

    textTestcorpus = originalcwTestCorpus
    testResult = generateEmbeddings(modelname=modelname, sentencelist=textTestcorpus, embeddingtype="word")
    testResult = np.array(testResult)
    
    fineTesttags = originalcwTestTags

    totalTags = finetags + fineTesttags
    totalTags = np.array(totalTags)                       
    label_map = {cat:index for index,cat in enumerate(np.unique(totalTags))}
    y_prep = np.asarray([label_map[l] for l in finetags])
    y_prep_Test = np.asarray([label_map[l] for l in fineTesttags])
    
    
    x_train = result
    y_train = y_prep
    x_test = testResult
    y_test = y_prep_Test


    X_axis_bm = []
    accuracy_bm = []
    F1_bm = []
    precision_bm = []
    recall_bm = []




    print("\n\n\n-------------------------------EVALUATION SCORES FOR THE UPPER BOUND MODELS -------------------------------\n\n\n")

    print("---------------------------------CREATIVE WORKS - UPPERBOUND -----------------------")


 
    print("XGBOOST\n")
    for i in range(100):
        XGB= XGBClassifier(n_estimators= i+1, random_state = 0)
        XGB.fit(x_train, y_train)

        X_axis_bm.append(i+1)
        y_pred = XGB.predict(x_test)

        accuracy_bm.append(accuracy_score(y_test,y_pred)*100)
        F1_bm.append(f1_score(y_test, y_pred, average='macro', labels=np.unique(y_pred))*100)
        recall_bm.append(recall_score(y_test, y_pred, average='macro', labels=np.unique(y_pred))*100)
        precision_bm.append(precision_score(y_test, y_pred, average='macro', labels=np.unique(y_pred))*100)

        print(len(accuracy_bm))
        print(len(F1_bm))
        print(len(precision_bm))
        print(len(recall_bm))

    
    #ACCURACY_bm 
    fig29 = plt.figure()
    plt.plot(X_axis_bm, accuracy_bm, marker='o')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Accuracy')
    plt.title('Accuracy v/s Number of Estimators (Upper Bound Models)')
    plt.show()
    fig29.savefig('eng-ub-xgb-cw-acc.png')

    #F1_bm SCORE 
    fig30 = plt.figure()
    plt.plot(X_axis_bm, F1_bm, marker='o')
    plt.xlabel('Number of Estimators')
    plt.ylabel('F1 Score')
    plt.title('F1 Score v/s Number of Estimators (Upper Bound Models)')
    plt.show()
    fig30.savefig('eng-ub-xgb-cw-f1.png')

    #RECALL 
    fig31 = plt.figure()
    plt.plot(X_axis_bm, recall_bm, marker='o')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Recall')
    plt.title('Recall v/s Number of Estimators (Upper Bound Models)')
    plt.show()
    fig31.savefig('eng-ub-xgb-cw-recall.png')

    #PRECISION 
    fig32 = plt.figure()
    plt.plot(X_axis_bm, precision_bm, marker='o')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Precision')
    plt.title('Precision v/s Number of Estimators (Upper Bound Models)')
    plt.show()
    fig32.savefig('eng-ub-xgb-cw-prec.png')
 
    print("---------------------------------------------------------------------------------------------------------------------------------------------\n")
    print("------------------------------------             LOCATION DOMAIN             ------------------------------------------------------")
    
    originallocationcorpus, originallocationtags = OriginalDomainCorpusGenerator('/home/omkar/multiconer-baseline/coarse-en-train90.conll', "Location", "train")
    originalLocationTestCorpus, originalLocationTestTags = OriginalDomainCorpusGenerator('/home/omkar/multiconer-baseline/coarse-en-dev.conll', "Location", "test")
    
    
    textcorpus = originallocationcorpus
    result = generateEmbeddings(modelname=modelname, sentencelist=textcorpus, embeddingtype="word")
    result = np.array(result)
    print(len(result))

    finetags = originallocationtags
    
    textTestcorpus = originalLocationTestCorpus
    testResult = generateEmbeddings(modelname=modelname, sentencelist=textTestcorpus, embeddingtype="word")
    testResult = np.array(testResult)
    
    fineTesttags = originalLocationTestTags

    totalTags = finetags + fineTesttags
    totalTags = np.array(totalTags)                           
    label_map = {cat:index for index,cat in enumerate(np.unique(totalTags))}
    y_prep = np.asarray([label_map[l] for l in finetags])
    y_prep_Test = np.asarray([label_map[l] for l in fineTesttags])

    x_train = result
    y_train = y_prep
    x_test = testResult
    y_test = y_prep_Test


    X_axis_bm = []
    accuracy_bm = []
    F1_bm = []
    precision_bm = []
    recall_bm = []




    print("\n\n\n-------------------------------EVALUATION SCORES FOR THE UPPER BOUND MODELS -------------------------------\n\n\n")

    print("---------------------------------LOCATION - UPPER BOUND -----------------------")


 
    print("XGBOOST\n")
    for i in range(100):
        XGB= XGBClassifier(n_estimators= i+1, random_state = 0)
        XGB.fit(x_train, y_train)

        X_axis_bm.append(i+1)
        y_pred = XGB.predict(x_test)

        accuracy_bm.append(accuracy_score(y_test,y_pred)*100)
        F1_bm.append(f1_score(y_test, y_pred, average='macro', labels=np.unique(y_pred))*100)
        recall_bm.append(recall_score(y_test, y_pred, average='macro', labels=np.unique(y_pred))*100)
        precision_bm.append(precision_score(y_test, y_pred, average='macro', labels=np.unique(y_pred))*100)

        print(len(accuracy_bm))
        print(len(F1_bm))
        print(len(precision_bm))
        print(len(recall_bm))

    
    #ACCURACY_bm 
    fig33 = plt.figure()
    plt.plot(X_axis_bm, accuracy_bm, marker='o')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Accuracy')
    plt.title('Accuracy v/s Number of Estimators (Upper Bound Models)')
    plt.show()
    fig33.savefig('eng-ub-xgb-location-acc.png')
    
    #F1_bm SCORE 
    fig34 = plt.figure()
    plt.plot(X_axis_bm, F1_bm, marker='o')
    plt.xlabel('Number of Estimators')
    plt.ylabel('F1 Score')
    plt.title('F1 Score v/s Number of Estimators (Upper Bound Models)')
    plt.show()
    fig34.savefig('eng-ub-xgb-location-f1.png')

    #RECALL 
    fig35 = plt.figure()
    plt.plot(X_axis_bm, recall_bm, marker='o')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Recall')
    plt.title('Recall v/s Number of Estimators (Upper Bound Models)')
    plt.show()
    fig35.savefig('eng-ub-xgb-location-recall.png')

    #PRECISION 
    fig36 = plt.figure()
    plt.plot(X_axis_bm, precision_bm, marker='o')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Precision')
    plt.title('Precision v/s Number of Estimators (Upper Bound Models)')
    plt.show()
    fig36.savefig('eng-ub-xgb-location-prec.png')

    print("---------------------------------------------------------------------------------------------------------------------------------------------\n")
    print("------------------------------------             PRODUCT DOMAIN             ------------------------------------------------------")
    
    originalproductcorpus, originalproducttags = OriginalDomainCorpusGenerator('/home/omkar/multiconer-baseline/coarse-en-train90.conll', "Product", "train")
    originalProductTestCorpus, originalProductTestTags = OriginalDomainCorpusGenerator('/home/omkar/multiconer-baseline/coarse-en-dev.conll', "Product", "test")
    
    textcorpus = originalproductcorpus
    result = generateEmbeddings(modelname=modelname, sentencelist=textcorpus, embeddingtype="word")
    result = np.array(result)
    print(len(result))

    finetags = originalproducttags
    
    textTestcorpus = originalProductTestCorpus
    testResult = generateEmbeddings(modelname=modelname, sentencelist=textTestcorpus, embeddingtype="word")
    testResult = np.array(testResult)
    
    fineTesttags = originalProductTestTags
    
    totalTags = finetags + fineTesttags
    totalTags = np.array(totalTags)                               
    label_map = {cat:index for index,cat in enumerate(np.unique(totalTags))}
    y_prep = np.asarray([label_map[l] for l in finetags])
    y_prep_Test = np.asarray([label_map[l] for l in fineTesttags])
    
    
    x_train = result
    y_train = y_prep
    x_test = testResult
    y_test = y_prep_Test

    X_axis_bm = []
    accuracy_bm = []
    F1_bm = []
    precision_bm = []
    recall_bm = []




    print("\n\n\n-------------------------------EVALUATION SCORES FOR THE UPPER BOUND MODELS -------------------------------\n\n\n")

    print("---------------------------------GROUP - UPPER BOUND -----------------------")


 

    print("XGBOOST\n")
    for i in range(100):
        XGB= XGBClassifier(n_estimators= i+1, random_state = 0)
        XGB.fit(x_train, y_train)

        X_axis_bm.append(i+1)
        y_pred = XGB.predict(x_test)

        accuracy_bm.append(accuracy_score(y_test,y_pred)*100)
        F1_bm.append(f1_score(y_test, y_pred, average='macro', labels=np.unique(y_pred))*100)
        recall_bm.append(recall_score(y_test, y_pred, average='macro', labels=np.unique(y_pred))*100)
        precision_bm.append(precision_score(y_test, y_pred, average='macro', labels=np.unique(y_pred))*100)

        print(len(accuracy_bm))
        print(len(F1_bm))
        print(len(precision_bm))
        print(len(recall_bm))

    
    #ACCURACY_bm 
    fig37 = plt.figure()
    plt.plot(X_axis_bm, accuracy_bm, marker='o')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Accuracy')
    plt.title('Accuracy v/s Number of Estimators (Upper Bound Models)')
    plt.show()
    fig37.savefig('eng-ub-xgb-group-acc.png')

    #F1_bm SCORE 
    fig38 = plt.figure()
    plt.plot(X_axis_bm, F1_bm, marker='o')
    plt.xlabel('Number of Estimators')
    plt.ylabel('F1 Score')
    plt.title('F1 Score v/s Number of Estimators (Upper Bound Models)')
    plt.show()
    fig38.savefig('eng-ub-xgb-group-f1.png')

    #RECALL 
    fig39 = plt.figure()
    plt.plot(X_axis_bm, recall_bm, marker='o')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Recall')
    plt.title('Recall v/s Number of Estimators (Upper Bound Models)')
    plt.show()
    fig39.savefig('eng-ub-xgb-group-recall.png')

    #PRECISION 
    fig40 = plt.figure()
    plt.plot(X_axis_bm, precision_bm, marker='o')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Precision')
    plt.title('Precision v/s Number of Estimators (Upper Bound Models)')
    plt.show()
    fig40.savefig('eng-ub-xgb-group-prec.png')


    # #Decision Trees
    # print("\n\nDecision Trees\n")
    # DT = DecisionTreeClassifier(criterion='entropy', random_state=0)
    # DT.fit(x_train, y_train)

    # score = DT.score(x_test,y_test)
    # print('Decision Trees Accuracy: ', "%.2f" % (score*100))

    # y_pred = DT.predict(x_test)
    # cm = confusion_matrix(y_test, y_pred)
    # print(cm)
    # f1 = f1_score(y_test, y_pred, average='weighted')
    # print('Decision Trees F1: ', "%.2f" % (f1*100))


    # print("\n\nRBF SVM\n")
    # rbfSVM = SVC(C= 1000000000, kernel = 'rbf', probability = True, random_state = 0)
    # rbfSVM.fit(x_train,y_train)

    # score = rbfSVM.score(x_test,y_test)
    # print('RBF SVM Accuracy: ', "%.2f" % (score*100))

    # y_pred = rbfSVM.predict(x_test)
    # cm = confusion_matrix(y_test, y_pred)
    # print(cm)
    # f1 = f1_score(y_test, y_pred, average='weighted')
    # print('RBF SVM F1: ', "%.2f" % (f1*100))

    # print("\n\nPolySVM\n")
    # polySVM = SVC(C= 1000000000, kernel = 'poly', probability = True, random_state = 0)
    # polySVM.fit(x_train,y_train)

    # score = polySVM.score(x_test,y_test)
    # print('PolySVM Accuracy: ', "%.2f" % (score*100))

    # y_pred = polySVM.predict(x_test)
    # cm = confusion_matrix(y_test, y_pred)
    # print(cm)
    # f1 = f1_score(y_test, y_pred, average='weighted')
    # print('PolySVM F1: ', "%.2f" % (f1*100))

    print("---------------------------------------------------------------------------------------------------------------------------------------------\n")
    print("------------------------------------             MEDICAL DOMAIN             ------------------------------------------------------")
    
    originalmedicalcorpus, originalmedicaltags = OriginalDomainCorpusGenerator('/home/omkar/multiconer-baseline/coarse-en-train90.conll', "Medical", "train")
    originalMedicalTestCorpus, originalMedicalTestTags = OriginalDomainCorpusGenerator('/home/omkar/multiconer-baseline/coarse-en-dev.conll', "Medical", "test")
    
    
    textcorpus = originalmedicalcorpus
    result = generateEmbeddings(modelname=modelname, sentencelist=textcorpus, embeddingtype="word")
    result = np.array(result)
    print(len(result))

    finetags = originalmedicaltags
    
    textTestcorpus = originalMedicalTestCorpus
    testResult = generateEmbeddings(modelname=modelname, sentencelist=textTestcorpus, embeddingtype="word")
    testResult = np.array(testResult)
    
    fineTesttags = originalMedicalTestTags

    totalTags = finetags + fineTesttags
    totalTags = np.array(totalTags)                                   
    label_map = {cat:index for index,cat in enumerate(np.unique(totalTags))}
    y_prep = np.asarray([label_map[l] for l in finetags])
    y_prep_Test = np.asarray([label_map[l] for l in fineTesttags])
    
    
    x_train = result
    y_train = y_prep    
    x_test = testResult
    y_test = y_prep_Test


    X_axis_bm = []
    accuracy_bm = []
    F1_bm = []
    precision_bm = []
    recall_bm = []
    
    print("\n\n\n-------------------------------EVALUATION SCORES FOR THE UPPER BOUND MODELS -------------------------------\n\n\n")

    print("---------------------------------MEDICAL - UPPER BOUND -----------------------")


    # #Decision Trees
    # print("\n\nDecision Trees\n")
    # DT = DecisionTreeClassifier(criterion='entropy', random_state=0)
    # DT.fit(x_train, y_train)

    # score = DT.score(x_test,y_test)
    # print('Decision Trees Accuracy: ', "%.2f" % (score*100))

    # y_pred = DT.predict(x_test)
    # cm = confusion_matrix(y_test, y_pred)
    # print(cm)
    # f1 = f1_score(y_test, y_pred, average='weighted')
    # print('Decision Trees F1: ', "%.2f" % (f1*100))

    # print("\n\nRBF SVM\n")
    # rbfSVM = SVC(kernel = 'rbf', probability = True, random_state = 0)
    # rbfSVM.fit(x_train,y_train)

    # score = rbfSVM.score(x_test,y_test)
    # print('RBF SVM Accuracy: ', "%.2f" % (score*100))

    # y_pred = rbfSVM.predict(x_test)
    # cm = confusion_matrix(y_test, y_pred)
    # print(cm)
    # f1 = f1_score(y_test, y_pred, average='weighted')
    # print('RBF SVM F1: ', "%.2f" % (f1*100))

    # print("\n\nPolySVM\n")
    # polySVM = SVC( kernel = 'poly', probability = True, random_state = 0)
    # polySVM.fit(x_train,y_train)

    # score = polySVM.score(x_test,y_test)
    # print('PolySVM Accuracy: ', "%.2f" % (score*100))

    # y_pred = polySVM.predict(x_test)
    # cm = confusion_matrix(y_test, y_pred)
    # print(cm)
    # f1 = f1_score(y_test, y_pred, average='weighted')
    # print('PolySVM F1: ', "%.2f" % (f1*100))    


    print("\n\n\n-------------------------------EVALUATION SCORES FOR THE UPPER BOUND MODELS -------------------------------\n\n\n")

    print("XGBOOST\n")
    for i in range(100):
        XGB= XGBClassifier(n_estimators= i+1, random_state = 0)
        XGB.fit(x_train, y_train)

        X_axis_bm.append(i+1)
        y_pred = XGB.predict(x_test)

        accuracy_bm.append(accuracy_score(y_test,y_pred)*100)
        F1_bm.append(f1_score(y_test, y_pred, average='macro', labels=np.unique(y_pred))*100)
        recall_bm.append(recall_score(y_test, y_pred, average='macro', labels=np.unique(y_pred))*100)
        precision_bm.append(precision_score(y_test, y_pred, average='macro', labels=np.unique(y_pred))*100)

        print(len(accuracy_bm))
        print(len(F1_bm))
        print(len(precision_bm))
        print(len(recall_bm))

    
    #ACCURACY_bm 
    fig200 = plt.figure()
    plt.plot(X_axis_bm, accuracy_bm, marker='o')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Accuracy')
    plt.title('Accuracy v/s Number of Estimators (Upper Bound Models)')
    plt.show()
    fig200.savefig('eng-ub-xgb-medical-acc.png')

    #F1_bm SCORE 
    fig201 = plt.figure()
    plt.plot(X_axis_bm, F1_bm, marker='o')
    plt.xlabel('Number of Estimators')
    plt.ylabel('F1 Score')
    plt.title('F1 Score v/s Number of Estimators (Upper Bound Models)')
    plt.show()
    fig201.savefig('eng-ub-xgb-medical-f1.png')

    #RECALL 
    fig202 = plt.figure()
    plt.plot(X_axis_bm, recall_bm, marker='o')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Recall')
    plt.title('Recall v/s Number of Estimators (Upper Bound Models)')
    plt.show()
    fig202.savefig('eng-ub-xgb-medical-recall.png')

    #PRECISION 
    fig203 = plt.figure()
    plt.plot(X_axis_bm, precision_bm, marker='o')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Precision')
    plt.title('Precision v/s Number of Estimators (Upper Bound Models)')
    plt.show()
    fig203.savefig('eng-ub-xgb-medical-prec.png')

if __name__ == "__main__":
    main()
    print("Script complete")