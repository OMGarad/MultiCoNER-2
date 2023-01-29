
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel, RobertaModel
import torch
import joblib
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
            if fineGrainedDomainTags[k]== 'MusicalGRP' or fineGrainedDomainTags[k]=='PublicCorp' or fineGrainedDomainTags[k]=='PrivateCorp' or fineGrainedDomainTags[k]=='OtherCORP' or fineGrainedDomainTags[k]=='SportsGRP' or fineGrainedDomainTags[k]=='TechCORP' or fineGrainedDomainTags[k]=='ORG' or fineGrainedDomainTags[k]=='CarManufacturer' or fineGrainedDomainTags[k]=='AerospaceManufacturer':
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
            print(domWord)
            print(len(domWord))
    
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
        trainingTags = FineGrainedTags('/home/omkar/multiconer-baseline/shuff-hinglish-train90.conll')
    elif test_or_train == 'test':
        trainingTags = FineGrainedTags('/home/omkar/multiconer-baseline/shuff-hinglish-dev.conll')
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

        
    

def main():

    modelname = "/home/omkar/coarse_ner_baseline_models/XLMR_Base/xlmr_base_eng_coarse_ner_e10/lightning_logs/version_0/checkpoints"

    print("\n\n----------------------------------------JOINT SYSTEM-----------------------------------------------------------------\n\n")
    print("\n\n----------------------------------------JOINT SYSTEM-----------------------------------------------------------------\n\n")

    print("---------------------------------------------------------------------------------------------------------------------------------------------\n")
    print("------------------------------------             PERSON DOMAIN             ------------------------------------------------------")
    
    personTrainCorpus, personTrainTags = DomainCorpusGenerator('/home/omkar/multiconer-baseline/eng_prediction.txt', domain = "Person", test_or_train= 'train')
    personTestCorpus, personTestTags = DomainCorpusGenerator('/home/omkar/multiconer-baseline/eng_test_prediction.txt', domain = "Person", test_or_train= 'test')
    
    
    #Testing Set
    
    #testResult = generateEmbeddings(modelname=modelname, sentencelist=textTestcorpus, embeddingtype="word")
    #testResult = np.array(testResult)
    
    
    # print(len(fineTesttags))
    # print(type(fineTesttags[0]))
    
    #Training Set
    textcorpus = personTrainCorpus
    textTestcorpus = personTestCorpus
    
    result = generateEmbeddings(modelname=modelname, sentencelist=textcorpus+textTestcorpus, embeddingtype="word")
    result = np.array(result)

    finetags = personTrainTags
    fineTesttags = personTestTags
    # print(len(finetags))
    # print(type(finetags[0]))

    totalTags = finetags + fineTesttags
    totalTags = np.array(totalTags)
    person_map_label = {}
    person_map_label['OtherPER'] = 0
    person_map_label['SportsManager'] = 1
    person_map_label['Cleric'] = 2
    person_map_label['Politician'] = 3
    person_map_label['Athlete'] = 4
    person_map_label['Artist'] = 5
    person_map_label['Scientist'] = 6
    person_map_label['Wrong'] = 7
    y_prep = np.asarray([person_map_label[l] for l in totalTags ]) #finetags
    #y_prep_Test = np.asarray([person_map_label[l] for l in fineTesttags])

    print(person_map_label)

    x_train = result 
    y_train = y_prep
#   x_test = testResult
#    y_test = y_prep_Test
    
    print("\n\n\n-------------------------------EVALUATION SCORES FOR THE JOINT SYSTEM---------------------------------\n\n\n")

    print("XGBoost\n") 

    personXGB= XGBClassifier()
    personXGB.fit(x_train, y_train)
    personFilename = 'final_eng_person_model.sav'
    joblib.dump(personXGB, open(personFilename, 'wb'))

    
    
    print("---------------------------------------------------------------------------------------------------------------------------------------------\n")
    print("------------------------------------             CREATIVEWORKS DOMAIN             ------------------------------------------------------")
    
    cwcorpus, cwtags = DomainCorpusGenerator('/home/omkar/multiconer-baseline/eng_prediction.txt', "CreativeWorks", "train")
    cwTestCorpus,cwTestTags = DomainCorpusGenerator('/home/omkar/multiconer-baseline/eng_test_prediction.txt', "CreativeWorks", "test")
    
    
    textcorpus = cwcorpus
    textTestcorpus = cwTestCorpus
    #testResult = generateEmbeddings(modelname=modelname, sentencelist=textTestcorpus, embeddingtype="word")
    #testResult = np.array(testResult)
    
    result = generateEmbeddings(modelname=modelname, sentencelist=textcorpus+textTestcorpus, embeddingtype="word")
    result = np.array(result)
    
    finetags = cwtags
    fineTesttags = cwTestTags

    totalTags = finetags + fineTesttags
    totalTags = np.array(totalTags)
    cw_map_label = {}
    cw_map_label['VisualWork'] = 0
    cw_map_label['MusicalWork'] = 1
    cw_map_label['WrittenWork'] = 2
    cw_map_label['ArtWork'] = 3
    cw_map_label['Software'] = 4
    cw_map_label['Wrong'] = 5
    y_prep = np.asarray([cw_map_label[l] for l in totalTags])
    #y_prep_Test = np.asarray([cw_map_label[l] for l in fineTesttags])
    
    print(cw_map_label)

    x_train = result
    y_train = y_prep 
    #x_test = testResult
    #y_test = y_prep_Test


    
    print("\n\n\n-------------------------------EVALUATION SCORES FOR THE JOINT SYSTEM---------------------------------\n\n\n")

    print("---------------------------------CREATIVEWORKS - JOINT SYSTEM -----------------------")


    cwXGB= XGBClassifier()
    cwXGB.fit(x_train, y_train)
    cwFilename = 'final_eng_cw_model.sav'
    joblib.dump(cwXGB, open(cwFilename, 'wb'))
    
    print("---------------------------------------------------------------------------------------------------------------------------------------------\n")
    print("------------------------------------             GROUP DOMAIN             ------------------------------------------------------")
    
    groupcorpus, grouptags = DomainCorpusGenerator('/home/omkar/multiconer-baseline/eng_prediction.txt', "Group", "train")
    groupTestCorpus, groupTestTags = DomainCorpusGenerator('/home/omkar/multiconer-baseline/eng_test_prediction.txt', "Group", "test")
    
    
    textcorpus = groupcorpus
    textTestcorpus = groupTestCorpus
    
    result = generateEmbeddings(modelname=modelname, sentencelist=textcorpus+textTestcorpus, embeddingtype="word")
    result = np.array(result)
    
    #testResult = generateEmbeddings(modelname=modelname, sentencelist=textTestcorpus, embeddingtype="word")
    #testResult = np.array(testResult)
    
    finetags = grouptags
    fineTesttags = groupTestTags
    
    
    totalTags = finetags + fineTesttags
    totalTags = np.array(totalTags)    
    group_map_label = {}
    group_map_label['MusicalGRP'] = 0
    group_map_label['PublicCorp'] = 1
    group_map_label['PrivateCorp'] = 2
    group_map_label['SportsGRP'] = 3
    group_map_label['ORG'] = 4
    group_map_label['CarManufacturer'] = 5
    group_map_label['AerospaceManufacturer'] = 6
    group_map_label['Wrong'] = 7
    y_prep = np.asarray([group_map_label[l] for l in totalTags])
    #y_prep_Test = np.asarray([group_map_label[l] for l in fineTesttags])
    
    print(group_map_label)

    x_train = result
    y_train = y_prep
    #x_test = testResult
    #y_test = y_prep_Test

    print("\n\n\n-------------------------------EVALUATION SCORES FOR THE JOINT SYSTEM---------------------------------\n\n\n")

    print("---------------------------------GROUP - JOINT SYSTEM -----------------------")


    groupXGB= XGBClassifier()
    groupXGB.fit(x_train, y_train)
    groupFilename = 'final_eng_group_model.sav'
    joblib.dump(groupXGB, open(groupFilename, 'wb'))


    print("---------------------------------------------------------------------------------------------------------------------------------------------\n")
    print("------------------------------------             PRODUCT DOMAIN             ------------------------------------------------------")
    
    productcorpus, producttags = DomainCorpusGenerator('/home/omkar/multiconer-baseline/eng_prediction.txt', "Product", "train")
    productTestCorpus, productTestTags = DomainCorpusGenerator('/home/omkar/multiconer-baseline/eng_test_prediction.txt', "Product", "test")
    
    textcorpus = productcorpus
    textTestcorpus = productTestCorpus
    
    result = generateEmbeddings(modelname=modelname, sentencelist=textcorpus + textTestcorpus, embeddingtype="word")
    result = np.array(result)
    #testResult = generateEmbeddings(modelname=modelname, sentencelist=textTestcorpus, embeddingtype="word")
    #testResult = np.array(testResult)
    
    finetags = producttags    
    fineTesttags = productTestTags
    
    totalTags = finetags + fineTesttags
    totalTags = np.array(totalTags)  
    
    for j in range(len(totalTags)):
        print(totalTags[j])
    
    
    product_map_label = {}  
    product_map_label['OtherPROD'] = 0
    product_map_label['Drink'] = 1
    product_map_label['Food'] = 2
    product_map_label['Vehicle'] = 3
    product_map_label['Clothing'] = 4
    product_map_label['Wrong'] = 5

    y_prep = np.asarray([product_map_label[l] for l in totalTags])
    #y_prep_Test = np.asarray([product_map_label[l] for l in fineTesttags])
    
    
    print(product_map_label)

    x_train = result
    y_train = y_prep
    #x_test = testResult
    #y_test = y_prep_Test

    

    print("---------------------------------PRODUCT - JOINT SYSTEM -----------------------")


    productXGB= XGBClassifier()
    productXGB.fit(x_train, y_train)
    prodFilename = 'final_eng_prod_model.sav'
    joblib.dump(productXGB, open(prodFilename, 'wb'))

    print("---------------------------------------------------------------------------------------------------------------------------------------------\n")
    print("------------------------------------             LOCATION DOMAIN             ------------------------------------------------------")
    
    locationcorpus, locationtags = DomainCorpusGenerator('/home/omkar/multiconer-baseline/eng_prediction.txt', "Location", "train")
    locationTestCorpus, locationTestTags = DomainCorpusGenerator('/home/omkar/multiconer-baseline/eng_test_prediction.txt', "Location", "test")
    
    
    textcorpus = locationcorpus
    textTestcorpus = locationTestCorpus
    
    result = generateEmbeddings(modelname=modelname, sentencelist=textcorpus + textTestcorpus, embeddingtype="word")
    result = np.array(result)
    #testResult = generateEmbeddings(modelname=modelname, sentencelist=textTestcorpus, embeddingtype="word")
    #testResult = np.array(testResult)
    
    finetags = locationtags    
    fineTesttags = locationTestTags

    totalTags = finetags + fineTesttags 
    totalTags = np.array(totalTags)    
    location_map_label = {}   
    location_map_label['Facility'] = 0
    location_map_label['OtherLOC'] = 1
    location_map_label['Station'] = 2
    location_map_label['HumanSettlement'] = 3
    location_map_label['Wrong'] = 4
    y_prep = np.asarray([location_map_label[l] for l in totalTags])
    #y_prep_Test = np.asarray([location_map_label[l] for l in fineTesttags])
    
    print(location_map_label)

    x_train = result
    y_train = y_prep
    #x_test = testResult
    #y_test = y_prep_Test


    print("\n\n\n-------------------------------EVALUATION SCORES FOR THE JOINT SYSTEM---------------------------------\n\n\n")

    print("---------------------------------LOCATION - JOINT SYSTEM -----------------------")

    locationXGB= XGBClassifier()
    locationXGB.fit(x_train, y_train)
    locFilename = 'final_eng_location_model.sav'
    joblib.dump(locationXGB, open(locFilename, 'wb'))

    print("---------------------------------------------------------------------------------------------------------------------------------------------\n")
    print("------------------------------------             MEDICAL DOMAIN             ------------------------------------------------------")
    
    medicalcorpus, medicaltags = DomainCorpusGenerator('/home/omkar/multiconer-baseline/eng_prediction.txt', "Medical", "train")
    medicalTestCorpus, medicalTestTags = DomainCorpusGenerator('/home/omkar/multiconer-baseline/eng_test_prediction.txt', "Medical", "test")

    
    
    textcorpus = medicalcorpus
    textTestcorpus = medicalTestCorpus
    
    result = generateEmbeddings(modelname=modelname, sentencelist=textcorpus + textTestcorpus, embeddingtype="word")
    result = np.array(result)
    
    #testResult = generateEmbeddings(modelname=modelname, sentencelist=textTestcorpus, embeddingtype="word")
    #testResult = np.array(testResult)
    
    finetags = medicaltags
    fineTesttags = medicalTestTags

    totalTags = finetags + fineTesttags
    totalTags = np.array(totalTags)    
    
    for i in range(len(totalTags)):
        print(totalTags[i])      
    
    medical_map_label={} 
    medical_map_label['Medication/Vaccine'] = 0
    medical_map_label['MedicalProcedure'] = 1
    medical_map_label['AnatomicalStructure'] = 2
    medical_map_label['Symptom'] = 3
    medical_map_label['Disease'] = 4
    medical_map_label['Wrong'] = 5
    y_prep = np.asarray([medical_map_label[l] for l in totalTags])
    #y_prep_Test = np.asarray([medical_map_label[l] for l in fineTesttags])
    

    x_train = result
    y_train = y_prep
    #x_test = testResult
    #y_test = y_prep_Test

    print("---------------------------------MEDICAL - JOINT SYSTEM -----------------------")
    print("\n\n\n-------------------------------EVALUATION SCORES FOR THE JOINT SYSTEM---------------------------------\n\n\n")

    medicalXGB= XGBClassifier()
    medicalXGB.fit(x_train, y_train)
    medFilename = 'final_eng_medical_model.sav'
    joblib.dump(medicalXGB, open(medFilename, 'wb'))


   

if __name__ == "__main__":
    main()
    print("Script complete")