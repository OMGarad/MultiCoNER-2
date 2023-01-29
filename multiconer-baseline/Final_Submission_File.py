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
from joblib import dump, load
from transformers import RobertaModel, BertModel
from transformers import RobertaTokenizer, RobertaForSequenceClassification, XLMRobertaTokenizer, BertTokenizer, AutoModel
import numpy as np
import pandas as pd
import tqdm
from sklearn.svm import SVC
import xgboost
from xgboost import XGBClassifier
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import random
import os
import matplotlib.pyplot as plt
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.utils import parse_args 
from utils.utils import get_reader
from utils.utils import load_model
from utils.utils import get_out_filename
from utils.utils import get_tagset


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
    print("THIS IS SENTENCELIST: ", sentencelist)
    encodedcorpus = PranayTokenizer(sentencelist, tokenizer)

    neuralmodel = Net(model=pretrainedmodel, embeddingtype=embeddingtype)
    neuralmodel.to(device)
    neuralmodel = nn.DataParallel(neuralmodel)

    outputlist = []
    for i, batch in tqdm(enumerate(encodedcorpus)):
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

def ModifiedRemoveLine():
    return 0


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
    cnt123 = 0
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
        trainingTags = FineGrainedTags('/home/omkar/multiconer-baseline/shuff-hinglish-train90.conll')
    elif test_or_train == 'test':
        trainingTags = FineGrainedTags('/home/omkar/multiconer-baseline/shuff-hinglish-dev.conll')
        
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

def FindMaxValue(arr):
    cleanedArr = arr
    print("THIS IS CLEANEDARR: ", cleanedArr)
    max = cleanedArr[0]
    index = 0
    for i in range(len(cleanedArr)):
        ch = cleanedArr[i]
        if ch>max:
            max=ch
            index = i
    return index, max
    

def FindCorrectPrediction(emb):
    medProb = medicalXGB.predict_proba(emb)
    medProb = medProb[0]
    locProb = locationXGB.predict_proba(emb)
    locProb = locProb[0]
    cwProb = cwXGB.predict_proba(emb)
    cwProb = cwProb[0]
    prodProb = productXGB.predict_proba(emb)
    prodProb = prodProb[0]
    perProb = personXGB.predict_proba(emb)
    perProb = perProb[0]
    grpProb = groupXGB.predict_proba(emb)
    grpProb =  grpProb[0]
    
    indArr = []
    probArr = []
    x, y = FindMaxValue(medProb)
    indArr.append(x)
    probArr.append(y)
    x, y = FindMaxValue(locProb)
    indArr.append(x)
    probArr.append(y)
    x, y = FindMaxValue(cwProb)
    indArr.append(x)
    probArr.append(y)
    x, y = FindMaxValue(prodProb)
    indArr.append(x)
    probArr.append(y)
    x, y = FindMaxValue(perProb)
    indArr.append(x)
    probArr.append(y)
    x, y = FindMaxValue(grpProb)
    indArr.append(x)
    probArr.append(y)
    
    a,b = FindMaxValue(probArr) #a is the index of the highest value
    innerArrInd = indArr[a]
    if a == 0:
        prediction = medical_label_map[innerArrInd]
    elif a == 1:
        prediction = location_label_map[innerArrInd]
    elif a == 2:
        prediction = cw_label_map[innerArrInd]
    elif a == 3:
        prediction = product_label_map[innerArrInd]
    elif a == 4:
        prediction = person_label_map[innerArrInd]
    elif a == 5:
        prediction =  group_label_map[innerArrInd]
    
    return prediction
        
    
     
def DeployMedicalXGBModel(emb):
    predInd = medicalXGB.predict(emb)
    predIndVal = predInd[0]
    if medical_label_map[predIndVal] == "Wrong":
        prediction = FindCorrectPrediction(emb)
    else:
        prediction =  medical_label_map[predIndVal]
    return prediction

def DeployLocationXGBModel(emb):
    predInd = locationXGB.predict(emb)
    predIndVal = predInd[0]
    if location_label_map[predIndVal] == "Wrong":
        prediction = FindCorrectPrediction(emb)
    else:
        prediction =  location_label_map[predIndVal]   
    return prediction
        
def DeployCWXGBModel(emb):
    predInd = cwXGB.predict(emb)
    predIndVal = predInd[0]
    if cw_label_map[predIndVal] == "Wrong":
        prediction = FindCorrectPrediction(emb)
    else:
        prediction = cw_label_map[predIndVal]
    return prediction
        
def DeployGroupXGBModel(emb):
    predInd = groupXGB.predict(emb)
    predIndVal = predInd[0]
    if group_label_map[predIndVal] == "Wrong":
        prediction = FindCorrectPrediction(emb)
    else:
        prediction = group_label_map[predIndVal]
        
    return prediction
        
def DeployProductXGBModel(emb):
    predInd = productXGB.predict(emb)
    predIndVal = predInd[0]
    if product_label_map[predIndVal] == "Wrong":
        prediction = FindCorrectPrediction(emb)
    else:
        prediction = product_label_map[predIndVal]
    
    return prediction

def DeployPersonXGBModel(emb):
    predInd = personXGB.predict(emb)
    predIndVal = predInd[0]
    if person_label_map[predIndVal] == "Wrong":
        prediction = FindCorrectPrediction(emb)
    else:
        prediction = person_label_map[predIndVal]

    return prediction

def FindFinalPrediction(domain, emb):
    if domain == "Medical":
        #load the medical XGB model
        finalPred = DeployMedicalXGBModel(emb)
    elif domain == "Location":
        #load the location XGB model
        finalPred = DeployLocationXGBModel(emb)
    elif domain == "CreativeWorks":
        #load the cw XGB model
        finalPred = DeployCWXGBModel(emb)
    elif domain == "Group":
        #load the group XGB model
        finalPred = DeployGroupXGBModel(emb)
    elif domain == "Product":
        #load the product XGB model
        finalPred = DeployProductXGBModel(emb)
    elif domain == "Person":
        #load the person XGB model
        finalPred = DeployPersonXGBModel(emb)
    
    return finalPred
         


def main():

    #Deploy all the XGBoost models
    global medicalXGB
    medicalXGB = joblib.load(open('final_eng_medical_model.sav', 'rb'))
    
    global groupXGB
    groupXGB = joblib.load(open('final_eng_group_model.sav', 'rb'))
    
    global personXGB
    personXGB = joblib.load(open('final_eng_person_model.sav', 'rb'))    
    
    global locationXGB
    locationXGB = joblib.load(open('final_eng_location_model.sav', 'rb'))
    
    global productXGB
    productXGB = joblib.load(open('final_eng_prod_model.sav', 'rb'))
    
    global cwXGB 
    cwXGB = joblib.load(open('final_eng_cw_model.sav', 'rb'))       
    
    #Add all the label_maps
    global product_label_map
    product_label_map = {}
    product_label_map[0] = 'OtherPROD'
    product_label_map[1] = 'Drink'
    product_label_map[2] = 'Food'
    product_label_map[3] = 'Vehicle'
    product_label_map[4] = 'Clothing'
    product_label_map[5] = 'Wrong'
    
    global medical_label_map
    medical_label_map = {}
    medical_label_map[0] = 'Medication/Vaccine'
    medical_label_map[1] = 'MedicalProcedure'
    medical_label_map[2] = 'AnatomicalStructure'
    medical_label_map[3] = 'Symptom'
    medical_label_map[4] = 'Disease'
    medical_label_map[5] = 'Wrong'
    
    global location_label_map
    location_label_map = {}
    location_label_map[0] = 'Facility'
    location_label_map[1] = 'OtherLOC'
    location_label_map[2] = 'Station'
    location_label_map[3] = 'HumanSettlement'
    location_label_map[4] = 'Wrong'
    
    global cw_label_map
    cw_label_map = {}
    cw_label_map[0] = 'VisualWork'
    cw_label_map[1] = 'MusicalWork'
    cw_label_map[2] = 'WrittenWork'
    cw_label_map[3] = 'ArtWork'
    cw_label_map[4] = 'Software'
    cw_label_map[5] = 'Wrong'
    
    global group_label_map
    group_label_map = {}
    group_label_map[0] = 'MusicalGRP'
    group_label_map[1] = 'PublicCorp'
    group_label_map[2] = 'PrivateCorp'
    group_label_map[3] = 'SportsGRP'
    group_label_map[4] = 'ORG'
    group_label_map[5] = 'CarManufacturer'
    group_label_map[6] = 'AerospaceManufacturer'
    group_label_map[7] = 'Wrong'
    
    global person_label_map
    person_label_map = {}
    person_label_map[0] = 'OtherPER'
    person_label_map[1] = 'SportsManager'
    person_label_map[2] = 'Cleric'
    person_label_map[3] = 'Politician'
    person_label_map[4] = 'Athlete'
    person_label_map[5] = 'Artist'
    person_label_map[6] = 'Scientist'
    person_label_map[7] = 'Wrong'
   
    print("OK1111111111111")
    
    timestamp = time.time()
    sg = parse_args()
    
    modelname = sg.model + "/checkpoints/"

    ckptfileName =  modelname + 'pytorch_modelfinal.ckpt'
    binFileName = modelname + 'pytorch_model.bin'

    if os.path.isfile(binFileName):
        os.rename(binFileName, ckptfileName)

    # load the dataset first
    test_data = get_reader(file_path=sg.test, target_vocab=get_tagset(sg.iob_tagging), max_instances=sg.max_instances, max_length=sg.max_length)
    
    print("OK222222222222222222")
    

    model, model_file = load_model(sg.model, tag_to_id=get_tagset(sg.iob_tagging))
    model = model.to(sg.cuda)
    
    
    # use pytorch lightnings saver here.
    eval_file = get_out_filename(sg.out_dir, model_file, prefix=sg.prefix)
    
    print("OK33333333333333")
    
    test_dataloaders = DataLoader(test_data, batch_size=sg.batch_size, collate_fn=model.collate_batch, shuffle=False, drop_last=False)
    out_str = ''
    index = 0
    for batch in tqdm(test_dataloaders, total=len(test_dataloaders)):
        pred_tags = model.predict_tags(batch, device=sg.cuda)
        

        for pred_tag_inst in pred_tags:
            out_str += '\n'.join(pred_tag_inst)
            out_str += '\n\n\n'
            #print(out_str)
        index += 1
    open(eval_file, 'wt').write(out_str)
    
    
    print("OK444444444444444444444444444444")
    
    #Open Test File
    conll_file = sg.test
    with open(conll_file) as sents:
        firstWords = sents.readlines()   #Contains all the tokens that we need to predict
    
    firstWords = list(firstWords)
    #End of Open Test File
    
    with open (eval_file) as sents1:
        coarse_tags = sents1.readlines() #Contains all the coarse-tags that the original model predicted
    
    newFirstWords = RemoveIDs(firstWords)   
    wordFile = ObtainTokens(newFirstWords)


   
    wordFile.pop(0)
    coarse_tags.pop()

    if os.path.isfile(ckptfileName):
        os.rename(ckptfileName, binFileName)
    
    finalSubmission = []
    dash = "-"    
    for x in range(len(coarse_tags)):
        print("THIS IS WORDFILE[x]: ", wordFile[x])  
        fineTag = ""
        domain = ""
        if dash in coarse_tags[x]:
            emb = generateEmbeddings(modelname=modelname, sentencelist=wordFile[x], embeddingtype="word")
            index = coarse_tags[x].find(dash)
            fineTag += coarse_tags[x][:index+1]
            domain += coarse_tags[x][index+1:-1]
            prediction = FindFinalPrediction(domain, emb)
        else:
            finalSubmission.append(coarse_tags[x])
            print("THIS IS AN O TAG: ", coarse_tags[x])
            continue
        fineTag += prediction
        print("WORD IS: ", wordFile[x])
        print("FINETAG IS: ", fineTag)
        finalSubmission.append(fineTag)

    #Write into a new file
    final_file = open('eng-submission.conll', 'w+')
    
    for i in range(len(finalSubmission)):
        final_file.write(finalSubmission[i])
    
        
        

        
    
    #Extract the words ONLY from the conll file 
    #--- Store the words in the file in an array where each sentence is separated by a newline value
    #--- Store the coarse tags in another array
    #Go through the tags part
    #If it is an O, then ignore else
    #Store the B,I part and keep the domain as a string
    #Find out which domain the tag belongs to and take out that particular XGBoost model
    #If the prediction is Wrong, find the prediction probability and store it
    #Predict using all other XGBoost models and store each of them
    #Then, store it in an array with fixed length specific positions
    #Find the element with highest value
    #Take the following fine-grained tag and append
    #ELSE, append that value to the string and add it to another array then write this array to a new file
    





if __name__ == "__main__":
    main()
    print("Script complete")