# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 23:06:13 2018

@author: ramgopal
"""

import os
import math 
import sys
D=[]           #training data
T=[]           #testing data
Class=[0,1]

train_ham=sys.argv[1]
train_spam=sys.argv[2]
test_ham=sys.argv[3]
test_spam=sys.argv[4]
stopwordsfile=sys.argv[5]
z=sys.argv[6]

D.append(train_ham)
D.append(train_spam)
stopWords=open(stopwordsfile).read()
stopWords=stopWords.split(',')
T.append(test_ham)
T.append(test_spam)


def fileOpening(mainPath,extension):   #opens doc from the specified paths
    absPath=mainPath+extension 
    file=open(absPath,encoding='latin1')
    string=file.read()
    string=string.lower()
    return string.split()


def filterDoc(d):  #removes all the stopwords,numbers in the doc
    doc=[]
    for word in d:
      if word.isalpha():
         if z =='yes':
            doc.append(word)
         else:
            if word not in stopWords:
              doc.append(word)
    return doc


def extractVocabulary(D): #extracts voacabulary for both Ham & Spam docs
    V=[]
    for path in D:
        dirs=os.listdir(path)
        for fName in dirs:  
           words=fileOpening(path,fName)
           word=filterDoc(words)
           for i in word:
              if i not in V:
                 V.append(i)
    return V           
        
        
def countDocs(D): #counts the total number of docs (N)
    N=0
    for paths in D:
        dirs=os.listdir(paths)
        N+=len(dirs)
    return N


def countDocsInClass(D,c): #counts the total number of docs in that class (Nc)
    size=os.listdir(D[c])
    return len(size)


def countTokens(text): #makes a dictionary out of text0(ham),text1(spam)
    count=[{},{}]
    for c in [0,1]:
        pairs={}
        pairs=countPairs(text[c])
        count[c]=pairs
    return count


def countPairs(doc):  #makes a dictionary for the given doc
    dic={}
    for word in doc:
        if word not in dic:
            dic[word]=1
        else:
            dic[word]=dic.get(word)+1
    return dic  


def concatenateTextOfAllDocsInClass(D,c):  #makes a single text doc out of all docs in that class
    text=[]
    dirs=os.listdir(D[c])
    for fName in dirs:  
        words=fileOpening(D[c],fName)
        word=filterDoc(words)
        for i in word:
            text.append(i)       
    return text   


def extractTokensFromDoc(vocab,doc): #extracts words that are only exisiting in V from doc
    d=[]
    for word in doc:
        if word in vocab:
            d.append(word)
    return d        

            
def condProb(word,c,f): #gives the conditional probability of a word in a class 
    if f[c].get(word) is None:
        return 0
    else:
        return f[c].get(word)
    

def accuracy(words,prior,freq,tc,modV): #classifies the doc as either spam/ham
    score=[]
    for i in [0,1]:
        var=0
        var=math.log(prior[i])
        for t in words:
            x=condProb(t,i,freq)
            var+=math.log((x+1)/(modV+tc))
        score.append(var)
    return score.index(max(score))

    
def trainMultinomialNB(D,C): #training using docs
    text=[[],[]]
    prior=[]
    vocab=extractVocabulary(D)
    count=countDocs(D)
    for c in C:
        count_c=countDocsInClass(D,c)
        prior.append(count_c/count)
        text[c]=concatenateTextOfAllDocsInClass(D,c) 
    return [vocab,prior,text]           

        
def applyMultinomialNB(V,prior,tD,freq): #testing of docs (accuracy)
    count=0
    modV=len(V)
    for c in [0,1]:
        tc=sum(freq[c].values()) 
        dirs=os.listdir(tD[c])
        for fName in dirs:  
           words=fileOpening(tD[c],fName)
           doc=filterDoc(words)
           W=extractTokensFromDoc(V,doc)
           if accuracy(W,prior,freq,tc,modV) is c:
               count+=1    
    return 100*(count/countDocs(tD))         
 
if __name__ == '__main__':
	
	store=trainMultinomialNB(D,Class) 
	bag=countTokens(store[2])
	accuracy_test=round(applyMultinomialNB(store[0],store[1],T,bag),2)
	print('Accuracy on test data:',accuracy_test)


    
    