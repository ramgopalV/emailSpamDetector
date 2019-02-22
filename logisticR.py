
"""
Created on Fri Mar  2 16:29:38 2018

@author: ramgopal
"""

import os
import math 
import copy
import sys
D=[]           #training data
T=[]           #testing data
Class=[0,1]
global vocab
global w


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


def fileOpening(mainPath,extension):   #opens doc from the specified paths
    absPath=mainPath+extension 
    file=open(absPath,encoding='latin1')
    string=file.read()
    string=string.lower()
    return string.split()


def extractVocabulary(): #extracts voacabulary for both Ham & Spam docs
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


def countPairs(doc): #makes a dictionary matrix for the given doc
    dic={}
    for word in vocab:
        if word in doc:
            counter=0
            for w in doc:
                if w == word:
                    counter+=1
            dic[word]=counter        
        else:
            dic[word]=0
    return dic  


def countVector(): #countVector matrix is formed 
    cv=[[],[]]
    for c in Class:
        dirs=os.listdir(D[c])
        for fName in dirs:  
           words=fileOpening(D[c],fName)
           word=filterDoc(words)
           cv[c].append(countPairs(word)) 
    return cv             

def prob(c,l):   #calculates p(y=1/x,w) for the training set from countVector Matrix tuples    
    wx=w[0]
    for i in range(1,len(w)):
        wx+=w[i]*matrix[c][l].get(vocab[i-1])
    y=sigmoid(wx)
    return c-y  
     

def sigmoid(x):  #sigmoid activation function 
    try:
        sig=1/(1+math.exp(-x))
    except OverflowError:
        sig=0
    return sig

    
    
def dynamicStoring(): #storing (y-p(y=1/x,w)) for all samples 
   dV=[[],[]]
   for c in [0,1]:
        if c == 0:
          l=len(matrix[c])
        else:
          l=len(matrix[c])
        for i in range(0,l):
            dV[c].append(prob(c,i)) 
   return dV            


def training(n,lamda,eta): #training using L2 regularization to tune w vector  
    global w 
    temp=[0]*len(w)
    for j in range(n):
        dynVal=dynamicStoring()
        for i in range(len(w)):
            total=0
            for c in Class:
                if c == 0:
                   l=len(matrix[c])
                else:
                   l=len(matrix[c])
                for x in range(0,l):
                     if i !=0: 
                       total+=matrix[c][x].get(vocab[i-1])*dynVal[c][x]
                     else:
                         total+=dynVal[c][x]
            temp[i]=(w[i]*(1-(lamda*eta)))+(eta*total)    
        w=copy.deepcopy(temp)     
 
    
def getProb(d,c):  #calculates p(y=1/x,w) for the testing tuple
    wx=w[0]
    for i in range(1,len(w)):
        wx+=w[i]*d.get(vocab[i-1])
    y=sigmoid(wx)
    return c-y          

    
def testing():  #classifying the e-mails as spam(0)/ham(1)  
    count=0
    l=0
    for c in Class:
        dirs=os.listdir(D[c])
        l+=len(dirs)
        for fName in dirs:  
           words=fileOpening(D[c],fName)
           word=filterDoc(words)
           dic=countPairs(word)
           p=getProb(dic,c)
           if abs(round(p,1))==0:
               count+=1
    return (count/l)*100           

if __name__ == '__main__':
  train_ham=sys.argv[1]
  train_spam=sys.argv[2]
  test_ham=sys.argv[3]
  test_spam=sys.argv[4]
  stopwordsfile=sys.argv[5]
  z=sys.argv[6]             # Stopwords included(yes/no)
  n=int(sys.argv[7])        # Iterations limit 
  lamda=float(sys.argv[8])  # regularization parameter for penalizing weights
  eta=float(sys.argv[9])    # learning rate 


  D.append(train_ham)
  D.append(train_spam)
  stopWords=open(stopwordsfile).read()
  stopWords=stopWords.split(',')
  T.append(test_ham)
  T.append(test_spam)

  vocab=extractVocabulary()       
  w=[1]*(len(vocab)+1)
  matrix=countVector()
  training(n,lamda,eta)
  print('Accuracy on test data:',round(testing(),2))

       
    
    