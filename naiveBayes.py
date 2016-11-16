#!/usr/bin/python

import sys
import os
import numpy as np
import math
from sklearn.naive_bayes import MultinomialNB

###############################################################################

NEG = -1
POS = 1
vocabulary = ['love', 'wonderful', 'best', 'great', 'superb', 'still',
            'beautiful', 'bad', 'worst', 'stupid', 'waste', 'boring', '?', '!', 'UNK']

def transfer(fileDj, vocabulary):

    freaks = []
    for i in vocabulary:
        freaks.append(0)
    with open(fileDj, 'r') as f:
        for line in f:
            new_line = line.replace('loving', 'love').replace('loves', 'love').replace('loved', 'love')
            words = new_line.split()
            for word in words: 
                if word in vocabulary:
                    freaks[vocabulary.index(word)] += 1
                else:
                    freaks[len(freaks) - 1] += 1
    return freaks


def loadData(Path):
    test_neg_files = os.listdir(Path + '/test_set/neg')
    test_pos_files = os.listdir(Path + '/test_set/pos')

    train_neg_files = os.listdir(Path + '/training_set/neg')
    train_pos_files = os.listdir(Path + '/training_set/pos')

    Xtrain = []
    ytrain = [[]]
    for j in train_neg_files:
        Xtrain.append(transfer(Path + '/training_set/neg/' + j, vocabulary))
        ytrain[0].append(NEG)
    for j in train_pos_files:
        Xtrain.append(transfer(Path + '/training_set/pos/' + j, vocabulary))
        ytrain[0].append(POS)

    Xtest = []
    ytest = [[]]
    for j in test_neg_files:
        Xtest.append(transfer(Path + '/test_set/neg/' + j, vocabulary))
        ytest[0].append(NEG)
    for j in test_pos_files:
        Xtest.append(transfer(Path + '/test_set/pos/' + j, vocabulary))
        ytest[0].append(POS)

    Xtrain = np.array(Xtrain)
    Xtest = np.array(Xtest)
    ytrain = np.array(ytrain).T
    ytest = np.array(ytest).T

    return Xtrain, Xtest, ytrain, ytest


def naiveBayesMulFeature_train(Xtrain, ytrain):
    ALPHA = 1.0
    thetaPos = []
    thetaNeg = []
    train_data = np.concatenate((Xtrain, ytrain), axis = 1)

    nPos = []
    nNeg = []

    for line in train_data:
        if len(nPos) < 1 and len(nNeg) < 1:
            for i in range(0, len(line) - 1):
                nPos.append(0)
                nNeg.append(0)
                thetaPos.append(0)
                thetaNeg.append(0)
        for index, feature in enumerate(line[:-1]):
            if line[len(line) - 1] == POS:
                nPos[index] += feature
            elif line[len(line) - 1] == NEG:
                nNeg[index] += feature

    nPosSum = 0
    for i in nPos:
        nPosSum += i
    nNegSum = 0
    for i in nNeg:
        nNegSum += i

    for index, i in enumerate(nPos):
        thetaPos[index] = (nPos[index] + ALPHA)/(nPosSum + ALPHA*len(nPos))
    for index, j in enumerate(nNeg):
        thetaNeg[index] = (nNeg[index] + ALPHA)/(nNegSum + ALPHA*len(nNeg))

    return thetaPos, thetaNeg


def naiveBayesMulFeature_test(Xtest, ytest,thetaPos, thetaNeg):
    yPredict = []
    for row in Xtest:
        prodNeg = 0.0
        prodPos = 0.0
        for index, s in enumerate(row):
            if s == 0:
                continue
            prodNeg += math.log(thetaNeg[index])*s
            prodPos += math.log(thetaPos[index])*s
        if prodNeg > prodPos:
            yPredict.append(NEG)
        else:
            yPredict.append(POS)

    total = 0.0
    correct = 0.0
    for index, i in enumerate(ytest):
        total += 1
        if i[0] == yPredict[index]:
            correct += 1
    Accuracy = correct / total

    return yPredict, Accuracy

def naiveBayesMulFeature_sk_MNBC(Xtrain, ytrain, Xtest, ytest):
    clf = MultinomialNB()
    clf.fit(Xtrain, ytrain.T[0])
    Accuracy = clf.score(Xtest, ytest)
    return Accuracy


def naiveBayesMulFeature_testDirectOne(path,thetaPos, thetaNeg, vocabulary):
    
    sumPos = 0.0
    sumNeg = 0.0
    with open(path, 'r') as f:
        for line in f:
            new_line = line.replace('loving', 'love').replace('loves', 'love').replace('loved', 'love')
            words = new_line.split()
            for index, word in enumerate(words): 
                if word in vocabulary:
                    sumPos += math.log(thetaPos[vocabulary.index(word)])
                    sumNeg += math.log(thetaNeg[vocabulary.index(word)])
                else:
                    sumPos += math.log(thetaPos[len(vocabulary) - 1])
                    sumNeg += math.log(thetaNeg[len(vocabulary) - 1])

    if sumPos > sumNeg:
        yPredict = POS
    else:
        yPredict = NEG

    return yPredict


def naiveBayesMulFeature_testDirect(path,thetaPos, thetaNeg, vocabulary):
    yPredict = []

    test_neg_files = os.listdir(path + '/neg')
    test_pos_files = os.listdir(path + '/pos')

    total = 0.0
    correct = 0.0

    for f in test_neg_files:
        total += 1
        value = naiveBayesMulFeature_testDirectOne(path + '/neg/' + f, thetaPos, thetaNeg, vocabulary)
        yPredict.append(value)
        if value == NEG:
            correct += 1

    for f in test_pos_files:
        total += 1
        value = naiveBayesMulFeature_testDirectOne(path + '/pos/' + f, thetaPos, thetaNeg, vocabulary)
        yPredict.append(value)
        if value == POS:
            correct += 1

    Accuracy = correct / total

    return yPredict, Accuracy



def naiveBayesBernFeature_train(Xtrain, ytrain):
    thetaPosTrue = []
    thetaNegTrue = []
    train_data = np.concatenate((Xtrain, ytrain), axis = 1)

    nPos = []
    nNeg = []
    nPosSum = 0.0
    nNegSum = 0.0

    for line in train_data:
        if len(nPos) < 1 and len(nNeg) < 1:
            for i in range(0, len(line) - 1):
                nPos.append(0)
                nNeg.append(0)
                thetaPosTrue.append(0)
                thetaNegTrue.append(0)
        for index, feature in enumerate(line[:-1]):
            if line[len(line) - 1] == POS:
                nPos[index] += min(feature, 1)
            elif line[len(line) - 1] == NEG:
                nNeg[index] += min(feature, 1)
        if line[len(line) - 1] == POS:
            nPosSum += 1
        elif line[len(line) - 1] == NEG:
            nNegSum += 1

    for index, i in enumerate(nPos):
        thetaPosTrue[index] = (nPos[index] + 1)/(nPosSum + 2)
    for index, j in enumerate(nNeg):
        thetaNegTrue[index] = (nNeg[index] + 1)/(nNegSum + 2)
    return thetaPosTrue, thetaNegTrue

    
def naiveBayesBernFeature_test(Xtest, ytest, thetaPosTrue, thetaNegTrue):
    yPredict = []
    for row in Xtest:
        prodNeg = 0.0
        prodPos = 0.0
        for index, s in enumerate(row):
            if s == 0:
                prodNeg += math.log(1 - thetaNegTrue[index])
                prodPos += math.log(1 - thetaPosTrue[index])
            else:
                prodNeg += math.log(thetaNegTrue[index])
                prodPos += math.log(thetaPosTrue[index])
        if prodNeg > prodPos:
            yPredict.append(NEG)
        else:
            yPredict.append(POS)

    total = 0.0
    correct = 0.0
    for index, i in enumerate(ytest):
        total += 1
        if i[0] == yPredict[index]:
            correct += 1
    Accuracy = correct / total
    
    return yPredict, Accuracy


if __name__ == "__main__":

    loadData('data_sets')

    if len(sys.argv) != 3:
        print "Usage: python naiveBayes.py dataSetPath testSetPath"
        sys.exit()

    print "--------------------"
    textDataSetsDirectoryFullPath = sys.argv[1]
    testFileDirectoryFullPath = sys.argv[2]


    Xtrain, Xtest, ytrain, ytest = loadData(textDataSetsDirectoryFullPath)

    thetaPos, thetaNeg = naiveBayesMulFeature_train(Xtrain, ytrain)

    print "thetaPos =", thetaPos
    print "thetaNeg =", thetaNeg
    print "--------------------"

    yPredict, Accuracy = naiveBayesMulFeature_test(Xtest, ytest, thetaPos, thetaNeg)
    print "MNBC classification accuracy =", Accuracy

    Accuracy_sk = naiveBayesMulFeature_sk_MNBC(Xtrain, ytrain, Xtest, ytest)
    print "Sklearn MultinomialNB accuracy =", Accuracy_sk

    yPredict, Accuracy = naiveBayesMulFeature_testDirect(testFileDirectoryFullPath, thetaPos, thetaNeg, vocabulary)
    print "Directly MNBC tesing accuracy =", Accuracy
    print "--------------------"

    thetaPosTrue, thetaNegTrue = naiveBayesBernFeature_train(Xtrain, ytrain)
    print "thetaPosTrue =", thetaPosTrue
    print "thetaNegTrue =", thetaNegTrue
    print "--------------------"

    yPredict, Accuracy = naiveBayesBernFeature_test(Xtest, ytest, thetaPosTrue, thetaNegTrue)
    print "BNBC classification accuracy =", Accuracy
    print "--------------------"
