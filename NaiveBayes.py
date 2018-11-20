
import math
import random
import pandas as pd

def loadCsv(filename):
    lines = pd.read_csv(r'iris.data')
    snf = lines.iloc[:,-1]
    veri=lines.iloc[:,0:4]
    
    pd.DataFrame(veri)
    
    from sklearn.preprocessing import LabelEncoder
    labelencoder=LabelEncoder()   
    snf=labelencoder.fit_transform(snf)  
    
    sonuc = pd.DataFrame(data = snf)
    sonuc1=pd.DataFrame(data = veri)                     
    s=pd.concat([sonuc1,sonuc],axis=1)
    return s

def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = dataset
    while len(trainSet) < trainSize:
        ind = random.randrange(len(copy))
        trainSet.append(dataset.iloc[[ind]])
        copy.drop(copy.index[ind], inplace=True)
    return [trainSet, copy]


def mean(numbers):
    topla=0
    for i, row in numbers.iterrows():
        for j, column in row.iteritems():
            topla  =float(topla)+ float(column)
    return topla/float(len(numbers.keys()))

def stdev(numbers):
    degerler=numbers
    avg = mean(numbers)
    topla=0
    for i, row in numbers.iterrows():
        for j, column in row.iteritems():
            topla += pow(float(column)-avg,2)
    variance=topla/float(len(degerler.keys())-1)
    return math.sqrt(variance)

def summarize(dataset):
    for attribute in dataset:
        summaries = [(mean(attribute), stdev(attribute))]        
    return summaries

def separateByClass(dataset):
    separated = {}
    pd.DataFrame(separated)
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector.iloc[:,-1].values[0] not in separated):
            sc=vector.iloc[:,-1].values[0]
            separated[sc] = []
        separated[vector.iloc[:,-1].values[0]].append(vector)
    return separated

def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        gelen=summarize(instances)
        summaries[classValue]=gelen
    return summaries

def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(float(x)-mean,2)/(2*math.pow(stdev,2))))
    return (1/(math.sqrt(2*math.pi)*stdev))*exponent

def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        mean=classSummaries[0][0]
        stdev = classSummaries[0][1]
        for i in range(len(inputVector.keys())):
            x = inputVector.iloc[0].values[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities

def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability >bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel

def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet.iloc[[i]])
        predictions.append(result)
    return predictions

def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet.iloc[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet)))*100.0

def main():
    filename = 'F:\Dropbox\Doktora\python\iris.data'
    splitRatio = 0.66
    dataset = loadCsv(filename)
    trainingSet, testSet = splitDataset(dataset, splitRatio)
    print('Split {0} rows into train = {1} and test = {2} rows'.format(len(dataset),len(trainingSet),len(testSet)))
    #prepare model
    summaries = summarizeByClass(trainingSet)
    #test model
    predictions = getPredictions(summaries, testSet)
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: {0}%'.format(accuracy))

main()
