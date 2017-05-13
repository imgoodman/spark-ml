#-*- coding:utf8-*-
from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.classification import NaiveBayes

from pyspark.mllib.tree import DecisionTree
#from pyspark.mllib.tree.configuration import Algo
#from pyspark.mllib.tree.impurity import Entropy

data_path="/usr/bigdata/data/train_noheader.tsv"

sc=SparkContext("local[2]","spark classification app")

train_data=sc.textFile(data_path)

records=train_data.map(lambda line:line.split("\t"))

#print records.first()

#test=records.map(lambda fields:fields[4:]).map(lambda fields:(float(fields[0].replace("\"","")),float(fields[1].replace("\"",""))))
#print test.first()

def convert_na(r):
    if r=="?":
        return 0.0
    else:
        return float(r)

#朴素贝叶斯 要求数值不能为负
def convert_na_nb(r):
    if r=="?":
        return 0.0
    else:
        if float(r)<0.0:
            return 0.0
        else:
            return float(r)

def dealwith(record):
    #print record
    trimmed=[r.replace("\"","") for r in record]
    #print trimmed
    label=int(trimmed[-1])
    #print label
    features=[convert_na(r) for r in trimmed[4:-1]]
    #print features
    return LabeledPoint(label, Vectors.dense(features))

def dealwithNB(record):
    trimmed=[r.replace("\"","") for r in record]
    label=int(trimmed[-1])
    features=[convert_na_nb(r) for r in trimmed[4:-1]]
    return LabeledPoint(label,Vectors.dense(features))

data=records.map(dealwith)
#为了朴素贝叶斯 
nbdata=records.map(dealwithNB)
#print data.first()

numIterations=10
maxTreeDepth=5

#训练逻辑回归模型
lrModel=LogisticRegressionWithSGD.train(data, numIterations)
#训练支持向量机模型
svmModel=SVMWithSGD.train(data,numIterations)
#训练朴素贝叶斯模型
nbModel=NaiveBayes.train(nbdata)
#训练决策树模型
#dtModel=DecisionTree.train(data, Algo.Classification, Entropy, maxDepth)
dtModel=DecisionTree.trainClassifier(data,numClasses=2, categoricalFeaturesInfo={},impurity='entropy', maxDepth=maxTreeDepth, maxBins=32 )

dataPoint=data.first()

lrPrediction=lrModel.predict(dataPoint.features)
print "true label is: %f; logistic regression predict is: %f" % (dataPoint.label, lrPrediction)
