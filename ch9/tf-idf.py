from pyspark import SparkContext
from pyspark.mllib.linalg import SparseVector as SV
from pyspark.mllib.feature import HashingTF,IDF
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.feature import Word2Vec
import math
sc=SparkContext("local[2]","spark tf-idf app")

file_path_train="/usr/bigdata/data/20news-bydate-train/*"
file_path_test="/usr/bigdata/data/20news-bydate-test/*"

rdd=sc.wholeTextFiles(file_path_train)
testRdd=sc.wholeTextFiles(file_path_test)

text=rdd.map(lambda (name,text):text)

#text=rdd.map(lambda (name,text):text)


#print text.count()

newsgroups=text.map(lambda (name,text):name.split("/")[-2:-1])

countByGroup=newsgroups.map(lambda group:(group,1)).reduceByKey(lambda (x,y):x+y)

#print countByGroup.collect()

whiteSpaceSplit=text.flatMap(lambda t:t.split(" ")).map(lambda w:w.lower())

#print whiteSpaceSplit.distinct().count()

nonWordSplit=text.flatMap(lambda t:t.split("""\W+""")).map(lambda w:w.lower())

#print nonWordSplit.distinct().count()

tokenCounts=nonWordSplit.map(lambda t:(t,1)).reduceByKey(lambda (x,y):x+y)

stopwords=["the", "a", "an", "of", "or", "in", "for", "by", "on", "but", "is", "not", "with", "as", "was", "if", "they", "are", "this", "and", "it", "have", "from", "at", "my", "be", "that", "to"]

tokenCountsFilteredStopwords=tokenCounts.filter(lambda (k,v) : k in stopwords)

#print tokenCountsFilteredStopwords.take(20)

def tokenize(line):
    words=line.split("""\W+""")
    lower_words=[w.lower() for w in words]
    return [w for w in lower_words if w not in stopwords and len(w)>=2]

tokens=text.map(lambda doc:tokenize(doc))

#print tokens.first()[:20]

dim=int(math.pow(2,18))
hashingTF=HashingTF(dim)

tf=hashingTF.transform(tokens)

tf.cache()

#v1=tf.first() as SV
#print v1.size
#print v1.values.size
#print v1.values[:10]
#print v1.indices[:10]

idf=IDF().fit(tf)

tfidf=idf.transform(tf)

#hockeyTf=rdd.filter(lambda (name,text):name.contains("hockey")).map(lambda (name,text):hashingTF.transform(tokenize(text)))
#hockeyTfidf=idf.transform(hockeyTf)

newsgroupsMap=newsgroups.distinct().collect().zipWithIndex().collectAsMap()

zipped=newsgroups.zip(tfidf)

train=zipped.map(lambda (topic,vector):LabeledPoint(newsgroupsMap(topic),vector))


testLables=testRdd.map(lambda (name,text): newsgroupsMap(name.split("/")[-2:-1]))

testTf=testRdd.map(lambda (name,text):hashingTF.transform(tokenize(text)))
testTfidf=idf.transform(testTf)
zippedTest=testLables.zip(testTfidf)
test=zippedTest.map(lambda (topic,vector):LabeledPoint(topic,vector))

predictionAndLabel=test.map(lambda p: (nbModel.predict(p.features), p.label))

accuracy=1.0*predictionAndLabel.filter(lambda (pred,actual) : pred==actual).count()/test.count()

metrics=MulticlassMetrics(predictionAndLabel)

#print accuracy
#print metrics.weightedFMeasure

"""
word2vec
"""
word2vec=Word2Vec()
word2vec.setSeed(42)
word2vecModel=word2vec.fit(tokens)

hockeySimilar=word2vecModel.findSynonyms("hockey",20)
print hockeySimilar.collec()
