from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS,Rating
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.mllib.clustering import KMeans
import numpy as np
from breeze.linalg import *
from breeze.numerics import pow

sc=SparkContext("local[2]","spark cluster app")
movie_file_path="/usr/bigdata/data/ml-100k/u.item"
genre_file_path="/usr/bigdata/data/ml-100k/u.genre"
rating_file_path="/usr/bigdata/data/ml-100k/u.data"

movie_raw_data=sc.textFile(movie_file_path)
genre_raw_data=sc.textFile(genre_file_path)
rating_raw_data=sc.textFile(rating_file_path)

#print movie_raw_data.map(lambda line:line.split("|")).take(5)
#print genre_raw_data.collect()

genre_data=genre_raw_data.filter(lambda line:line!="").map(lambda line:line.split("|")).map(lambda fields:(int(fields[1]), fields[0])).collectAsMap()

#print genre_data

def genresOfMovie(fields):
    g=[]
    for i in range(len(fields)):
        if int(fields[i])==1:
            g.append(genre_data[i])
    return g
    #return fields.zipWithIndex().filter(lambda (g,idx):g==1).map(lambda (g,idx):genre_data[idx])

titlesAndGenres=movie_raw_data.map(lambda line:line.split("|")).map(lambda fields:(int(fields[0]),fields[1], genresOfMovie(fields[5:])  ))

#print titlesAndGenres.first()

ratings=rating_raw_data.map(lambda line:line.split("\t")).map(lambda fields: Rating(int(fields[0]),int(fields[1]),float(fields[2])) )
#print ratings.take(5)
ratings.cache()

alsModel=ALS.train(ratings, 50, 10, 0.1)

movieFactors=alsModel.productFeatures().map(lambda (id,factor):(id, Vectors.dense(factor)))
movieVectors=movieFactors.map(lambda (id,factor):factor)
userFactors=alsModel.userFeatures().map(lambda (id,factor):(id,Vectors.dense(factor)))
userVectors=userFactors.map(lambda (id,factor):factor)

movieMatrix=RowMatrix(movieVectors)
movieMatrixSummary=movieMatrix.computeColumnSummaryStatistics()

userMatrix=RowMatrix(userVectors)
userMatrixSummary=userMatrix.computeColumnSummaryStatistics()

#print "movie factors mean: " 
#print movieMatrixSummary.mean()
#print "movie factors variance: "
#print movieMatrixSummary.variance()

#print "user factors mean: " 
#print userMatrixSummary.mean()
#print "user factors mean: "
#print userMatrixSummary.mean()


numCLusters=5
numIterations=10
numRuns=3

movieClusterModel=KMeans.train(movieVectors, numClusters, numIterations, numRuns)

userClusterModel=KMeans.train(userVectors, numClusters, numIterations, numRuns)


def computeDistance(v1,v2):
    return pow(v1-v2).sum()


titlesWithFactors=titlesAndGenres.join(movieFactors)

def movie_dist(factor):
    pred=movieClusterModel.predict(factor)
    clusterCenter=movieClusterModel.clusterCenters(pred)
    return computeDistance(DenseVector(clusterCenter), DenseVector(factor))

moviesAssigned=titlesWithFactors.map(lambda (id,title,genres,factor):(id, title, genres, movieClusterModel.predict(factor), movie_dist(factor) ))

clusterAssignments=moviesAssigned.groupByKey(lambda (id,title,genres,cluster,dist):cluster).collectAsMap()

print clusterAssignments[0][:5]

movieCost=movieClusterModel.computeCost(movieVectors)
