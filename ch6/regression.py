#-*- coding:utf8-*-
from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint
import numpy as np

file_path="/usr/bigdata/data/Bike-Sharing-Dataset/hour_noheader.csv"

sc=SparkContext("local[2]","spark regression app")

raw_data=sc.textFile(file_path)

records=raw_data.map(lambda line:line.split(","))
num_records=records.count()
#print records.first()
#print num_records

records.cache()

cat_idxs=range(2,10)
num_idxs=range(10,14)

"""
get a specific column
distinct():get its distinct values
zipWithIndex():get key-value pair (key is the column value, value is its index)
collectAsMap():transfer rdd to a python dict type
"""
def get_mapping(data,idx):
    return data.map(lambda r:r[idx]).distinct().zipWithIndex().collectAsMap()

mappings=[get_mapping(records,i) for i in cat_idxs]

#print mappings

cat_len=sum(map(len, mappings))
num_len=len(num_idxs)
total_len=cat_len+num_len

#print "total columns is: %d" % (cat_len+num_len)

def extract_features(record):
    cat_vec=np.zeros(cat_len)
    step=0
    for i in range(len(cat_idxs)):
        mapping=mappings[i]
        field=mapping[record[i+2]]
        cat_vec[step+field]=1
        step=step+len(mapping)
    num_vec=np.array([float(record[r]) for r in num_idxs])
    #print cat_vec
    #print num_vec
    return np.concatenate((cat_vec, num_vec))

def extract_label(record):
    return float(record[-1])

def extract_features_dt(record):
    return np.array([float(r) for r in record[2:14]])

data=records.map(lambda r: LabeledPoint(extract_label(r), extract_features(r)))
first_data=data.first()
print "raw data: "+str(records.first()[2:])

print "linear model feature vector: " + str(first_data.features)
print "length of linear model feature vector: %d" % len(first_data.features)
print "label of linear model fetaure: %f" % first_data.label

data_dt=records.map(lambda r: LabeledPoint(extract_label(r), extract_features_dt(r)))
first_data_dt=data_dt.first()
print "decision tree feature vector: " + str(first_data_dt.features)
print "length of decision tree feature vector: %d" % len(first_data_dt.features)


