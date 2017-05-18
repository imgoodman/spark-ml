#-*- coding:utf8-*-
from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.regression import LinearRegressionWithSGD
from pyspark.mllib.tree import DecisionTree
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
#afirst_data=data.first()
#print "raw data: "+str(records.first()[2:])

#print "linear model feature vector: " + str(first_data.features)
#print "length of linear model feature vector: %d" % len(first_data.features)
#print "label of linear model fetaure: %f" % first_data.label

data_dt=records.map(lambda r: LabeledPoint(extract_label(r), extract_features_dt(r)))
#first_data_dt=data_dt.first()
#print "decision tree feature vector: " + str(first_data_dt.features)
#print "length of decision tree feature vector: %d" % len(first_data_dt.features)

"""
train linear model
"""
linear_model=LinearRegressionWithSGD.train(data, iterations=10,step=0.1, intercept=False)
true_vs_predicted=data.map(lambda p: (p.label,linear_model.predict(p.features)))
#print "linear model predictions:"+str(true_vs_predicted.take(5))


"""
train decision tree
"""
dt_model=DecisionTree.trainRegressor(data_dt, {})
#true_vs_predicted_dt=data_dt.map(lambda p:(p.label,dt_model.predict(p.features)))
#print "decision tree predictions:"+str(true_vs_predicted_dt.take(5))
#print "depth of decision tree"+str(dt_model.depth())
#print "nodes of decision tree"+str(dt_model.numNodes())

"""
performance of regression model
actual and prediction
"""
def squared_error(actual, pred):
    return (pred-actual)**2

def abs_error(actual,pred):
    return np.abs(pred-actual)

def squared_log_error(actual,pred):
    return (np.log(pred+1) - np.log(actual+1))**2


"""
error of linear model
"""
mse=true_vs_predicted.map(lambda (actual,pred): squared_error(actual,pred)).mean()
mae=true_vs_predicted.map(lambda (actual,pred):abs_error(actual,pred)).mean()
rmsle=true_vs_predicted.map(lambda (actual,pred):squared_log_error(actual,pred)).mean()
#print "Mean Squared Error of Linear Model: %f" % mse
#print "Mean Absolute Error of Linear Model: %f" % mae
#print "Root Mean Squared Log Error of Linear Model: %f" % rmsle



"""
transfer target value with log, sqrt
to get normal distribution
"""
log_data=data.map(lambda p: LabeledPoint(np.log(p.label),p.features))
log_targets=records.map(lambda r : np.log(float(r[-1])))
log_lrmodel=LinearRegressionWithSGD.train(log_data, iterations=10,step=0.1)
sqrt_targets=records.map(lambda r: np.sqrt(float(r[-1])))



"""
cross validation
with different parameters
to evaluate performance of model
"""
data_with_idx=data.zipWithIndex().map(lambda (k,v):(v,k))
test=data_with_idx.sample(False, 0.2, 42)
train=data_with_idx.subtractByKey(test)
train_data=train.map(lambda (k,v):v)
test_data=test.map(lambda (k,v):v)
train_size=train_data.count()
test_size=test_data.count()
#print "train data size: %d" % train_size
#print "test data size: %d" % test_size

"""
evaluate the performance of model 
in different params
"""
def evaluate(train,test,iterations,step,regParam,regType,intercept):
    model=LinearRegressionWithSGD.train(train, iterations, step, regParam=regParam, regType=regType, intercept=intercept)
    tp=test.map(lambda p:(p.label, model.predict(p.features)))
    rmsle=np.sqrt(tp.map(lambda (t,p):squared_log_error(t,p)).mean())
    return rmsle

"""
iterations influence
"""
def evaluate_iterations():
    params=[1,5,10,20,50,100]
    metrics=[evaluate(train_data, test_data, p, 0.01, 0.0, 'l2', False) for p in params]
    print params
    print metrics
#evaluate_iterations()


"""
step influence
"""
def evaluate_step():
    params=[0.01, 0.025,0.05,0.1,1.0]
    metrics=[evaluate(train_data, test_data, 10, p, 0.0, 'l2', False) for p in params]
    print params
    print metrics
#evaluate_step()


"""
regulation L2 influence
"""
def evaluate_l2():
    params=[0.0, 0.01, 0.1, 1.0, 5.0, 10.0, 20.0]
    metrics=[evaluate(train_data, test_data, 10, 0.1, p, 'l2', False) for p in params]
    print params
    print metrics
#evaluate_l2()


"""
regulation L1 influence
"""
def evaluate_l1():
    params=[0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    metrics=[evaluate(train_data, test_data, 10, 0.1, p, 'l1', False) for p in params]
    print params
    print metrics
evaluate_l1()
