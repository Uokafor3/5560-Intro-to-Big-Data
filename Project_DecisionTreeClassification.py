# Databricks notebook source
from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, CrossValidator
from pyspark.ml.classification import GBTClassifier

from pyspark.sql import functions as F
import pyspark.sql.functions as func

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer, VectorIndexer, MinMaxScaler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml import Pipeline
from pyspark.sql.types import DoubleType
from pyspark.ml.linalg import Vectors, SparseVector
import re


# COMMAND ----------

# DataFrame Schema, that should be a Table schema by Team 4 
df_mlSchema = StructType([
  StructField("user_id", IntegerType(), False),
  StructField("text", StringType(), False),
  StructField("date", TimestampType(), False),
  StructField("review_id", IntegerType(), False),
  StructField("business_id", IntegerType(), False),
  StructField("funny", IntegerType(), False),
  StructField("cool", IntegerType(), False),
  StructField("useful", IntegerType(), False),
  StructField("stars", IntegerType(), False),
])

# COMMAND ----------

# MAGIC %fs ls /FileStore/tables/df_ml.csv

# COMMAND ----------

IS_SPARK_SUBMIT_CLI = True
if IS_SPARK_SUBMIT_CLI:
    sc = SparkContext.getOrCreate()
    spark = SparkSession(sc)

# COMMAND ----------

if IS_SPARK_SUBMIT_CLI:
   df_ml = spark.read.csv('df_norm.csv', inferSchema=True, header=True)
else:
    df_ml = spark.sql("SELECT * FROM scaled_subset_csv")

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors, SparseVector
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
import re

# COMMAND ----------

#indexer = StringIndexer(inputCol="user_id", outputCol="user_index").fit(df_ml)
#df_ind = indexer.transform(df_ml)
df_ml.show(5)

# COMMAND ----------

import pyspark.sql.functions as func

# COMMAND ----------

data = df_ml.select("user_id", "review_id", "business_id", ((col("stars") > 2).cast("Double").alias("label")))

data.show(5)

# COMMAND ----------

splits = data.randomSplit([0.7, 0.3])
train = splits[0]
test = splits[1].withColumnRenamed("label", "trueLabel")
train_rows = train.count()
test_rows = test.count()
print ("Training Rows:", train_rows, " Testing Rows:", test_rows)

# COMMAND ----------

gbtassembler = VectorAssembler(inputCols=["user_id", "review_id", "business_id"], outputCol="features")

# COMMAND ----------

gbt = GBTClassifier(labelCol="label", featuresCol="features", maxIter=10) 

# COMMAND ----------

gbtp = Pipeline(stages=[gbtassembler, gbt])

# COMMAND ----------

paramGrid = (ParamGridBuilder()
             .addGrid(gbt.maxDepth,[2,3,4])
             .addGrid(gbt.maxBins, [49, 52, 55])
             .addGrid(gbt.minInfoGain,[0.0, 0.1, 0.2, 0.3])
             .addGrid(gbt.stepSize,[0.05, 0.1, 0.2, 0.4])
         
             .build())


# COMMAND ----------

gbt_tvs = TrainValidationSplit(estimator=gbtp, evaluator=MulticlassClassificationEvaluator(), estimatorParamMaps=paramGrid, trainRatio=0.8)

gbtModel = gbt_tvs.fit(train)


# COMMAND ----------

prediction = gbtModel.transform(test)
predicted = prediction.select("features", "prediction", "trueLabel")
predicted.show(10)

# COMMAND ----------

tp = float(predicted.filter("prediction == 1.0 AND truelabel == 1").count())
fp = float(predicted.filter("prediction == 1.0 AND truelabel == 0").count())
tn = float(predicted.filter("prediction == 0.0 AND truelabel == 0").count())
fn = float(predicted.filter("prediction == 0.0 AND truelabel == 1").count())
metrics = spark.createDataFrame([
      ("TP", tp),
      ("FP", fp),
      ("TN", tn),
      ("FN", fn),
      ("Precision", tp / (tp + fp)),
      ("Recall", tp / (tp + fn))],["metric", "value"])
metrics.show()

# COMMAND ----------

gbt_evaluator =  MulticlassClassificationEvaluator(labelCol="trueLabel", predictionCol="prediction")
gbt_auc = gbt_evaluator.evaluate(prediction)

print("AUC for Gradient Boost Classifier = ", gbt_auc)

# COMMAND ----------


