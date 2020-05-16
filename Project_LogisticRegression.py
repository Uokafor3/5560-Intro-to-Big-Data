# Databricks notebook source
from pyspark.sql.types import *
from pyspark.sql.functions import *

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer, StopWordsRemover

from pyspark.ml.feature import HashingTF, IDF, Tokenizer, CountVectorizer
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.linalg import Vectors, SparseVector
from pyspark.ml.clustering import LDA, BisectingKMeans
from pyspark.sql.functions import monotonically_increasing_id

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession

import re

# COMMAND ----------

IS_SPARK_SUBMIT_CLI = True

if IS_SPARK_SUBMIT_CLI:
    sc = SparkContext.getOrCreate()
    spark = SparkSession(sc)

# COMMAND ----------

# MAGIC %fs ls /FileStore/tables/df_ml.csv

# COMMAND ----------

if IS_SPARK_SUBMIT_CLI:
   df_ml = spark.read.csv('df_ml.csv', inferSchema=True, header=True)
else:
    df_ml = spark.sql("SELECT * FROM df_ml_csv")

# COMMAND ----------

df_ml.show(5)

# COMMAND ----------

data = df_ml.select("text", ((col("stars") > 2).cast("Double").alias("label")))
# data = csv
data.show(5)

# COMMAND ----------

splits = data.randomSplit([0.7, 0.3])
train = splits[0]
test = splits[1].withColumnRenamed("label", "trueLabel")
train_rows = train.count()
test_rows = test.count()
print("Training Rows:", train_rows, " Testing Rows:", test_rows)

# COMMAND ----------

# convert sentence to words' list
tokenizer = Tokenizer(inputCol="text", outputCol="SentimentWords")
# remove stop words
swr = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol="MeaningfulWords")
# convert word to number as word frequency
hashTF = HashingTF(inputCol=swr.getOutputCol(), outputCol="features")
# set the model
lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10, regParam=0.01)

# process pipeline with the series of transforms - 4 transforms
pipeline = Pipeline(stages=[tokenizer, swr, hashTF, lr])

# COMMAND ----------

piplineModel = pipeline.fit(train)
print("Pipeline complete!")

# COMMAND ----------

prediction = piplineModel.transform(test)
predicted = prediction.select("text", "prediction", "trueLabel")
predicted.show(10)

# COMMAND ----------

predicted10 = prediction.select("*")
predicted10.show(10)

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

from pyspark.ml.evaluation import BinaryClassificationEvaluator

# COMMAND ----------

evaluator = BinaryClassificationEvaluator(labelCol="trueLabel", rawPredictionCol="prediction", metricName="areaUnderROC")
aur = evaluator.evaluate(prediction)
print ("AUR =", aur)

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# COMMAND ----------

gbt_evaluator =  MulticlassClassificationEvaluator(labelCol="trueLabel", predictionCol="prediction")
gbt_auc = gbt_evaluator.evaluate(prediction)

print("AUC for Logistic Regression = ", gbt_auc)

# COMMAND ----------


