# Databricks notebook source
# MAGIC %md ## CIS5560: PySpark Pipeline Text Sentiment Analysis in Databricks
# MAGIC 
# MAGIC ### by Team 4 (Uche, Raymond, Tofunmi and Sweta) edited on 05/15/2020
# MAGIC Tested in Runtime 6.5 (Spark 2.4.5/2.4.0 Scala 2.11) of Databricks CE

# COMMAND ----------

# MAGIC %md ## Text Analysis
# MAGIC In this project, we created a classification model that performs sentiment analysis of reviews of different businesses.
# MAGIC ### Import Spark SQL and Spark ML Libraries
# MAGIC 
# MAGIC First, import the libraries you will need:

# COMMAND ----------

# MAGIC %md ## Steps to download dataset and do some data engineering (Cleaning up dataset) before importing into databricks
# MAGIC 
# MAGIC all dataset engineering were done in Jupyter Notebook before importing into databricks
# MAGIC 
# MAGIC dataset link: https://www.kaggle.com/darshank2019/review#yelp_academic_dataset_review.csv
# MAGIC 
# MAGIC download dataset and using a Jupyter Notebook(we used google colab), we accessed the dataset with total rows = 6685900
# MAGIC 
# MAGIC we took a slice of the full dataset of the first 1500000 rows and used that as our full dataset.
# MAGIC 
# MAGIC we removed the inverted commas and the letter "b" present in all rows (data cleaning)
# MAGIC 
# MAGIC we converted the alphanumeric values in the user_id, review_id, & business_id to numeric values
# MAGIC 
# MAGIC we tried to drop rows wit missing values and counted the total number of rows again and it was still 1500000.
# MAGIC 
# MAGIC we created a subset of our cleaned dataset named df_ml_csv with 120000 rows which we used for both Azure ML & Databricks
# MAGIC 
# MAGIC NOTE: the .py & .ipynb files containing all codes used for data engineering and analysis is included in the total submission package and is availble in our github link

# COMMAND ----------

# MAGIC %md Import the df_ml.csv dataset

# COMMAND ----------

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

# MAGIC %md ### Load Source Data
# MAGIC Now load the df_ml data into a DataFrame. This data consists of reviews(column = "text") that have been previously captured and classified as positive or negative.

# COMMAND ----------

# MAGIC %md Read csv file from DBFS (Databricks File Systems)
# MAGIC 
# MAGIC ## follow the direction to read your table after upload it to Data at the left frame
# MAGIC NOTE: See above for the data type - 
# MAGIC 
# MAGIC After df_ml_csv file is added to the data of the left frame, create a table using the UI, especially, "Upload File"
# MAGIC tick header and infer schema before creating table

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

# MAGIC %md ### Prepare the Data
# MAGIC The features for the classification model will be derived from the review text. The label is the stars (stars > 2 = 1 (positive review) else: 0 (negative review)

# COMMAND ----------

data = df_ml.select("text", ((col("stars") > 2).cast("Double").alias("label")))
# data = csv
data.show(5)

# COMMAND ----------

# MAGIC %md ### Split the Data
# MAGIC In common with most classification modeling processes, we splitted the data into a set for training, and a set for testing the trained model.

# COMMAND ----------

splits = data.randomSplit([0.7, 0.3])
train = splits[0]
test = splits[1].withColumnRenamed("label", "trueLabel")
train_rows = train.count()
test_rows = test.count()
print("Training Rows:", train_rows, " Testing Rows:", test_rows)

# COMMAND ----------

# MAGIC %md ### Define the Pipeline
# MAGIC The pipeline for the model consist of the following stages:
# MAGIC - A Tokenizer to split the tweets into individual words.
# MAGIC - A StopWordsRemover to remove common words such as "a" or "the" that have little predictive value.
# MAGIC - A HashingTF class to generate numeric vectors from the text values.
# MAGIC - A LogisticRegression algorithm to train a binary classification model.

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

# MAGIC %md ### Run the Pipeline as an Estimator
# MAGIC The pipeline itself is an estimator, and so it has a **fit** method that we called to run the pipeline on a specified DataFrame. In this case, we ran the pipeline on the training data to train a model. 

# COMMAND ----------

piplineModel = pipeline.fit(train)
print("Pipeline complete!")

# COMMAND ----------

# MAGIC %md ### Test the Pipeline Model
# MAGIC The model produced by the pipeline is a transformer that will apply all of the stages in the pipeline to a specified DataFrame and apply the trained model to generate predictions. In this case, we transformed the **test** DataFrame using the pipeline to generate label predictions.

# COMMAND ----------

prediction = piplineModel.transform(test)
predicted = prediction.select("text", "prediction", "trueLabel")
predicted.show(10)

# COMMAND ----------

predicted10 = prediction.select("*")
predicted10.show(10)

# COMMAND ----------

# MAGIC %md ##TP, FP, TN, and FN all calculated
# MAGIC Precision and recall also calculated

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

# MAGIC %md ## AUC is calculated

# COMMAND ----------

gbt_evaluator =  MulticlassClassificationEvaluator(labelCol="trueLabel", predictionCol="prediction")
gbt_auc = gbt_evaluator.evaluate(prediction)

print("AUC for Logistic Regression = ", gbt_auc)

# COMMAND ----------

# MAGIC %md ## Generated AUC for Logistic Regression =  0.8916586030414259

# COMMAND ----------


