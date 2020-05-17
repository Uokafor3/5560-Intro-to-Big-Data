# Databricks notebook source
# MAGIC %md ## CIS5560: PySpark Decision Tree Classifier in Databricks
# MAGIC 
# MAGIC ### by Team 4 (Uche, Raymond, Tofunmi and Sweta) edited on 05/15/2020
# MAGIC Tested in Runtime 6.5 (Spark 2.4.5/2.4.0 Scala 2.11) of Databricks CE

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

# MAGIC %md ## For this project, we further normalised the user_id, review_id, and business_id columns of our df_ml dataset(subset with 120000 rows) 
# MAGIC 
# MAGIC Normalised dataset is named scaled_subset 
# MAGIC 
# MAGIC scaled_subset is imported into databricks and used for Decision tree classification model
# MAGIC 
# MAGIC NOTE: Codes used for normalisation of the above listed columns are contained in thedata engineering and analysis .py & ipynb files uploaded to the github link 

# COMMAND ----------

# MAGIC %md Import the scaled_subset.csv dataset

# COMMAND ----------

# MAGIC %md ##Prepare the Data
# MAGIC First, import the libraries you will need and prepare the training and test data:

# COMMAND ----------

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
from pyspark.ml.linalg import Vectors, SparseVector
from pyspark.sql.types import DoubleType
import re

# COMMAND ----------

# MAGIC %fs ls /FileStore/tables/df_ml.csv

# COMMAND ----------

# MAGIC %md ## Create a DataFrame Schema, 
# MAGIC that should be a Table schema

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

IS_SPARK_SUBMIT_CLI = False
if IS_SPARK_SUBMIT_CLI:
    sc = SparkContext.getOrCreate()
    spark = SparkSession(sc)

# COMMAND ----------

# MAGIC %md ##Load Dataset 
# MAGIC 
# MAGIC ensure command line above: IS_SPARK_SUBMIT_CLI = False. Also remember to set it to 'True' before exporting

# COMMAND ----------

# MAGIC %md Read csv file from DBFS (Databricks File Systems)
# MAGIC 
# MAGIC ## follow the direction to read your table after upload it to Data at the left frame
# MAGIC NOTE: See above for the data type - 
# MAGIC 
# MAGIC After df_ml_csv file is added to the data of the left frame, create a table using the UI, especially, "Upload File"
# MAGIC tick header and infer schema before creating table

# COMMAND ----------

if IS_SPARK_SUBMIT_CLI:
   df_ml = spark.read.csv('df_ml.csv', inferSchema=True, header=True)
else:
    df_ml = spark.sql("SELECT * FROM scaled_subset_csv")

# COMMAND ----------

df_ml.show(5)

# COMMAND ----------

# MAGIC %md ##Create a New Dataframe with columns "user_id", "review_id", "business_id" and "stars"(label)
# MAGIC The label is the stars (stars > 2 = 1 (positive review) else: 0 (negative review)
# MAGIC 
# MAGIC These are the columns we used in building of Decision Tree Classifier Model

# COMMAND ----------

data = df_ml.select( "user_id", "review_id", "business_id", ((col("stars") > 2).cast("Double").alias("label")))

data.show(5)

# COMMAND ----------

splits = data.randomSplit([0.7, 0.3])
train = splits[0]
test = splits[1].withColumnRenamed("label", "trueLabel")
train_rows = train.count()
test_rows = test.count()
print ("Training Rows:", train_rows, " Testing Rows:", test_rows)

# COMMAND ----------

# MAGIC %md ### Build the Recommender
# MAGIC  user_id, review_id and business_id are columns we used to build the Decision Tree Classifier Model.
# MAGIC 
# MAGIC #### Latent Features
# MAGIC We can use the features to produce some sort of algorithm (**DecisionTreeClassifier**) to intelligently calculate stars(ratings) 
# MAGIC 
# MAGIC The dt class is an estimator, so you can use its **fit** method to traing a model, or you can include it in a pipeline. Rather than specifying a feature vector and as label, the dt algorithm requries user_id, review_id and business_id columns are Normalized  
# MAGIC NOTE: all columns are normalized in python jupyter notebook before dataframe was imported

# COMMAND ----------

dtassembler = VectorAssembler(inputCols=["user_id", "review_id", "business_id"], outputCol="features")
dt = DecisionTreeClassifier(labelCol="label", featuresCol="features", maxDepth=3) 
dtp = Pipeline(stages=[dtassembler, dt])

# COMMAND ----------

# MAGIC %md #### Add paramGrid and Validation

# COMMAND ----------

paramGrid = (ParamGridBuilder()
             .addGrid(dt.maxDepth, [1, 2, 6])
             .addGrid(dt.maxBins, [20, 40])
             .build())

# COMMAND ----------

# MAGIC %md ### To build a general model, _TrainValidationSplit_ is used by us 

# COMMAND ----------

dt_tvs = TrainValidationSplit(estimator=dtp, evaluator=MulticlassClassificationEvaluator(), estimatorParamMaps=paramGrid, trainRatio=0.8)

dtModel = dt_tvs.fit(train)

# COMMAND ----------

# MAGIC %md ### Test the Recommender
# MAGIC Now that we've trained the recommender, lets see how accurately it predicts known stars in the test set.

# COMMAND ----------

prediction = dtModel.transform(test)
predicted = prediction.select("features", "prediction", "trueLabel")
predicted.show(10)

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

dt_evaluator =  MulticlassClassificationEvaluator(labelCol="trueLabel", predictionCol="prediction")
#dt_evaluator =  BinaryClassificationEvaluator(labelCol="trueLabel", rawPredictionCol="prediction", metricName="areaUnderROC")
dt_auc = dt_evaluator.evaluate(prediction)

print("AUC for Decision Tree Classifier = ", dt_auc)

# COMMAND ----------

# MAGIC %md ## TrainValidationSplit AUC for Decision Tree Classifier =  0.6732488403529867

# COMMAND ----------

# MAGIC %md ## Building same model using a CrossValidator

# COMMAND ----------

from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, CrossValidator

# COMMAND ----------

# MAGIC %md ## number of folds = 5

# COMMAND ----------

# TODO: K = 2 you may test it with 5, 10
# K=2, 3, 5, 
# K= 10 takes too long
cv = CrossValidator(estimator=dtp, evaluator=BinaryClassificationEvaluator(), \
                    estimatorParamMaps=paramGrid, numFolds=5)

# the third best model
model = cv.fit(train)

# COMMAND ----------

prediction = model.transform(test)
predicted = prediction.select("features", "prediction", "trueLabel")
predicted.show(10)

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

#dt_evaluator =  MulticlassClassificationEvaluator(labelCol="trueLabel", predictionCol="prediction")
dt_evaluator =  BinaryClassificationEvaluator(labelCol="trueLabel", rawPredictionCol="prediction", metricName="areaUnderROC")
dt_auc = dt_evaluator.evaluate(prediction)

print("AUC for Decision Tree Classifier = ", dt_auc)

# COMMAND ----------

# MAGIC %md ## CrossValidator AUC for Decision Tree Classifier =  0.5

# COMMAND ----------


