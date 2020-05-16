# Databricks notebook source
# MAGIC %md ## CIS5560: PySpark Gradient Boost Classifier in Databricks
# MAGIC 
# MAGIC ### by Team 4 (Uche, Raymond, Tofunmi and Sweta) edited on 05/15/2020
# MAGIC Tested in Runtime 6.5 (Spark 2.4.5/2.4.0 Scala 2.11) of Databricks CE

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
from pyspark.sql.types import DoubleType
from pyspark.ml.linalg import Vectors, SparseVector
import re


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

# MAGIC %fs ls /FileStore/tables/df_ml.csv

# COMMAND ----------

IS_SPARK_SUBMIT_CLI = True
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
   df_ml = spark.read.csv('df_norm.csv', inferSchema=True, header=True)
else:
    df_ml = spark.sql("SELECT * FROM scaled_subset_csv")

# COMMAND ----------

df_ml.show(5)

# COMMAND ----------

# MAGIC %md ##Create a New Dataframe with columns "user_id", "review_id", "business_id" and "stars"(label)
# MAGIC The label is the stars (stars > 2 = 1 (positive review) else: 0 (negative review)
# MAGIC 
# MAGIC These are the columns we used in building of Gradient Boost Classifier Model

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

# MAGIC %md ### Build the Recommender
# MAGIC  user_id, review_id and business_id are columns we used to build the Gradient Boost Classifier Model.
# MAGIC 
# MAGIC #### Latent Features
# MAGIC We can use the features to produce some sort of algorithm (**GBTClassifier**) to intelligently calculate stars(ratings) 
# MAGIC 
# MAGIC The GBT class is an estimator, so you can use its **fit** method to traing a model, or you can include it in a pipeline. Rather than specifying a feature vector and as label, the GBT algorithm requries user_id, review_id and business_id columns are Normalized
# MAGIC NOTE: all columns are normalized in python jupyter notebook before dataframe was imported

# COMMAND ----------

gbtassembler = VectorAssembler(inputCols=["user_id", "review_id", "business_id"], outputCol="features")

# COMMAND ----------

gbt = GBTClassifier(labelCol="label", featuresCol="features", maxIter=10) 

# COMMAND ----------

gbtp = Pipeline(stages=[gbtassembler, gbt])

# COMMAND ----------

# MAGIC %md #### Add paramGrid and Validation

# COMMAND ----------

paramGrid = (ParamGridBuilder()
             .addGrid(gbt.maxDepth,[2,3,4])
             .addGrid(gbt.maxBins, [49, 52, 55])
             .addGrid(gbt.minInfoGain,[0.0, 0.1, 0.2, 0.3])
             .addGrid(gbt.stepSize,[0.05, 0.1, 0.2, 0.4])
         
             .build())


# COMMAND ----------

# MAGIC %md ### To build a general model, _TrainValidationSplit_ is used by us as it is much faster than _CrossValidator_
# MAGIC CrossValidator takes a very long time to run.

# COMMAND ----------

gbt_tvs = TrainValidationSplit(estimator=gbtp, evaluator=MulticlassClassificationEvaluator(), estimatorParamMaps=paramGrid, trainRatio=0.8)

gbtModel = gbt_tvs.fit(train)


# COMMAND ----------

# MAGIC %md ### Test the Recommender
# MAGIC Now that we've trained the recommender, lets see how accurately it predicts known stars in the test set.

# COMMAND ----------

prediction = gbtModel.transform(test)
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

# MAGIC %md ## AUC is calculated

# COMMAND ----------

gbt_evaluator =  MulticlassClassificationEvaluator(labelCol="trueLabel", predictionCol="prediction")
gbt_auc = gbt_evaluator.evaluate(prediction)

print("AUC for Gradient Boost Classifier = ", gbt_auc)

# COMMAND ----------

# MAGIC %md ## AUC for Gradient Boost Classifier =  0.675471596680073

# COMMAND ----------


