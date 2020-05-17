# Databricks notebook source
# MAGIC %md ## CIS5560: PySpark Collaborative Filtering in Databricks
# MAGIC 
# MAGIC ### by Team 4 (Uche, Raymond, Tofunmi and Sweta) edited on 05/15/2020
# MAGIC Tested in Runtime 6.5 (Spark 2.4.5/2.4.0 Scala 2.11) of Databricks CE

# COMMAND ----------

# MAGIC %md ## Collaborative Filtering
# MAGIC Collaborative filtering is a machine learning technique that predicts ratings awarded to items by users.
# MAGIC 
# MAGIC Import the ALS class
# MAGIC In this exercise, we used the Alternating Least Squares collaborative filtering algorithm to creater a recommender.

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

# MAGIC %md ##Prepare the Data
# MAGIC First, import the libraries you will need and prepare the training and test data:

# COMMAND ----------

from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, CrossValidator

from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql.functions import *

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer, VectorIndexer, MinMaxScaler
from pyspark.sql.types import StructField, StringType, IntegerType, StructType

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
    df_ml = spark.sql("SELECT * FROM df_ml_csv")

# COMMAND ----------

df_ml.na.drop()

# COMMAND ----------

df_ml.select("review_id").distinct().count()

# COMMAND ----------

# MAGIC %md ## the label column, stars is conditioned as follows:  
# MAGIC stars (stars > 2 = 1 (positive review) else: 0 (negative review)

# COMMAND ----------

df_ml = df_ml.select("user_id", "business_id", ((col("stars") > 2).cast("Double").alias("stars")))
# data = csv
df_ml.show(5)

# COMMAND ----------

df_ml.select("user_id").distinct().count()

# COMMAND ----------

df_ml.select("business_id").distinct().count()

# COMMAND ----------



# COMMAND ----------

# MAGIC %md ##Create a New Dataframe with columns "user_id", "business_id" and "stars"(Label)
# MAGIC 
# MAGIC These are the columns we used in building of ALS Model

# COMMAND ----------

data = df_ml.select("user_id", "business_id", "stars")
splits = data.randomSplit([0.7, 0.3])
train = splits[0].withColumnRenamed("stars", "label") 
test = splits[1].withColumnRenamed("stars", "trueLabel")
train_rows = train.count()
test_rows = test.count()
print ("Training Rows:", train_rows, " Testing Rows:", test_rows)

# COMMAND ----------

data.show(5)

# COMMAND ----------

# MAGIC %md ### Build the Recommender
# MAGIC In ALS, user_id and business_id are to columns used for userCol, itemCol respectively.
# MAGIC 
# MAGIC #### Latent Features
# MAGIC We can use the features to produce some sort of algorithm (**ALS**) to intelligently calculate stars(ratings) 
# MAGIC 
# MAGIC The ALS class is an estimator, so you can use its **fit** method to traing a model, or you can include it in a pipeline. Rather than specifying a feature vector and as label, the ALS algorithm requries a numeric user ID, item ID, and stars.

# COMMAND ----------

als = ALS(userCol="user_id", itemCol="business_id", ratingCol="label")
#als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="label")
#model = als.fit(train)

# COMMAND ----------

# MAGIC %md #### Add paramGrid and Validation

# COMMAND ----------

paramGrid = ParamGridBuilder() \
                    .addGrid(als.rank, [1, 5]) \
                    .addGrid(als.maxIter, [5, 10]) \
                    .addGrid(als.regParam, [0.3, 0.1]) \
                    .addGrid(als.alpha, [2.0,3.0]) \
                    .build()

# COMMAND ----------

# MAGIC %md ### To build a general model, _TrainValidationSplit_ is used by us as it is much faster than _CrossValidator_
# MAGIC CrossValidator takes a very long time to run.
# MAGIC 
# MAGIC You can run a code with __CrossValidator__ instead as follows:
# MAGIC ```
# MAGIC cv = CrossValidator(estimator=alsImplicit, estimatorParamMaps=paramGrid, evaluator=RegressionEvaluator())
# MAGIC ```

# COMMAND ----------

cv = TrainValidationSplit(estimator=als, evaluator=RegressionEvaluator(), estimatorParamMaps=paramGrid, trainRatio=0.8)

# COMMAND ----------

train.printSchema()

# COMMAND ----------

model = cv.fit(train)

# COMMAND ----------

test.printSchema()

# COMMAND ----------

# MAGIC %md ### Test the Recommender
# MAGIC Now that we've trained the recommender, lets see how accurately it predicts known stars in the test set.

# COMMAND ----------

prediction = model.transform(test)

# COMMAND ----------

prediction = model.transform(test)
# Remove NaN values from prediction (due to SPARK-14489) [1]
prediction = prediction.filter(prediction.prediction != float('nan'))

# Round floats to whole numbers
prediction = prediction.withColumn("prediction", F.abs(F.round(prediction["prediction"],0)))

#prediction.join(df_ml, "business_index").select("user_index", "prediction", "trueLabel").show(100, truncate=False)

# COMMAND ----------

prediction.show(20)

# COMMAND ----------

# MAGIC %md #### RegressionEvaluator
# MAGIC Calculate RMSE using RegressionEvaluator.
# MAGIC 
# MAGIC __NOTE:__ make sure to set [predictionCol="prediction"]

# COMMAND ----------

# RegressionEvaluator: predictionCol="prediction", metricName="rmse"
evaluator = RegressionEvaluator(labelCol="trueLabel", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(prediction)
print ("Root Mean Square Error (RMSE):", rmse)

# COMMAND ----------

# MAGIC %md ## Root Mean Square Error (RMSE): 0.6850465221305958

# COMMAND ----------


