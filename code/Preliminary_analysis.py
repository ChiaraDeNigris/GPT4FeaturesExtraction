# Databricks notebook source
# MAGIC %md
# MAGIC # Preliminary analysis
# MAGIC

# COMMAND ----------

service_credential = dbutils.secrets.get(scope="",key="")
service_app = dbutils.secrets.get(scope="",key="")
service_tenant = dbutils.secrets.get(scope="",key="")
service_endpoint = dbutils.secrets.get(scope="",key="")

# COMMAND ----------

spark.conf.set("", "OAuth")
spark.conf.set("", "")
spark.conf.set("", service_app)
spark.conf.set("", service_credential)
spark.conf.set("", service_endpoint)

# COMMAND ----------

dbutils.fs.ls('')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reading data
# MAGIC

# COMMAND ----------

#libraries
from pyspark.sql.functions import col,isnan,when,count

# COMMAND ----------

file_path = "./TRANSCRIPTION_CHAT.csv"
df = spark.read.csv(file_path, header=True, inferSchema=True, sep=";")

#adding header to data
header = ["DATE_TIME", "INTERACTION_ID", "USER_ID","TIMESHIFT", "VISIBILITY","EVENT_ID", "PERSON_ID","USERNICK","USERTYPE", "PROTOCOLTYPE","TZOFFSET", "CLIENTVERSION","MSGTEXT", "ASKERID","REASON" ]
df = df.toDF(*header)

#create temporary table to work with sql
df.createOrReplaceTempView("chat")

# COMMAND ----------

display(df)

# COMMAND ----------

line_count = df.count()
print(f"Number of lines in the file: {line_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Removing errors

# COMMAND ----------

#checking null values in each column
df2 = df.select([count(when(col(c).contains('None') | \
                            col(c).contains('NULL') | \
                            (col(c) == '' ) | \
                            col(c).isNull() | \
                            isnan(c), c 
                           )).alias(c)
                    for c in df.columns])
display(df2)

# COMMAND ----------

#removing null values
error_df = spark.sql("""
    SELECT *
    FROM chat
    WHERE INTERACTION_ID IS NULL OR USER_ID IS NULL OR TRY_CAST(DATE_TIME AS DATE) IS NULL
""")

errors_count = error_df.count()
#display(error_df)

# COMMAND ----------

#subtracting errors
dfcorrect=df.subtract(error_df)
correct_count = dfcorrect.count()
display(dfcorrect)

# COMMAND ----------

#checks
print(f"Number of lines in the original df: {line_count}")
print(f"Number of errors: {errors_count}")
print(f"Number of lines resulting from subtraction: {correct_count}" )

# COMMAND ----------

#checking null values in corrected dataset
df3 = dfcorrect.select([count(when(col(c).contains('None') | \
                            col(c).contains('NULL') | \
                            (col(c) == '' ) | \
                            col(c).isNull() | \
                            isnan(c), c 
                           )).alias(c)
                    for c in dfcorrect.columns])
display(df3)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Filling Dataset with User Types

# COMMAND ----------

#converting timestamp format

from pyspark.sql.functions import *
dfcorrect=dfcorrect.withColumn("DATETIME",to_timestamp(col("DATE_TIME"),"yyyy-MM-dd HH:mm:ss"))
#display(dfcorrect)

# COMMAND ----------

dfcorrect.createOrReplaceTempView("corrects")

# COMMAND ----------

df_filled = spark.sql("""
    SELECT 
        DATETIME AS DATE_TIME,
        CAST(DATE_TIME AS DATE) As DATE,
        SUBSTRING(DATE_TIME, 12, 8) AS TIME,
        INTERACTION_ID, 
        USER_ID, 
        CAST(TIMESHIFT AS INT), 
        VISIBILITY, 
        CAST(EVENT_ID AS INT) ,
        FIRST(PERSON_ID) OVER (PARTITION BY USER_ID  ORDER BY CAST(EVENT_ID AS INT)) AS PERSON_ID,
        FIRST(USERNICK) OVER (PARTITION BY USER_ID  ORDER BY CAST(EVENT_ID AS INT)) AS USERNICK,
        FIRST(USERTYPE) OVER (PARTITION BY USER_ID  ORDER BY CAST(EVENT_ID AS INT)) AS USERTYPE, 
        FIRST(PROTOCOLTYPE) OVER (PARTITION BY USER_ID  ORDER BY CAST(EVENT_ID AS INT)) AS PROTOCOLTYPE, 
        TZOFFSET, 
        FIRST(CLIENTVERSION) OVER (PARTITION BY USER_ID  ORDER BY CAST(EVENT_ID AS INT)) AS CLIENTVERSION, 
        MSGTEXT,
        ASKERID,
        REASON
    FROM corrects
    ORDER BY INTERACTION_ID, EVENT_ID
""")

df_filled.createOrReplaceTempView("filled")
display(df_filled)

# COMMAND ----------

#type(df_filled)

# COMMAND ----------

#import pandas as pd
#df_filled=df_filled.toPandas()
#df_filled['DATE_TIME'] = pd.to_datetime(df_filled['DATE_TIME']) + pd.to_timedelta(df_filled['TIMESHIFT'], unit='s')

# COMMAND ----------

df_filled

# COMMAND ----------

#df_filled=spark.createDataFrame(df_filled)

# COMMAND ----------

df_filled.createOrReplaceTempView("filled_corr")

# COMMAND ----------

filled_count = df_filled.count()
print(f"Number of lines in df filled: {filled_count}" )

# COMMAND ----------

#isolating humans interactions
df_humans = spark.sql("""
    SELECT *     
FROM filled_corr
WHERE USERTYPE='CLIENT' OR USERTYPE='AGENT' AND USERNICK != 'Live Chat'
ORDER  BY INTERACTION_ID, EVENT_ID;
""")

#isolating systems and bot interactions
df_systems = spark.sql("""
    SELECT *     
FROM filled_corr
WHERE USERTYPE='EXTERNAL' OR  USERNICK = 'Live Chat'
ORDER  BY INTERACTION_ID, EVENT_ID;
""")

# COMMAND ----------

display(df_humans)

# COMMAND ----------

display(df_systems)

# COMMAND ----------

#checks
system_count = df_systems.count()
human_count = df_humans.count()

print(f"Number of lines in the original filled df: {filled_count}")
print(f"Number of system messages: {system_count}")
print(f"Number of human messages: {human_count}" )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Writing files in CSV

# COMMAND ----------

display(df_filled)

# COMMAND ----------

output_path = f"/"

#errors
error_df.coalesce(1).write.mode("overwrite").option("header", "true").option("sep", ",").mode("overwrite").csv(output_path+ 'errors')

# COMMAND ----------

output_path = f"/"

#cleaned dataset
df_filled.coalesce(1).write.mode("overwrite").option("header", "true").option("sep", ",").mode("overwrite").csv(output_path+ 'cleaned_data')

#human messages
df_humans.coalesce(1).write.mode("overwrite").option("header", "true").option("sep", ",").mode("overwrite").csv(output_path+ 'humans_messages')

#bot messages
df_systems.coalesce(1).write.mode("overwrite").option("header", "true").option("sep", ",").mode("overwrite").csv(output_path+ 'system_messages')