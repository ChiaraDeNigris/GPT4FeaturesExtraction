# Databricks notebook source
# MAGIC %md
# MAGIC # Statistics

# COMMAND ----------

# MAGIC %pip install --upgrade tiktoken
# MAGIC %pip install --upgrade openai

# COMMAND ----------

#%run ./Shared/common/connections

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
# MAGIC # Reading data

# COMMAND ----------

#adding a text widget to select input folder
dbutils.widgets.text(name = "Input Folder Name", defaultValue = "cleaned_output/cleaned_data")
#adding a text widget to select input file
dbutils.widgets.text(name = "Input File Name", defaultValue = "data.csv")

InputFolder = dbutils.widgets.get("Input Folder Name")
InputFile = dbutils.widgets.get("Input File Name")

# COMMAND ----------

file_path = "/"+InputFolder+"/"+InputFile

df = spark.read.csv(file_path, header=True, inferSchema=True, sep=",")
df.createOrReplaceTempView("chat")

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## General statistics 

# COMMAND ----------

#total lines
line_count = df.count()

#number of distinct conversation
query_nconversation = "SELECT COUNT(DISTINCT INTERACTION_ID) FROM chat"
n_conversation = spark.sql(query_nconversation)
count_conv = n_conversation.collect()[0][0]

#number of distinct conversation per day
query_nconversationD = """ SELECT ROUND(AVG(res.count), 2) AS avg
FROM (
    SELECT COUNT(DISTINCT INTERACTION_ID) AS count
    FROM chat 
    GROUP BY DATE
) AS res;"""
n_conversationD = spark.sql(query_nconversationD)
count_convD = n_conversationD.collect()[0][0]

#number of distinct user types
query_ntypes = "SELECT COUNT(DISTINCT USERTYPE) FROM chat"
n_types = spark.sql(query_ntypes)
count_user = n_types.collect()[0][0]

#number of human messages
query_humans = "SELECT COUNT(*) FROM chat WHERE USERTYPE='CLIENT' OR USERTYPE='AGENT' AND USERNICK != 'Live Chat'"
n_humans = spark.sql(query_humans)
count_human = n_humans.collect()[0][0]

#number of bot and system messages
query_nsystem = "SELECT COUNT(*) FROM chat WHERE USERTYPE='EXTERNAL' OR  USERNICK = 'Live Chat'"
n_sys = spark.sql(query_nsystem)
count_sys = n_sys.collect()[0][0]

#number of null messages
query_null = "SELECT COUNT(*) AS null_messages FROM chat WHERE MSGTEXT IS NULL"
n_null = spark.sql(query_null)
count_null = n_null.collect()[0][0]

# COMMAND ----------

# MAGIC %md
# MAGIC Stesso risultato in PySpark

# COMMAND ----------

from pyspark.sql.functions import count, countDistinct, avg, round

line_count = df.count()

# Number of distinct conversation
count_conv = df.select(countDistinct("INTERACTION_ID")).first()[0]

# Number of distinct conversation per day
count_convD = df.groupBy("DATE").agg(countDistinct("INTERACTION_ID").alias("count")) \
    .agg(round(avg("count"), 2).alias("avg")).first()[0]

# Number of distinct user types
count_user = df.select(countDistinct("USERTYPE")).first()[0]

# Number of human messages
count_human = df.filter((df["USERTYPE"] == "CLIENT") | (df["USERTYPE"] == "AGENT") & (df["USERNICK"] != "Live Chat")) \
    .select(count("*")).first()[0]

# Number of bot and system messages
count_sys = df.filter((df["USERTYPE"] == "EXTERNAL") | (df["USERNICK"] == "Live Chat")) \
    .select(count("*")).first()[0]

# Number of null messages
count_null = df.filter(df["MSGTEXT"].isNull()).select(count("*").alias("null_messages")).first()[0]


# COMMAND ----------

print(f"\033[1mNumber of lines in the file:\033[0m  {line_count}")
print(f"\033[1mNumber of distinct conversation:\033[0m  {count_conv}")
print(f"\033[1mAvg number of distinct conversation per day:\033[0m  {count_convD}")
print(f"\033[1mNumber of distinct user types:\033[0m  {count_user}")
print(f"\033[1mNumber of messages by humans (clients and agents):\033[0m  {count_human}")
print(f"\033[1mNumber of messages by bot (system and Live chat):\033[0m {count_sys}")


# COMMAND ----------

#number of client
query_cl = "SELECT COUNT(*) FROM chat WHERE USERTYPE='CLIENT'"
n_cl = spark.sql(query_cl)
count_cl = n_cl.collect()[0][0]

#number of agent
query_ag = "SELECT COUNT(*) FROM chat WHERE USERTYPE='AGENT' AND USERNICK != 'Live Chat'"
n_ag = spark.sql(query_ag)
count_ag = n_ag.collect()[0][0]

#number of agent human
query_agh = "SELECT COUNT(*) FROM chat WHERE USERTYPE='AGENT'"
n_agh = spark.sql(query_agh)
count_agh = n_agh.collect()[0][0]


#number of live chat
query_lc = "SELECT COUNT(*) FROM chat WHERE USERNICK = 'Live Chat'"
n_lc = spark.sql(query_lc)
count_lc = n_lc.collect()[0][0]

#number of external
query_sy = "SELECT COUNT(*) FROM chat WHERE USERTYPE='EXTERNAL'"
n_sy = spark.sql(query_sy)
count_sy = n_sy.collect()[0][0]

# COMMAND ----------

# MAGIC %md
# MAGIC Stesso risultato in PySpark

# COMMAND ----------

# Number of client
count_cl = df.filter(df["USERTYPE"] == "CLIENT").select(count("*")).first()[0]

# Number of agent
count_ag = df.filter((df["USERTYPE"] == "AGENT") & (df["USERNICK"] != "Live Chat")) \
    .select(count("*")).first()[0]

# Number of agent human
count_agh = df.filter(df["USERTYPE"] == "AGENT").select(count("*")).first()[0]

# Number of live chat
count_lc = df.filter(df["USERNICK"] == "Live Chat").select(count("*")).first()[0]

# Number of external
count_sy = df.filter(df["USERTYPE"] == "EXTERNAL").select(count("*")).first()[0]

# COMMAND ----------

print(f"\033[1mNumber of client mess in the file:\033[0m  {count_cl}")
print(f"\033[1mNumber of human agent mess in the file:\033[0m  {count_ag}")
print(f"\033[1mNumber of  agent mess in the file:\033[0m  {count_agh}")
print(f"\033[1mNumber of live chat mess in the file:\033[0m  {count_lc}")
print(f"\033[1mNumber of system mess in the file:\033[0m  {count_sy}")


# COMMAND ----------

user_types_mess = df.groupBy("USERTYPE").count()

user_types_mess.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Null messages
# MAGIC Spesso i messaggi con valore null coincidono con la richiesta di chiusura di chiamata e con la fien di quest'ultima

# COMMAND ----------

#number of total null messages
query_null = "SELECT COUNT(*) AS null_messages FROM chat WHERE MSGTEXT IS NULL"
n_null = spark.sql(query_null)
count_null = n_null.collect()[0][0]

#number of null messages which are not connected with the 'exit' from the conversation
query_realnull = "SELECT COUNT(*) FROM chat WHERE MSGTEXT IS NULL AND REASON IS NOT NULL" 
n_realnull = spark.sql(query_realnull)
count_raelnull = n_realnull.collect()[0][0]

# COMMAND ----------

# MAGIC %md
# MAGIC Stesso risultato in PySpark

# COMMAND ----------

# Number of total null messages
count_null = df.filter(df["MSGTEXT"].isNull()).select(count("*").alias("null_messages")).first()[0]

# Number of null messages which are not connected with the 'exit' from the conversation
count_realnull = df.filter((df["MSGTEXT"].isNull()) & (df["REASON"].isNotNull())) \
    .select(count("*")).first()[0]

# COMMAND ----------

print(f"\033[1mNumber of total null messages:\033[0m  {count_null}")
print(f"\033[1mNumber of real null messages:\033[0m  {count_realnull}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conversation statistics

# COMMAND ----------

# MAGIC %md
# MAGIC ### Chats without human agents

# COMMAND ----------

chats_without_agents = "SELECT COUNT(DISTINCT INTERACTION_ID) FROM chat WHERE INTERACTION_ID NOT IN (SELECT INTERACTION_ID FROM chat WHERE USERTYPE='AGENT' AND USERNICK != 'Live Chat' GROUP BY INTERACTION_ID)"
n_chat_without_ag = spark.sql(chats_without_agents)
count_chat_without_ag = n_chat_without_ag.collect()[0][0]

# COMMAND ----------

print(f"\033[1mNumber of chat without human agents:\033[0m  {count_chat_without_ag}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Chat fuori orario

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(DISTINCT INTERACTION_ID) AS ChatFuoriOrario
# MAGIC FROM chat
# MAGIC WHERE EXTRACT(HOUR FROM TIMESTAMP(TIME)) < 9 OR EXTRACT(HOUR FROM TIMESTAMP(TIME)) > 22;

# COMMAND ----------

# MAGIC %md
# MAGIC ###Duration

# COMMAND ----------

#average conversation duration
query_avgchat = "SELECT ROUND(AVG(result.CHAT_DURATION),2) FROM ( SELECT INTERACTION_ID, (MAX(TIMESHIFT) - MIN(TIMESHIFT))/ 60.0 AS CHAT_DURATION FROM chat GROUP BY  INTERACTION_ID) AS result"
avgchat = spark.sql(query_avgchat)
count_duration = avgchat.collect()[0][0]

#longest conversation duration
query_maxchat = "SELECT ROUND(MAX(result.CHAT_DURATION),2) FROM ( SELECT INTERACTION_ID, (MAX(TIMESHIFT) - MIN(TIMESHIFT))/ 60.0 AS CHAT_DURATION FROM chat GROUP BY  INTERACTION_ID) AS result"
maxchat = spark.sql(query_maxchat)
max_duration = maxchat.collect()[0][0]

#shortest conversation duration
query_minchat = "SELECT ROUND(MIN(result.CHAT_DURATION),2) FROM ( SELECT INTERACTION_ID, (MAX(TIMESHIFT) - MIN(TIMESHIFT))/ 60.0 AS CHAT_DURATION FROM chat GROUP BY  INTERACTION_ID) AS result"
minchat = spark.sql(query_minchat)
min_duration = minchat.collect()[0][0]

# COMMAND ----------

from pyspark.sql.functions import round, avg as spark_avg,max as spark_max, min as spark_min

# Average conversation duration
count_duration = df.groupBy("INTERACTION_ID").agg(
    ((spark_max("TIMESHIFT") - spark_min("TIMESHIFT")) / 60.0).alias("CHAT_DURATION")
).agg(round(spark_avg("CHAT_DURATION"), 2).alias("average_duration")).first()[0]

# Longest conversation duration
max_duration = df.groupBy("INTERACTION_ID").agg(
    ((spark_max("TIMESHIFT") - spark_min("TIMESHIFT")) / 60.0).alias("CHAT_DURATION")
).agg(round(spark_max("CHAT_DURATION"), 2).alias("longest_duration")).first()[0]

# Shortest conversation duration
min_duration = df.groupBy("INTERACTION_ID").agg(
    ((spark_max("TIMESHIFT") - spark_min("TIMESHIFT")) / 60.0).alias("CHAT_DURATION")
).agg(round(spark_min("CHAT_DURATION"), 2).alias("shortest_duration")).first()[0]


# COMMAND ----------

print(f"\033[1mAverage chat duration (in minutes):\033[0m {count_duration}")
print(f"\033[1mLongest chat duration (in minutes):\033[0m {max_duration}")
print(f"\033[1mShortes chat duration (in minutes):\033[0m {min_duration}")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT ROUND(result.CHAT_DURATION,2) FROM ( SELECT INTERACTION_ID, (MAX(TIMESHIFT) - MIN(TIMESHIFT))/ 60.0 AS CHAT_DURATION FROM chat GROUP BY  INTERACTION_ID) AS result ORDER BY result.CHAT_DURATION  DESC

# COMMAND ----------

#find the longest conversation details
import pandas as pd

#find id
query_idlongestconv = """
    SELECT result.INTERACTION_ID, ROUND(MAX(result.CHAT_DURATION), 2) AS MAX_CHAT_DURATION
    FROM (
        SELECT 
            INTERACTION_ID, 
            (MAX(TIMESHIFT) - MIN(TIMESHIFT))/60.0 AS CHAT_DURATION 
        FROM 
            chat 
        GROUP BY  
            INTERACTION_ID
    ) AS result
    GROUP BY result.INTERACTION_ID
    ORDER BY MAX_CHAT_DURATION DESC
    LIMIT 1
"""

result_df = spark.sql(query_idlongestconv).toPandas()
max_interaction_id = result_df.iloc[0]['INTERACTION_ID']

#find details
query_longestconv = f"""
    SELECT *
    FROM chat
    WHERE INTERACTION_ID = '{max_interaction_id}'
"""

print(f"Max Interaction ID: {max_interaction_id}")
longestconv_df = spark.sql(query_longestconv)

display(longestconv_df)

# COMMAND ----------

query_chatdurations = """
    SELECT ROUND((MAX(TIMESHIFT) - MIN(TIMESHIFT))/60.0,2) AS CHAT_DURATION, INTERACTION_ID 
    FROM chat 
    GROUP BY INTERACTION_ID
"""

chatdurations_df = spark.sql(query_chatdurations)
display(chatdurations_df)

# COMMAND ----------

from pyspark.sql import functions as F

#calculate standard deviation of number of messages
chatdurations_df.agg(F.stddev('CHAT_DURATION')).collect()[0][0]

# COMMAND ----------

import matplotlib.pyplot as plt

pandas_df = chatdurations_df.toPandas()

num_bins = 50
duration_range = (pandas_df['CHAT_DURATION'].min(), 40)
plt.figure(figsize=(9,6), dpi=200)
plt.hist(pandas_df['CHAT_DURATION'], bins=num_bins, range=duration_range, edgecolor='black', alpha=0.7)
plt.xlabel('Chat Duration (minutes)', fontsize=11)
plt.ylabel('Number of Interactions', fontsize=11)
plt.title('Distribution of Chat Durations', fontsize=13)
plt.savefig('chat_minutes.png')
plt.show()
#plt.savefig('chat_minutes.eps', format='eps')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Number of messagess

# COMMAND ----------

#average number of messages per conversation
query_avgmess = "SELECT ROUND(AVG(result.CHAT_MESS),2) FROM ( SELECT INTERACTION_ID, COUNT(*) AS CHAT_MESS FROM chat GROUP BY  INTERACTION_ID) AS result"
avgmess = spark.sql(query_avgmess)
count_mess = avgmess.collect()[0][0]

#max number of messages per conversation
query_maxmess = "SELECT ROUND(MAX(result.CHAT_MESS),2) FROM ( SELECT INTERACTION_ID, COUNT(*) AS CHAT_MESS FROM chat GROUP BY  INTERACTION_ID) AS result"
maxmess = spark.sql(query_maxmess)
count_max = maxmess.collect()[0][0]

#min number of messages per conversation
query_minmess = "SELECT ROUND(MIN(result.CHAT_MESS),2) FROM ( SELECT INTERACTION_ID, COUNT(*) AS CHAT_MESS FROM chat GROUP BY  INTERACTION_ID) AS result"
minmess = spark.sql(query_minmess)
count_min = minmess.collect()[0][0]

# COMMAND ----------

# Average number of messages per conversation
count_mess = df.groupBy("INTERACTION_ID").agg(count("*").alias("CHAT_MESS")) \
    .agg(round(spark_avg("CHAT_MESS"), 2).alias("average_messages")).first()[0]

# Max number of messages per conversation
count_max = df.groupBy("INTERACTION_ID").agg(count("*").alias("CHAT_MESS")) \
    .agg(round(spark_max("CHAT_MESS"), 2).alias("max_messages")).first()[0]

# Min number of messages per conversation
count_min = df.groupBy("INTERACTION_ID").agg(count("*").alias("CHAT_MESS")) \
    .agg(round(spark_min("CHAT_MESS"), 2).alias("min_messages")).first()[0]

# COMMAND ----------

print(f"\033[1mAverage number of messages per conversation:\033[0m {count_mess}")
print(f"\033[1mMax number of messages per conversation:\033[0m {count_max}")
print(f"\033[1mMin number of messages per conversation:\033[0m {count_min}")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT ROUND(result.CHAT_MESS,2) FROM ( SELECT INTERACTION_ID, COUNT(*) AS CHAT_MESS FROM chat GROUP BY  INTERACTION_ID) AS result ORDER BY result.CHAT_MESS  DESC

# COMMAND ----------

query_chatmess = """
    SELECT INTERACTION_ID, COUNT(*) AS CHAT_MESS FROM chat GROUP BY  INTERACTION_ID
"""

chatmess_df = spark.sql(query_chatmess)
#display(chatmess_df)

# COMMAND ----------

from pyspark.sql import functions as F

#calculate standard deviation of number of messages
chatmess_df.agg(F.stddev('CHAT_MESS')).collect()[0][0]

# COMMAND ----------


import matplotlib.pyplot as plt
pandas_df = chatmess_df.toPandas()

num_bins = 50
duration_range = (pandas_df['CHAT_MESS'].min(), 40)
plt.figure(figsize=(9,6), dpi=200)
plt.hist(pandas_df['CHAT_MESS'], bins=num_bins, range=duration_range, edgecolor='black', alpha=0.7)
plt.xlabel('Number of messages', fontsize=11)
plt.ylabel('Number of inteactions', fontsize=11)
plt.title('Messages distribution in the chats', fontsize= 13)
plt.savefig('chat_mess.png')
plt.show()

# COMMAND ----------

chatmess_df.createOrReplaceTempView("count")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT CHAT_MESS, COUNT(CHAT_MESS) AS occurrence_count
# MAGIC FROM count
# MAGIC GROUP BY CHAT_MESS
# MAGIC ORDER BY occurrence_count DESC
# MAGIC LIMIT 1;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tokens
# MAGIC
# MAGIC https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken

# COMMAND ----------

import tiktoken
import openai
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType

# COMMAND ----------

@udf(IntegerType())
def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    if string is None:
        return 0 
    else:
        encoding = tiktoken.get_encoding('cl100k_base')
        num_tokens = len(encoding.encode(string))
        return num_tokens

df_token_counts = df.withColumn("token_count", num_tokens_from_string(df['MSGTEXT']))
df_token_counts.createOrReplaceTempView("token")

# COMMAND ----------

#average number of tokens per conversation
query_avgtoken = "SELECT ROUND(AVG(result.count),2) FROM  (SELECT sum(token_count) AS count FROM token GROUP BY INTERACTION_ID) AS result"
avgtoken = spark.sql(query_avgtoken)
count_avgtok = avgtoken.collect()[0][0]

#total number of token
query_tokentot = "SELECT sum(token_count) FROM token"
tokentot = spark.sql(query_tokentot)
count_tokentot = tokentot.collect()[0][0]

#total number of token in human messages
query_Htokentot = "SELECT sum(token_count) FROM token WHERE USERTYPE='CLIENT' or USERTYPE='AGENT' AND USERNICK != 'Live Chat'"
Htokentot = spark.sql(query_Htokentot)
count_Htokentot = Htokentot.collect()[0][0]

#avg number of token in human messages per conversation
query_avghtoken = "SELECT ROUND(AVG(result.count),2) FROM (SELECT sum(token_count) AS count FROM token WHERE USERTYPE='CLIENT' or USERTYPE='AGENT' AND USERNICK != 'Live Chat' GROUP BY INTERACTION_ID) AS result"
query_avghtoken = spark.sql(query_avghtoken)
count_avghtoken= query_avghtoken.collect()[0][0]

# COMMAND ----------

from pyspark.sql.functions import round, sum

# Average number of tokens per conversation
count_avgtok = df_token_counts.groupBy("INTERACTION_ID").agg(sum("token_count").alias("count")) \
    .agg(round(spark_avg("count"), 2).alias("average_tokens")).first()[0]

# Total number of tokens
count_tokentot = df_token_counts.agg(sum("token_count").alias("total_tokens")).first()[0]


# COMMAND ----------

print(f"\033[1mTotal number of tokens:\033[0m {count_tokentot}")
print(f"\033[1mAverage number of tokens per conversation:\033[0m {count_avgtok}")
print(f"\033[1mTotal number of tokens in human messages:\033[0m {count_Htokentot}")
print(f"\033[1mAverage number of tokens per conversation considering only human messages:\033[0m {count_avghtoken}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## User statistics

# COMMAND ----------

# MAGIC %md
# MAGIC ### User types

# COMMAND ----------

#numero mess CLIENT
query_client = "SELECT COUNT(*) FROM chat WHERE USERTYPE='CLIENT'"
n_client = spark.sql(query_client)
count_client = n_client.collect()[0][0]

#numero mess AGENT
query_agent = "SELECT COUNT(*) FROM chat WHERE USERTYPE='AGENT'"
n_agent = spark.sql(query_agent)
count_agent = n_agent.collect()[0][0]

#numero mess EXTERNAL
query_external = "SELECT COUNT(*) FROM chat WHERE USERTYPE='EXTERNAL'"
n_ext = spark.sql(query_external)
count_ext = n_ext.collect()[0][0]

# COMMAND ----------

print(f"\033[1mNumber of Client messages:\033[0m  {count_client}")
print(f"\033[1mNumber of Agent messages:\033[0m  {count_agent}")
print(f"\033[1mNumber of External messages:\033[0m  {count_ext}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### User connection

# COMMAND ----------

#finding clients coming back
query_retclients = "SELECT USERNICK, COUNT(DISTINCT INTERACTION_ID) as NumberOfConversations FROM chat WHERE USERTYPE = 'CLIENT' GROUP BY USERNICK HAVING COUNT(DISTINCT INTERACTION_ID) > 1"

returning_df = spark.sql(query_retclients)

display(returning_df)

# COMMAND ----------

#number of retouring clients
ret_count = returning_df.count()

#number of distinct clients
distinct_clients = "SELECT COUNT(DISTINCT(USERNICK)) FROM chat WHERE USERTYPE = 'CLIENT'"
n_client_dist = spark.sql(distinct_clients)
count_client_dist = n_client_dist.collect()[0][0]

# COMMAND ----------

# Finding clients coming back and counting the total returning clients
ret_count = (df.filter((df["USERTYPE"] == "CLIENT")) \
    .groupBy("USERNICK").agg(countDistinct("INTERACTION_ID").alias("NumberOfConversations")) \
    .filter("NumberOfConversations > 1")).count()

# Number of distinct clients
count_client_dist = df.filter((df["USERTYPE"] == "CLIENT")) \
    .select(countDistinct("USERNICK")).first()[0]

# COMMAND ----------

print(f"\033[1mNumber of clients calling more than one time:\033[0m  {ret_count}")
print(f"\033[1mNumber of distinct clients:\033[0m  {count_client_dist}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Messages per month

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT EXTRACT(MONTH FROM DATE_TIME) AS mese, COUNT(DISTINCT DATE_TIME) AS message_count
# MAGIC FROM chat
# MAGIC GROUP BY EXTRACT(MONTH FROM DATE_TIME);

# COMMAND ----------

# MAGIC %md
# MAGIC ## Messages per hour

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT DISTINCT( EXTRACT(HOUR FROM DATE_TIME)) as ORE
# MAGIC FROM chat
# MAGIC ORDER BY ORE

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT  EXTRACT(HOUR FROM DATE_TIME) as ORE, DATE_TIME
# MAGIC FROM chat
# MAGIC GROUP BY ORE, DATE_TIME
# MAGIC HAVING ORE = '12'
# MAGIC ORDER BY ORE

# COMMAND ----------

query_chathour = """
    SELECT EXTRACT(HOUR FROM DATE_TIME) AS hour_of_day, COUNT(DISTINCT INTERACTION_ID) AS interaction_count
    FROM chat
    GROUP BY EXTRACT(HOUR FROM DATE_TIME)
    ORDER BY interaction_count DESC;
"""

chathour_df = spark.sql(query_chathour)

# COMMAND ----------

display(chathour_df)

# COMMAND ----------

chathour_df = chathour_df.toPandas()

# COMMAND ----------

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

plt.bar(chathour_df['hour_of_day'], chathour_df['interaction_count'], color='blue', alpha=0.7)
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Interactions')
plt.title('Distribuzione delle interazioni per ora')

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dashboard graphs

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %sql
# MAGIC --chat duration
# MAGIC SELECT AVG(result.CHAT_DURATION)
# MAGIC FROM (
# MAGIC     SELECT INTERACTION_ID, (MAX(TIMESHIFT) - MIN(TIMESHIFT))/ 60.0 AS CHAT_DURATION
# MAGIC     FROM chat
# MAGIC     GROUP BY  INTERACTION_ID) AS result;

# COMMAND ----------

# MAGIC %sql
# MAGIC --total conversation
# MAGIC SELECT COUNT(DISTINCT INTERACTION_ID) 
# MAGIC FROM chat

# COMMAND ----------

# MAGIC %sql
# MAGIC --total rows
# MAGIC SELECT COUNT(*) 
# MAGIC FROM chat
