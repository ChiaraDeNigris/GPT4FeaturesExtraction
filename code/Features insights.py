# Databricks notebook source
# MAGIC %md
# MAGIC # Features insights
# MAGIC
# MAGIC Prime analisi su 10k conversazioni annotate 

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

import pandas as pd
from pyspark.sql.functions import explode, col, trim
from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data

# COMMAND ----------

Folder_data= "cleaned_output/Data Anonymized/"
File_data="data.csv"

file_path_data = ""+Folder_data+"/"+File_data

df_total_data = spark.read.csv(file_path_data, header=True, inferSchema=True, sep=",")

df_total_data.createOrReplaceTempView("total_data")

# COMMAND ----------

file_path = ""+Folder+"/"+File

df = spark.read.csv(file_path, header=True, inferSchema=True, sep=";")

df.createOrReplaceTempView("feature")

# COMMAND ----------

display(df)

# COMMAND ----------

from pyspark.sql.functions import split, col, regexp_replace

df = df.withColumn("Emozioni_cliente", regexp_replace(col("Emozioni_cliente"), "[\\s\\'\\'\\[\\]]", ""))
df = df.withColumn("Emozioni_cliente", split(col("Emozioni_cliente"), ","))

df = df.withColumn("Emozioni_operatore", regexp_replace(col("Emozioni_operatore"), "[\\s\\'\\'\\[\\]]", ""))
df = df.withColumn("Emozioni_operatore", split(col("Emozioni_operatore"), ","))

df = df.withColumn("Emozioni cliente", regexp_replace(col("Emozioni cliente"), "[\\s\\'\\'\\[\\]]", ""))
df = df.withColumn("Emozioni cliente", split(col("Emozioni cliente"), ","))

df = df.withColumn("Emozioni operatore", regexp_replace(col("Emozioni operatore"), "[\\s\\'\\'\\[\\]]", ""))
df = df.withColumn("Emozioni operatore", split(col("Emozioni operatore"), ","))

df = df.withColumn("Argomento_conversazione", regexp_replace(col("Argomento_conversazione"), "[\\'\\'\\[\\]]", ""))
df = df.withColumn("Argomento_conversazione", split(col("Argomento_conversazione"), ","))

# COMMAND ----------

display(df)

# COMMAND ----------

df.createOrReplaceTempView("corrects")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM corrects
# MAGIC WHERE Tono_cliente = 'nan'

# COMMAND ----------

df = df.toPandas()

# COMMAND ----------

df['Tono_cliente_fixed'] = df.apply(lambda x: x['Tono cliente'] if str(x['Tono_cliente']) in ['nan', 'NaN'] else x['Tono_cliente'], axis=1)
df['Tono_operatore_fixed'] = df.apply(lambda x: x['Tono operatore'] if str(x['Tono_operatore']) in ['nan', 'NaN'] else x['Tono_operatore'], axis=1)
df['Emozioni_cliente_fixed'] = df.apply(lambda x: x['Emozioni cliente'] if (x['Emozioni_cliente']).any() in ['nan', 'NaN'] else x['Emozioni_cliente'], axis=1)
df['Emozioni_operatore_fixed'] = df.apply(lambda x: x['Emozioni operatore'] if (x['Emozioni_operatore']).any() in ['nan', 'NaN'] else x['Emozioni_operatore'], axis=1)

# COMMAND ----------

df = spark.createDataFrame(df)

# COMMAND ----------

cols = ('Tono cliente', 'Tono_cliente','Tono_operatore', 'Tono operatore', 'Emozioni_cliente', 'Emozioni cliente', 'Emozioni operatore', 'Emozioni_operatore')

df = df.drop(*cols) 

# COMMAND ----------

df = df.withColumnRenamed("Emozioni_cliente_fixed", "Emozioni_cliente")\
       .withColumnRenamed("Emozioni_operatore_fixed", "Emozioni_operatore")\
        .withColumnRenamed("Tono_cliente_fixed", "Tono_cliente")\
        .withColumnRenamed("Tono_operatore_fixed", "Tono_operatore")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conversazioni annotate male

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM feature
# MAGIC WHERE SUCCESS = 'FALSE'

# COMMAND ----------

error_df = spark.sql("""
    SELECT *
    FROM feature
    WHERE SUCCESS = 'FALSE' OR INTERACTION_ID = '00016aJC0H244PR0'
""")

errors_count = error_df.count()
#display(error_df)

# COMMAND ----------

df.createOrReplaceTempView("corrects")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature da classificare

# COMMAND ----------

#Risoluzione_richiesta_cliente
query_Risoluzione_richiesta_cliente = """ SELECT COUNT(*)
FROM corrects
WHERE Risoluzione_richiesta != 'Risolto' AND Risoluzione_richiesta != 'Non risolto' AND Risoluzione_richiesta != 'Non risolvibile' AND Risoluzione_richiesta != 'NA'"""
Risoluzione_richiesta_cliente= spark.sql(query_Risoluzione_richiesta_cliente)
Count_Risoluzione_richiesta_cliente = Risoluzione_richiesta_cliente.collect()[0][0]

#Percezione_soddisfazione_cliente
query_Percezione_soddisfazione_cliente = """ SELECT COUNT(*)
FROM corrects
WHERE Soddisfazione_cliente != 'Soddisfatto' AND Soddisfazione_cliente != 'Insoddisfatto' AND Soddisfazione_cliente != 'Neutrale' AND Soddisfazione_cliente != 'Molto insoddisfatto' AND Soddisfazione_cliente != 'Molto soddisfatto' AND Soddisfazione_cliente != 'NA';"""
Percezione_soddisfazione_cliente= spark.sql(query_Percezione_soddisfazione_cliente)
Count_Percezione_soddisfazione_cliente = Percezione_soddisfazione_cliente.collect()[0][0]

#Tono_conversazione_cliente
query_Tono_conversazione_cliente = """ SELECT COUNT(*)
FROM corrects
WHERE Tono_cliente != 'Neutrale' AND Tono_cliente != 'Emotivo' AND Tono_cliente != 'Ironico' AND Tono_cliente != 'Sarcastico' AND Tono_cliente != 'NA';"""
Tono_conversazione_cliente= spark.sql(query_Tono_conversazione_cliente)
Count_Tono_conversazione_cliente = Tono_conversazione_cliente.collect()[0][0]

#Tono_conversazione_operatore
query_Tono_conversazione_operatore = """ SELECT COUNT(*)
FROM corrects
WHERE Tono_operatore != 'Neutrale' AND Tono_operatore != 'Emotivo' AND Tono_operatore != 'Ironico' AND Tono_operatore != 'Sarcastico' AND Tono_operatore != 'NA'"""
Tono_conversazione_operatore= spark.sql(query_Tono_conversazione_operatore)
Count_Tono_conversazione_operatore = Tono_conversazione_operatore.collect()[0][0]

#Livello_linguistico_cliente
query_Livello_linguistico_cliente = """ SELECT COUNT(*)
FROM corrects
WHERE Livello_linguistico_cliente != 'Scarso' AND Livello_linguistico_cliente != 'Medio' AND Livello_linguistico_cliente != 'Alto' AND Livello_linguistico_cliente != 'NA'"""
Livello_linguistico_cliente= spark.sql(query_Livello_linguistico_cliente)
Count_Livello_linguistico_cliente = Livello_linguistico_cliente.collect()[0][0]

# COMMAND ----------

print(f"\033[1mFeature annotate con altre classi rispetto a quelle definite\033[0m")
print(f"\033[1mRisoluzione_richiesta_cliente:\033[0m  {Count_Risoluzione_richiesta_cliente}")
print(f"\033[1mPercezione_soddisfazione_cliente:\033[0m  {Count_Percezione_soddisfazione_cliente}")
print(f"\033[1mTono_conversazione_cliente:\033[0m  {Count_Tono_conversazione_cliente}")
print(f"\033[1mTono_conversazione_operatore:\033[0m  {Count_Tono_conversazione_operatore}")
print(f"\033[1mLivello_linguistico_cliente:\033[0m {Count_Livello_linguistico_cliente}")

# COMMAND ----------

#Classi generate per Tono_cliente
query_Gen_Tono_cliente = """ SELECT COUNT(DISTINCT(Tono_cliente))
FROM corrects
WHERE Tono_cliente != 'Neutrale' AND Tono_cliente != 'Emotivo' AND Tono_cliente != 'Ironico' AND Tono_cliente != 'Sarcastico' AND Tono_cliente != 'NA';"""
Gen_Tono_cliente= spark.sql(query_Gen_Tono_cliente)
Count_Gen_Tono_cliente = Gen_Tono_cliente.collect()[0][0]

print(f"\033[1mClassi distinte generate dal modello non presenti tra quelle definite \033[0m")
print(f"\033[1mNumero per Tono_cliente:\033[0m  {Count_Gen_Tono_cliente}")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT DISTINCT(Tono_cliente)
# MAGIC FROM corrects
# MAGIC WHERE Tono_cliente != 'Neutrale' AND Tono_cliente != 'Emotivo' AND Tono_cliente != 'Ironico' AND Tono_cliente != 'Sarcastico' AND Tono_cliente != 'NA';

# COMMAND ----------

# MAGIC %md
# MAGIC ## Emozioni

# COMMAND ----------

dfcorrect = df.toPandas()

# COMMAND ----------

list2 = ["Neutralità", "Rabbia", "Tristezza", "Sorpresa", "Gioia", "Paura", "NA"]

dfcorrect['Generated_Emozioni_cliente'] = dfcorrect['Emozioni_cliente'].apply(lambda x: [item for item in x if item not in list2])
dfcorrect['Generated_Emozioni_operatore'] = dfcorrect['Emozioni_operatore'].apply(lambda x: [item for item in x if item not in list2])

# COMMAND ----------

df = spark.createDataFrame(dfcorrect)
df.createOrReplaceTempView("feature")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM feature
# MAGIC WHERE SIZE(Emozioni_cliente) > 1 

# COMMAND ----------

df_exploded_em = df.select(explode(col("Generated_Emozioni_cliente")).alias("Generated_Emozioni_cliente"))

# COMMAND ----------

total_count = df_exploded_em.select("Generated_Emozioni_cliente").count()
distinct_count = df_exploded_em.select("Generated_Emozioni_cliente").distinct().count()

print("Total Count:", total_count)
print("Distinct Count:", distinct_count)

# COMMAND ----------

x= df_exploded_em.select("Generated_Emozioni_cliente").distinct()

# COMMAND ----------

frequency_counts = df_exploded_em.groupBy('Generated_Emozioni_cliente').count()
frequency_counts.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature de generare

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT ROUND(AVG(LENGTH(Riassunto_conversazione)),2) AS AVG_Lunghezza_riassunto, MAX(LENGTH(Riassunto_conversazione)) AS MAX_Lunghezza_riassunto
# MAGIC FROM feature;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT ROUND(AVG(em_cl),2) AS vg_emozioni_cliente, ROUND(AVG(em_op),2) AS avg_emozioni_operatore, ROUND(AVG(kw),2) AS avg_kw
# MAGIC FROM (
# MAGIC     SELECT  SIZE (Emozioni_cliente) as em_cl, SIZE (Emozioni_operatore) AS em_op, SIZE (Argomento_conversazione) AS kw
# MAGIC     FROM feature
# MAGIC )

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(DISTINCT (Argomento_conversazione)) As coppie_kw_distinte
# MAGIC FROM feature

# COMMAND ----------

df_exploded = df.select(explode(col("Argomento_conversazione")).alias("Argomento_conversazione"))
total_count = df_exploded.withColumn("Argomento_conversazione", trim(df_exploded.Argomento_conversazione)).count()
distinct_count = df_exploded.withColumn("Argomento_conversazione", trim(df_exploded.Argomento_conversazione)).distinct().count()

print("Total Count:", total_count)
print("Distinct Count:", distinct_count)

# COMMAND ----------

df_exploded = df_exploded.toPandas()

# COMMAND ----------

df_uni = pd.DataFrame(columns=['Argomento_conversazione'])
df_uni['Argomento_conversazione'] = df_exploded['Argomento_conversazione'].str.strip().unique()

# COMMAND ----------

rg_regi ='[Rr]egistrazion*'
rg_operatore ='[Oo]perator*'
rg_assistenza ='[Aa]ssistenza gioc*'
rg_prelievi = '[Pp]reliev*'
rg_conto ='[Cc]onto*'
rg_ric ='[Rr]icaric*'
rg_bonus = '[Bb]onus*'
rg_gioco = '[Gg]ioco [rR]esponsabile*'

#registrazione
regi = df_uni['Argomento_conversazione'][df_uni['Argomento_conversazione'].str.contains(rg_regi, case=False, regex=True)].tolist()

#assistenza
ass = df_uni['Argomento_conversazione'][df_uni['Argomento_conversazione'].str.contains(rg_assistenza, case=False, regex=True)].tolist()

#operatore
op = df_uni['Argomento_conversazione'][df_uni['Argomento_conversazione'].str.contains(rg_operatore, case=False, regex=True)].tolist()

#prelievi
prelievi = df_uni['Argomento_conversazione'][df_uni['Argomento_conversazione'].str.contains(rg_prelievi, case=False, regex=True)].tolist()

#conto g
conto = df_uni['Argomento_conversazione'][df_uni['Argomento_conversazione'].str.contains(rg_conto, case=False, regex=True)].tolist()

#ricarica
ricariche = df_uni['Argomento_conversazione'][df_uni['Argomento_conversazione'].str.contains(rg_ric, case=False, regex=True)].tolist()

# bonus
bonus = df_uni['Argomento_conversazione'][df_uni['Argomento_conversazione'].str.contains(rg_bonus, case=False, regex=True)].tolist()

#gioco responsabile
gresp = df_uni['Argomento_conversazione'][df_uni['Argomento_conversazione'].str.contains(rg_gioco, case=False, regex=True)].tolist()

# COMMAND ----------

ass

# COMMAND ----------



# COMMAND ----------

df_uni['Classe'] = df_uni['Argomento_conversazione'].map(lambda x: 
    'Bonus' if x in bonus else
    'Assistenza giochi' if x in ass else
    'Registrazione' if x in regi else
    'Operatore' if x in op else
    'Prelievi' if x in prelievi else
    'Conto' if x in conto else
    'Ricarica' if x in ricariche else
    'Gioco Responsabile' if x in gresp else
    'Altro'
)

# COMMAND ----------

df_uni['Classe'] = df_uni['Classe'].str.lower()
frequency_counts = df_uni.groupby('Classe').size().reset_index(name='count')

# COMMAND ----------

frequency_counts

# COMMAND ----------

df_uni

# COMMAND ----------

# MAGIC %md
# MAGIC ## NA

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) AS NA
# MAGIC FROM corrects
# MAGIC WHERE Risoluzione_richiesta = 'NA' AND Soddisfazione_cliente = 'NA'  AND Tono_cliente = 'NA' AND Tono_operatore = 'NA' AND Livello_linguistico_cliente = 'NA'
