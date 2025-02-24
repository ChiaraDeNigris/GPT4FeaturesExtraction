# Databricks notebook source
# MAGIC %md
# MAGIC # Test qualità annotazione

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

# MAGIC %pip install tiktoken
# MAGIC %pip install tqdm

# COMMAND ----------

import os
import openai
from openai import AzureOpenAI
import tiktoken

from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType

import sys
import tqdm

import time
import json
import pandas as pd
from pandas import json_normalize

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lettura dati

# COMMAND ----------

#adding a text widget to select input folder
dbutils.widgets.text(name = "Folder", defaultValue = "Sample/Sample2kChat")
#adding a text widget to select input file
dbutils.widgets.text(name = "File", defaultValue = "")

# COMMAND ----------

Folder = dbutils.widgets.get("Folder")
File = dbutils.widgets.get("File")

# COMMAND ----------

file_path = ""+Folder+"/"+File

df = spark.read.csv(file_path, header=True, inferSchema=True, sep=",")
df.createOrReplaceTempView("chat")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modello

# COMMAND ----------

openaikey = dbutils.secrets.get(scope="",key="")

client = AzureOpenAI(
    api_key=openaikey,  
    api_version="2023-10-01-preview",
    azure_endpoint = ''
    )

deployment_name='gpt4-1106-preview'

deployment_name3='gpt_35'

# COMMAND ----------

def chat_completion(sys_prompt, prompt):
    time.sleep(1)
    try:
        response = client.chat.completions.create(
            model=deployment_name,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0
        )   
        result = response.choices[0].message.content
        return result
    
    except Exception as ex:
        template = "Error: {0}; {1!r}"
        message = template.format(type(ex).__name__, ex.args)
        return message

# COMMAND ----------

#prompt nuovo
sys_prompt="""
Analizza una conversazione tra un cliente e un servizio clienti di un'azienda nel mercato del gioco pubblico legale.
Estrai sette feature con valori prestabiliti: Risoluzione_richiesta (Risolto/Non risolto/Non risolvibile), Soddisfazione_cliente (Molto insoddisfatto/Insoddisfatto/Neutrale/Soddisfatto/Molto soddisfatto), Emozioni_cliente (Gioia/Rabbia/Sorpresa/Tristezza/Neutralità/Paura), Emozioni_operatore (Gioia/Rabbia/Sorpresa/Tristezza/Neutralità/Paura), Tono_cliente (Neutrale/Emotivo/Ironico/Sarcastico), Tono_operatore (Neutrale/Emotivo/Ironico/Sarcastico) Livello_linguistico_cliente (Scarso/Medio/Alto). Estrai anche due feature con valori liberi: Riassunto_conversazione, Argomento_conversazione.

### Istruzioni Risoluzione_richiesta ###
Ad ogni conversazione associa una feature che definisca se il problema è stato risolto o meno alla fine della conversazione. Valori: Risolto/Non risolto/Non risolvibile. Il valore Non risolvibile deve essere applicato a quei casi in cui il problema presentato non è di competenza reale del servizio clienti o ai casi in cui non c'è un vero problema espresso ma solo delle lamentele generiche.

### Istruzioni Soddisfazione_cliente ###
Ad ogni conversazione associa una feature che definisca la soddisfazione del cliente, deducendola dai suoi messaggi. Valori: Molto insoddisfatto/Insoddisfatto/Neutrale/Soddisfatto/Molto soddisfatto.

### Istruzioni Emozioni_cliente e Emozioni_operatore ###
Ad ogni conversazione associa da una a tre emozioni provate dal cliente e da una a tre emozioni provate dall'operatore. Valori : Gioia/Rabbia/Sorpresa/Tristezza/Neutralità/Paura. Limitati esclusivamente a queste classi e non ad altre possibili emozioni. Le emozioni scelte devono essere quelle prevalenti nella conversazione, che traspaiono maggiormente dai messaggi.

### Istruzioni Tono_cliente e Tono_operatore ###
Ad ogni conversazione associa il tono prevalente assunto dal cliente e il tono asssunto dall'operatore per comunicare durante la conversazione. Valori: Neutrale/Emotivo/Ironico/Sarcastico. Limitati esclusivamente a queste classi e non ad altri possibili toni.

### Istruzioni Livello_linguistico_cliente ###
Ad ogni conversazione associa il livello linguistico del cliente, ovvero il livello di conoscenza e applicazione delle regole della lingua italiana da parte del cliente nello scrivere i messaggi. Valori: Scarso/Medio/Alto. Un cliente con un livello linguistico Scarso, costruisce male le frasi e non rispetta le regole grammaticali e la punteggiatura o si esprime con un dialetto regionale. Un cliente con un livello linguistico Medio, usa correttamente i costrutti della lingua italiana. Un cliente con un livello linguistico Alto, non solo usa correttamente le regole della lingua italiana, ma si esprime usando parole complesse.

### Istruzioni Riassunto e Argomento_conversazione###
Segui le seguenti istruzioni step by step per analizzare la conversazione e definire il topic della conversazione:

-Step 1 : Sintetizza il contenuto della conversazione in un testo di massimo 50 parole, concentrandoti  sull'argomento di cui si è parlato.

-Step 2: In base al riassunto dello Step 1 restituisci da una a due keyword che identifichino gli argomenti della conversazione. La keyword deve rappresentare in modo specifico i motivi per cui il cliente ha contattato l'assistenza e non essere generica. Assicurati che ogni keyword identifichi chiaramente il tema centrale della richiesta del cliente e non sia più lunga di 3 parole.

### Istruzioni generali ###
Se non sai che valore associare ad una feature, etichettala come NA e in nessun altro modo. Ad esempio, se l'operatore umano non interviene mai nella conversazione ma c'è solo la Live Chat, l'emozione e il tono dell'operatore devono essere necessariamente NA. Se la conversazione si interrompe prima che il problema possa essere risolto, le feature sulla risoluzione della richiesta o sulla soddisfazione devono essere NA.

Formatta l'output finale come un JSON. Tutte le feature devono essere scritte con la prima lettera maiuscola e sempre nello stesso modo. 

### Chat ###

"""

# COMMAND ----------

#prompt classi topic

#prompt nuova classificazione a 3 livelli e nuova feature
class_prompt_3lv="""
Analizza una conversazione tra un cliente e un servizio clienti di un'azienda nel mercato del gioco pubblico legale.
Estrai otto feature con valori prestabiliti: Risoluzione_richiesta (Risolto/Non risolto/Non risolvibile), Soddisfazione_cliente (Molto insoddisfatto/Insoddisfatto/Neutrale/Soddisfatto/Molto soddisfatto), Emozioni_cliente (Gioia/Rabbia/Sorpresa/Tristezza/Neutralità/Paura), Emozioni_operatore (Gioia/Rabbia/Sorpresa/Tristezza/Neutralità/Paura), Tono_cliente (Neutrale/Emotivo/Ironico/Sarcastico), Tono_operatore (Neutrale/Emotivo/Ironico/Sarcastico) Livello_linguistico_cliente (Scarso/Medio/Alto), Argomento_conversazione (Bonus/Campagna Outbound/Contatto non completato/Gestione Conto Gioco/Giochi/Loyalty/Malfunzionamento). Estrai anche due feature con valore libero: Riassunto_conversazione e Keyword.

### Istruzioni Risoluzione_richiesta ###
Ad ogni conversazione associa una feature che definisca se il problema è stato risolto o meno alla fine della conversazione. Valori: Risolto/Non risolto/Non risolvibile. Il valore Non risolvibile deve essere applicato a quei casi in cui il problema presentato non è di competenza reale del servizio clienti o ai casi in cui non c'è un vero problema espresso ma solo delle lamentele generiche.

### Istruzioni Soddisfazione_cliente ###
Ad ogni conversazione associa una feature che definisca la soddisfazione del cliente, deducendola dai suoi messaggi. Valori: Molto insoddisfatto/Insoddisfatto/Neutrale/Soddisfatto/Molto soddisfatto.

### Istruzioni Emozioni_cliente e Emozioni_operatore ###
Ad ogni conversazione associa da una a tre emozioni provate dal cliente e da una a tre emozioni provate dall'operatore. Valori : Gioia/Rabbia/Sorpresa/Tristezza/Neutralità/Paura. Limitati esclusivamente a queste classi e non ad altre possibili emozioni. Le emozioni scelte devono essere quelle prevalenti nella conversazione, che traspaiono maggiormente dai messaggi.

### Istruzioni Tono_cliente e Tono_operatore ###
Ad ogni conversazione associa il tono prevalente assunto dal cliente e il tono assunto dall'operatore per comunicare durante la conversazione. Valori: Neutrale/Emotivo/Ironico/Sarcastico. Limitati esclusivamente a queste classi e non ad altri possibili toni.

### Istruzioni Livello_linguistico_cliente ###
Ad ogni conversazione associa il livello linguistico del cliente, ovvero il livello di conoscenza e applicazione delle regole della lingua italiana da parte del cliente nello scrivere i messaggi. Valori: Scarso/Medio/Alto. Un cliente con un livello linguistico Scarso, costruisce male le frasi e non rispetta le regole grammaticali e la punteggiatura o si esprime con un dialetto regionale. Un cliente con un livello linguistico Medio, usa correttamente i costrutti della lingua italiana. Un cliente con un livello linguistico Alto, non solo usa correttamente le regole della lingua italiana, ma si esprime usando parole complesse.

### Istruzioni Riassunto, Keyword e Argomento_conversazione###
Segui le seguenti istruzioni step by step per analizzare la conversazione e definire il topic della conversazione:

-Step 1 : Sintetizza il contenuto della conversazione in un testo di massimo 50 parole, concentrandoti  sull'argomento di cui si è parlato.

-Step 2: In base al riassunto dello Step 1 restituisci da una a due keyword che identifichino gli argomenti della conversazione. La keyword deve rappresentare in modo specifico i motivi per cui il cliente ha contattato l'assistenza e non essere generica. Assicurati che ogni keyword identifichi chiaramente il tema centrale della richiesta del cliente e non sia più lunga di 3 parole. 

-Step 3: In base al riassunto dello Step 1 restituisci una tripla di classi che identifichino gli argomenti della conversazione. La prima classe è più generale, la seconda sotto classe specifica in modo più preciso la prima e la terza sotto calsse specifica in modo più preciso la seconda. La classificazione deve attenersi al contenuto di questo dizionario con i seguenti valori : 

{class_list}

La feature deve essere una lista composta in questo modo: ['Classe lv1', 'Classe lv1_Classe lv2', 'Classe lv2_Classe lv3']. Non generare altri possibili valori, se non sai che valore associare alla feature usa l'etichetta Altro.

### Istruzioni generali ###
Se non sai che valore associare ad una feature, etichettala come NA e in nessun altro modo. Ad esempio, se l'operatore umano non interviene mai nella conversazione ma c'è solo la Live Chat, l'emozione e il tono dell'operatore devono essere necessariamente NA. Se la conversazione si interrompe prima che il problema possa essere risolto, le feature sulla risoluzione della richiesta o sulla soddisfazione devono essere NA.

Formatta l'output finale come un JSON. Tutte le feature devono essere scritte con la prima lettera maiuscola. 

### Chat ###

"""

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sample

# COMMAND ----------

df = df.toPandas()

# COMMAND ----------

import re
#pattern per trovare gli interaction ID
pattern = re.compile(r'([0-9a-zA-Z]{16})')

# COMMAND ----------

labeled_ids= """
00013aJ2JUYYT6BU
00013aJ2JUYYT9F0
00013aJ2JUYYTA5X
00013aJ2JUYYTAGK
00013aJ2JUYYTB5T
00013aJ2JUYYTCQE
00013aJ2JUYYTD9A
00013aJ2JUYYTECJ
00013aJ2JUYYTEHS
00013aJ2JUYYTFSA
00013aJ2JUYYTG2A
00013aJ2JUYYTGD0
00013aJ2JUYYTJ5P
00013aJ2JUYYTNDM
00013aJ2JUYYTQS1
00013aJ2JUYYTSHA
00013aJ2JUYYTSUJ
00013aJ2JUYYTTC5
00013aJ2JUYYTTU0
00013aJ2JUYYTVGP
00013aJ2JUYYTVPQ
00013aJ2JUYYTWTX
00013aJ2JUYYTWU4
00013aJ2JUYYTY15
00013aJ2JUYYTYJJ
00013aJ2JUYYU0F1
00013aJ2JUYYU42Y
00013aJ2JUYYU91M
00013aJ2JUYYU92N
00013aJ2JUYYUB6P
00013aJ2JUYYUCRC
00013aJ2JUYYUD0Y
00013aJ2JUYYUD5S
00013aJ2JUYYUES1
00013aJ2JUYYUEUF
00013aJ2JUYYUFRR
00013aJ2JUYYUHVH
00013aJ2JUYYUKKQ
00013aJ2JUYYUNK8
00013aJ2JUYYUNPP
00013aJ2JUYYUNQD
"""

# COMMAND ----------

result_ids = re.sub(pattern, r'\1,', labeled_ids)
result_ids_list = result_ids.split(',\n')
result_ids_list= list(result_ids_list)
print(len(result_ids_list))

# COMMAND ----------

print(type(result_ids_list))

# COMMAND ----------

# Select relevant columns including 'INTERACTION_ID'
selected_columns = ['INTERACTION_ID', 'USERNICK', 'USERTYPE', 'AN_MSGTEXT']

# Filter DataFrame and select specific applications
desired_applications = result_ids_list
filtered_df = df[df['INTERACTION_ID'].isin(desired_applications)]

filtered_df

# COMMAND ----------

def convert_to_json(my_list):
    return json.dumps(my_list, ensure_ascii=False)

def parse_json_or_return_default(value):
    def safe_json_loads(x):
        try:
            return json.loads(x), True
        except json.JSONDecodeError:
            return value, False

    if not value:
        return {}, False
    
    return safe_json_loads(value)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Annotazione Sample

# COMMAND ----------

type(filtered_df)

# COMMAND ----------

# Group chats by INTERACTION_ID into arrays
grouped_chats = filtered_df.groupby('INTERACTION_ID')[selected_columns[1:]].apply(lambda x: x.to_dict('records')).reset_index(name='chats')

# Apply json.dumps() to each row of 'chats' column --the model expects string 
grouped_chats['json_string'] = grouped_chats['chats'].apply(lambda x: convert_to_json(x))

# COMMAND ----------

from tqdm import tqdm
tqdm.pandas()

grouped_chats['result_column'] = tqdm(grouped_chats['json_string'].progress_apply(lambda x: chat_completion(class_prompt_3lv,x)))

# COMMAND ----------

grouped_chats[['result_column', 'success']] = grouped_chats['result_column'].apply(lambda x: parse_json_or_return_default(x)).apply(pd.Series).rename(columns={0: 'result_column', 1: 'success'})

# COMMAND ----------

df1 = pd.concat([grouped_chats, json_normalize(grouped_chats['result_column'])], axis=1)

#droppo le colonne duplicate
columns_to_drop = ['chats', 'json_string']
df_final = df1.drop(columns=columns_to_drop, errors='ignore')

#riordino le colonne in output
#desired_column_order = ['INTERACTION_ID', 'Risoluzione_richiesta_cliente', 'Percezione_soddisfazione_cliente','Emozione_prevalente_cliente','Tono_conversazione_cliente','Livello_linguistico_cliente', 'Emozione_prevalente_operatore', 'Tono_conversazione_operatore']
#df_final = df_final[desired_column_order]

# COMMAND ----------

df_final = df_final.apply(lambda x: x.astype(str))

df_final = spark.createDataFrame(df_final)
display(df_final)

# COMMAND ----------

output_path = f"./Sample/SampleAnnotati/"

df_final.coalesce(1).write.mode("overwrite").option("header", "true").option("sep", ";").mode("overwrite").csv(output_path+ 'GPT_Annotations_3lv_Class_19_01')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conto i matches con l'annotazione

# COMMAND ----------

df_final = spark.read.csv("./data.csv", header=True, inferSchema=True, sep=";")

# COMMAND ----------

df_annotato_VK = spark.read.csv("./Sample_VK.csv", header=True, inferSchema=True, sep=";")

df_annotato_GDP = spark.read.csv("./Sample_GDP.csv", header=True, inferSchema=True, sep=";")

df_annotato_CDN = spark.read.csv("./Sample_CDN.csv", header=True, inferSchema=True, sep=";")

df_annotato_NF = spark.read.csv("./Sample_NF.csv", header=True, inferSchema=True, sep=";")

# COMMAND ----------

df_annotato_GDP = df_annotato_GDP.toPandas()
df_annotato_NF = df_annotato_NF.toPandas()
df_annotato_CDN = df_annotato_CDN.toPandas()
df_annotato_VK = df_annotato_VK.toPandas()
df_final = df_final.toPandas()

df_annotato_GDP["Emozione_prevalente_cliente"] = df_annotato_GDP["Emozione_prevalente_cliente"].str.extract(r'Emozione primaria: (.+?);')
df_annotato_GDP["Emozione_prevalente_operatore"] = df_annotato_GDP["Emozione_prevalente_operatore"].str.extract(r'Emozione primaria: (.+?);')
df_annotato_GDP["Tono_conversazione_cliente"] = df_annotato_GDP["Tono_conversazione_cliente"].replace({'Formale': 'Neutrale', 'Informale': 'Neutrale'})
df_annotato_GDP["Tono_conversazione_operatore"] = df_annotato_GDP["Tono_conversazione_operatore"].replace({'Formale': 'Neutrale', 'Informale': 'Neutrale'})

df_annotato_NF["Emozione_prevalente_cliente"] = df_annotato_NF["Emozione_prevalente_cliente"].str.extract(r'Emozione primaria: (.+?);')
df_annotato_NF["Emozione_prevalente_operatore"] = df_annotato_NF["Emozione_prevalente_operatore"].str.extract(r'Emozione primaria: (.+?);')
df_annotato_NF["Tono_conversazione_cliente"] = df_annotato_NF["Tono_conversazione_cliente"].replace({'Formale': 'Neutrale', 'Informale': 'Neutrale'})
df_annotato_NF["Tono_conversazione_operatore"] = df_annotato_NF["Tono_conversazione_operatore"].replace({'Formale': 'Neutrale', 'Informale': 'Neutrale'})

df_annotato_CDN["Emozione_prevalente_cliente"] = df_annotato_CDN["Emozione_prevalente_cliente"].str.extract(r'Emozione primaria: (.+?);')
df_annotato_CDN["Emozione_prevalente_operatore"] = df_annotato_CDN["Emozione_prevalente_operatore"].str.extract(r'Emozione primaria: (.+?);')
df_annotato_CDN["Tono_conversazione_cliente"] = df_annotato_CDN["Tono_conversazione_cliente"].replace({'Formale': 'Neutrale', 'Informale': 'Neutrale'})
df_annotato_CDN["Tono_conversazione_operatore"] = df_annotato_CDN["Tono_conversazione_operatore"].replace({'Formale': 'Neutrale', 'Informale': 'Neutrale'})

df_annotato_VK["Emozione_prevalente_cliente"] = df_annotato_VK["Emozione_prevalente_cliente"].str.extract(r'Emozione primaria: (.+?);')
df_annotato_VK["Emozione_prevalente_operatore"] = df_annotato_VK["Emozione_prevalente_operatore"].str.extract(r'Emozione primaria: (.+?);')
df_annotato_VK["Tono_conversazione_cliente"] = df_annotato_VK["Tono_conversazione_cliente"].replace({'Formale': 'Neutrale', 'Informale': 'Neutrale'})
df_annotato_VK["Tono_conversazione_operatore"] = df_annotato_VK["Tono_conversazione_operatore"].replace({'Formale': 'Neutrale', 'Informale': 'Neutrale'})

# COMMAND ----------

df_annotato_GDP

# COMMAND ----------

columns_to_drop = ['Livello_linguistico_operatore','Adeguatezza_competenza_operatore']
columns_topic = ['Riassunto_conversazione','Argomento_conversazione']

df_annotato_GDP = df_annotato_GDP.drop(columns=columns_to_drop, errors='ignore')
df_annotato_VK = df_annotato_VK.drop(columns=columns_to_drop, errors='ignore')
df_annotato_VK = df_annotato_VK.drop(0, axis=0)
df_annotato_GDP = df_annotato_GDP.drop(0, axis=0)

df_final = df_final.drop(columns=columns_topic, errors='ignore')
df_final = df_final.drop(0, axis=0)

# COMMAND ----------

df_final

# COMMAND ----------

df_final["Emozioni_operatore"] = df_final["Emozioni_operatore"].str.extract(r'\[\'(.+?)\'\]')
df_final["Emozioni_cliente"] = df_final["Emozioni_cliente"].str.split("'", expand=True)[1]

# COMMAND ----------

df_final

# COMMAND ----------

df_annotato_GDP = spark.createDataFrame(df_annotato_GDP)
df_annotato_NF = spark.createDataFrame(df_annotato_NF)
df_annotato_CDN = spark.createDataFrame(df_annotato_CDN)
df_annotato_VK = spark.createDataFrame(df_annotato_VK)

df_final= spark.createDataFrame(df_final)

# COMMAND ----------

df_annotato_VK.createOrReplaceTempView("annotazioneVK")
df_annotato_GDP.createOrReplaceTempView("annotazioneGDP")
df_annotato_CDN.createOrReplaceTempView("annotazioneCDN")
df_annotato_NF.createOrReplaceTempView("annotazioneNF")

# COMMAND ----------

df_final = df_final.withColumnRenamed("Emozioni.Cliente", "Emozioni_cliente")\
       .withColumnRenamed("Emozioni.Operatore", "Emozioni_operatore")\
        .withColumnRenamed("Tono.Cliente", "Tono_cliente")\
        .withColumnRenamed("Tono.Operatore", "Tono_operatore")


display(df_final)
df_final.createOrReplaceTempView("final")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Match esatti

# COMMAND ----------

query_VK = """SELECT SUM(res.risoluzione_diff) AS Diff_Risoluzione_richiesta_cliente, SUM(res.percezione_diff) AS Diff_Percezione_soddisfazione_cliente,  SUM(res.em_cl_diff) AS Diff_Emozione_prevalente_cliente,SUM(res.tono_cl_diff) AS Diff_Tono_conversazione_cliente, SUM(res.lv_diff) AS Diff_Livello_linguistico_cliente, SUM(res.em_op_diff) AS Diff_Emozione_prevalente_operatore, SUM(res.tono_op_diff) AS Diff_Tono_conversazione_operatore FROM( SELECT t1.INTERACTION_ID,
       CASE WHEN t1.Risoluzione_richiesta <> t2.Risoluzione_richiesta_cliente THEN 1 ELSE 0 END as risoluzione_diff,
       CASE WHEN t1.Soddisfazione_cliente <> t2.Percezione_soddisfazione_cliente THEN 1 ELSE 0 END as percezione_diff,
       CASE WHEN t1.Emozioni_cliente <> t2.Emozione_prevalente_cliente THEN 1 ELSE 0 END as em_cl_diff,
       CASE WHEN t1.Tono_cliente <> t2.Tono_conversazione_cliente THEN 1 ELSE 0 END as tono_cl_diff,
       CASE WHEN t1.Livello_linguistico_cliente <> t2.Livello_linguistico_cliente THEN 1 ELSE 0 END as lv_diff,
       CASE WHEN t1.Emozioni_operatore <> t2.Emozione_prevalente_operatore THEN 1 ELSE 0 END as em_op_diff,
       CASE WHEN t1.Tono_operatore <> t2.Tono_conversazione_operatore THEN 1 ELSE 0 END as tono_op_diff
FROM final t1
JOIN annotazioneVK t2 ON t1.INTERACTION_ID = t2.INTERACTION_ID) AS res""" 

query_GDP = """SELECT SUM(res.risoluzione_diff) AS Diff_Risoluzione_richiesta_cliente, SUM(res.percezione_diff) AS Diff_Percezione_soddisfazione_cliente,  SUM(res.em_cl_diff) AS Diff_Emozione_prevalente_cliente,SUM(res.tono_cl_diff) AS Diff_Tono_conversazione_cliente, SUM(res.lv_diff) AS Diff_Livello_linguistico_cliente, SUM(res.em_op_diff) AS Diff_Emozione_prevalente_operatore, SUM(res.tono_op_diff) AS Diff_Tono_conversazione_operatore FROM( SELECT t1.INTERACTION_ID,
       CASE WHEN t1.Risoluzione_richiesta <> t2.Risoluzione_richiesta_cliente THEN 1 ELSE 0 END as risoluzione_diff,
       CASE WHEN t1.Soddisfazione_cliente <> t2.Percezione_soddisfazione_cliente THEN 1 ELSE 0 END as percezione_diff,
       CASE WHEN t1.Emozioni_cliente <> t2.Emozione_prevalente_cliente THEN 1 ELSE 0 END as em_cl_diff,
       CASE WHEN t1.Tono_cliente <> t2.Tono_conversazione_cliente THEN 1 ELSE 0 END as tono_cl_diff,
       CASE WHEN t1.Livello_linguistico_cliente <> t2.Livello_linguistico_cliente THEN 1 ELSE 0 END as lv_diff,
       CASE WHEN t1.Emozioni_operatore <> t2.Emozione_prevalente_operatore THEN 1 ELSE 0 END as em_op_diff,
       CASE WHEN t1.Tono_operatore <> t2.Tono_conversazione_operatore THEN 1 ELSE 0 END as tono_op_diff
FROM final t1
JOIN annotazioneGDP t2 ON t1.INTERACTION_ID = t2.INTERACTION_ID) AS res"""

query_CDN = """SELECT SUM(res.risoluzione_diff) AS Diff_Risoluzione_richiesta_cliente, SUM(res.percezione_diff) AS Diff_Percezione_soddisfazione_cliente,  SUM(res.em_cl_diff) AS Diff_Emozione_prevalente_cliente,SUM(res.tono_cl_diff) AS Diff_Tono_conversazione_cliente, SUM(res.lv_diff) AS Diff_Livello_linguistico_cliente, SUM(res.em_op_diff) AS Diff_Emozione_prevalente_operatore, SUM(res.tono_op_diff) AS Diff_Tono_conversazione_operatore FROM( SELECT t1.INTERACTION_ID,
       CASE WHEN t1.Risoluzione_richiesta <> t2.Risoluzione_richiesta_cliente THEN 1 ELSE 0 END as risoluzione_diff,
       CASE WHEN t1.Soddisfazione_cliente <> t2.Percezione_soddisfazione_cliente THEN 1 ELSE 0 END as percezione_diff,
       CASE WHEN t1.Emozioni_cliente <> t2.Emozione_prevalente_cliente THEN 1 ELSE 0 END as em_cl_diff,
       CASE WHEN t1.Tono_cliente <> t2.Tono_conversazione_cliente THEN 1 ELSE 0 END as tono_cl_diff,
       CASE WHEN t1.Livello_linguistico_cliente <> t2.Livello_linguistico_cliente THEN 1 ELSE 0 END as lv_diff,
       CASE WHEN t1.Emozioni_operatore <> t2.Emozione_prevalente_operatore THEN 1 ELSE 0 END as em_op_diff,
       CASE WHEN t1.Tono_operatore <> t2.Tono_conversazione_operatore THEN 1 ELSE 0 END as tono_op_diff
FROM final t1
JOIN annotazioneCDN t2 ON t1.INTERACTION_ID = t2.INTERACTION_ID) AS res"""

query_NF = """SELECT SUM(res.risoluzione_diff) AS Diff_Risoluzione_richiesta_cliente, SUM(res.percezione_diff) AS Diff_Percezione_soddisfazione_cliente,  SUM(res.em_cl_diff) AS Diff_Emozione_prevalente_cliente,SUM(res.tono_cl_diff) AS Diff_Tono_conversazione_cliente, SUM(res.lv_diff) AS Diff_Livello_linguistico_cliente, SUM(res.em_op_diff) AS Diff_Emozione_prevalente_operatore, SUM(res.tono_op_diff) AS Diff_Tono_conversazione_operatore FROM( SELECT t1.INTERACTION_ID,
       CASE WHEN t1.Risoluzione_richiesta <> t2.Risoluzione_richiesta_cliente THEN 1 ELSE 0 END as risoluzione_diff,
       CASE WHEN t1.Soddisfazione_cliente <> t2.Percezione_soddisfazione_cliente THEN 1 ELSE 0 END as percezione_diff,
       CASE WHEN t1.Emozioni_cliente <> t2.Emozione_prevalente_cliente THEN 1 ELSE 0 END as em_cl_diff,
       CASE WHEN t1.Tono_cliente <> t2.Tono_conversazione_cliente THEN 1 ELSE 0 END as tono_cl_diff,
       CASE WHEN t1.Livello_linguistico_cliente <> t2.Livello_linguistico_cliente THEN 1 ELSE 0 END as lv_diff,
       CASE WHEN t1.Emozioni_operatore <> t2.Emozione_prevalente_operatore THEN 1 ELSE 0 END as em_op_diff,
       CASE WHEN t1.Tono_operatore <> t2.Tono_conversazione_operatore THEN 1 ELSE 0 END as tono_op_diff
FROM final t1
JOIN annotazioneNF t2 ON t1.INTERACTION_ID = t2.INTERACTION_ID) AS res"""

# COMMAND ----------

#GPT 4 turbo
annotazione_VK = spark.sql(query_VK)
annotazione_GDP = spark.sql(query_GDP)
annotazione_CDN = spark.sql(query_CDN)
annotazione_NF = spark.sql(query_NF)

# COMMAND ----------

from pyspark.sql.functions import lit

annotatore_col = lit("VK")
annotazione_VK = annotazione_VK.withColumn("Annotatore", annotatore_col)
annotatore_col = lit("GDP")
annotazione_GDP = annotazione_GDP.withColumn("Annotatore", annotatore_col)
annotatore_col = lit("CDN")
annotazione_CDN = annotazione_CDN.withColumn("Annotatore", annotatore_col)
annotatore_col = lit("NF")
annotazione_NF = annotazione_NF.withColumn("Annotatore", annotatore_col)

result_df = annotazione_VK.union(annotazione_GDP).union(annotazione_CDN).union(annotazione_NF)

# COMMAND ----------

column_order = ['Annotatore', 'Diff_Risoluzione_richiesta_cliente', 'Diff_Percezione_soddisfazione_cliente','Diff_Emozione_prevalente_cliente','Diff_Tono_conversazione_cliente','Diff_Livello_linguistico_cliente', 'Diff_Emozione_prevalente_operatore', 'Diff_Tono_conversazione_operatore']

result_df = result_df.select(column_order)

# COMMAND ----------

#calcolo la media dei mismatch

from pyspark.sql.functions import avg

columns=result_df.columns
avg_values = [avg(column).alias(f'avg_{column}') for column in columns]
average_row = result_df.agg(*avg_values)
result_df = result_df.union(average_row)

# COMMAND ----------

display(result_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Trovo i campi con le differenze

# COMMAND ----------

diff_VK="""
SELECT res.INTERACTION_ID, res.risoluzione_diff, res.percezione_diff, res.em_cl_diff, res.tono_cl_diff, res.lv_diff, res.em_op_diff, res.tono_op_dif
FROM(
SELECT t1.INTERACTION_ID,
       CASE WHEN t1.Risoluzione_richiesta <> t2.Risoluzione_richiesta_cliente THEN 'VK' ELSE 0 END as risoluzione_diff,
       CASE WHEN t1.Soddisfazione_cliente <> t2.Percezione_soddisfazione_cliente THEN 'VK' ELSE 0 END as percezione_diff,
       CASE WHEN t1.Emozioni_cliente <> t2.Emozione_prevalente_cliente THEN 'VK' ELSE 0 END as em_cl_diff,
       CASE WHEN t1.Tono_cliente <> t2.Tono_conversazione_cliente THEN 'VK' ELSE 0 END as tono_cl_diff,
       CASE WHEN t1.Livello_linguistico_cliente <> t2.Livello_linguistico_cliente THEN 'VK' ELSE 0 END as lv_diff,
       CASE WHEN t1.Emozioni_operatore <> t2.Emozione_prevalente_operatore THEN 'VK' ELSE 0 END as em_op_diff,
       CASE WHEN t1.Tono_operatore <> t2.Tono_conversazione_operatore THEN 'VK' ELSE 0 END as tono_op_dif
FROM final t1
JOIN annotazioneVK t2 ON t1.INTERACTION_ID = t2.INTERACTION_ID) AS res
"""

diff_GDP="""
SELECT res.INTERACTION_ID, res.risoluzione_diff, res.percezione_diff, res.em_cl_diff, res.tono_cl_diff, res.lv_diff, res.em_op_diff, res.tono_op_dif
FROM(
SELECT t1.INTERACTION_ID,
       CASE WHEN t1.Risoluzione_richiesta <> t2.Risoluzione_richiesta_cliente THEN 'GDP' ELSE 0 END as risoluzione_diff,
       CASE WHEN t1.Soddisfazione_cliente <> t2.Percezione_soddisfazione_cliente THEN 'GDP' ELSE 0 END as percezione_diff,
       CASE WHEN t1.Emozioni_cliente <> t2.Emozione_prevalente_cliente THEN 'GDP' ELSE 0 END as em_cl_diff,
       CASE WHEN t1.Tono_cliente <> t2.Tono_conversazione_cliente THEN 'GDP' ELSE 0 END as tono_cl_diff,
       CASE WHEN t1.Livello_linguistico_cliente <> t2.Livello_linguistico_cliente THEN 'GDP' ELSE 0 END as lv_diff,
       CASE WHEN t1.Emozioni_operatore <> t2.Emozione_prevalente_operatore THEN' GDP' ELSE 0 END as em_op_diff,
       CASE WHEN t1.Tono_operatore <> t2.Tono_conversazione_operatore THEN 'GDP' ELSE 0 END as tono_op_dif
FROM final t1
JOIN annotazioneGDP t2 ON t1.INTERACTION_ID = t2.INTERACTION_ID) AS res
"""

diff_CDN="""
SELECT res.INTERACTION_ID, res.risoluzione_diff, res.percezione_diff, res.em_cl_diff, res.tono_cl_diff, res.lv_diff, res.em_op_diff, res.tono_op_dif
FROM(
SELECT t1.INTERACTION_ID,
       CASE WHEN t1.Risoluzione_richiesta <> t2.Risoluzione_richiesta_cliente THEN 'CDN' ELSE 0 END as risoluzione_diff,
       CASE WHEN t1.Soddisfazione_cliente <> t2.Percezione_soddisfazione_cliente THEN 'CDN' ELSE 0 END as percezione_diff,
       CASE WHEN t1.Emozioni_cliente <> t2.Emozione_prevalente_cliente THEN 'CDN' ELSE 0 END as em_cl_diff,
       CASE WHEN t1.Tono_cliente <> t2.Tono_conversazione_cliente THEN 'CDN' ELSE 0 END as tono_cl_diff,
       CASE WHEN t1.Livello_linguistico_cliente <> t2.Livello_linguistico_cliente THEN 'CDN' ELSE 0 END as lv_diff,
       CASE WHEN t1.Emozioni_operatore <> t2.Emozione_prevalente_operatore THEN 'CDN' ELSE 0 END as em_op_diff,
       CASE WHEN t1.Tono_operatore <> t2.Tono_conversazione_operatore THEN 'CDN' ELSE 0 END as tono_op_dif
FROM final t1
JOIN annotazioneCDN t2 ON t1.INTERACTION_ID = t2.INTERACTION_ID) AS res
"""

diff_NF="""
SELECT res.INTERACTION_ID, res.risoluzione_diff, res.percezione_diff, res.em_cl_diff, res.tono_cl_diff, res.lv_diff, res.em_op_diff, res.tono_op_dif
FROM(
SELECT t1.INTERACTION_ID,
       CASE WHEN t1.Risoluzione_richiesta <> t2.Risoluzione_richiesta_cliente THEN 'NF' ELSE 0 END as risoluzione_diff,
       CASE WHEN t1.Soddisfazione_cliente <> t2.Percezione_soddisfazione_cliente THEN 'NF' ELSE 0 END as percezione_diff,
       CASE WHEN t1.Emozioni_cliente <> t2.Emozione_prevalente_cliente THEN 'NF' ELSE 0 END as em_cl_diff,
       CASE WHEN t1.Tono_cliente <> t2.Tono_conversazione_cliente THEN 'NF' ELSE 0 END as tono_cl_diff,
       CASE WHEN t1.Livello_linguistico_cliente <> t2.Livello_linguistico_cliente THEN 'NF' ELSE 0 END as lv_diff,
       CASE WHEN t1.Emozioni_operatore <> t2.Emozione_prevalente_operatore THEN 'NF' ELSE 0 END as em_op_diff,
       CASE WHEN t1.Tono_operatore <> t2.Tono_conversazione_operatore THEN 'NF' ELSE 0 END as tono_op_dif
FROM final t1
JOIN annotazioneNF t2 ON t1.INTERACTION_ID = t2.INTERACTION_ID) AS res
"""

# COMMAND ----------

diff_VK = spark.sql(diff_VK)
diff_GDP = spark.sql(diff_GDP)
diff_CDN = spark.sql(diff_CDN)
diff_NF = spark.sql(diff_NF)

# COMMAND ----------

display(diff_NF)

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import collect_list

diff_df = diff_VK.union(diff_GDP).union(diff_CDN).union(diff_NF)
final_df = diff_df.groupBy("INTERACTION_ID").agg(collect_list("risoluzione_diff").alias("risoluzione_richiesta"),collect_list("percezione_diff").alias("percezione_soddisfazione"),collect_list("em_cl_diff").alias("emozione_cliente"),collect_list("tono_cl_diff").alias("tono_cliente"),collect_list("lv_diff").alias("livello_linguistico"),collect_list("em_op_diff").alias("emozione_operatore"),collect_list("tono_op_dif").alias("tono_operatore"))

# Show the result
display(final_df)
