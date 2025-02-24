# Databricks notebook source
# MAGIC %md
# MAGIC # Running 10k chat

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

#pip install --upgrade pyarrow

# COMMAND ----------

import os
import openai
from openai import AzureOpenAI
#import tiktoken

from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType

import time
import json
import pandas as pd
from pandas import json_normalize
from tqdm import tqdm

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lettura dati

# COMMAND ----------

#adding a text widget to select input folder
dbutils.widgets.text(name = "Folder", defaultValue = "cleaned_output/Data Anonymized/")
#adding a text widget to select input file
dbutils.widgets.text(name = "File", defaultValue = "data.csv")

# COMMAND ----------

Folder = dbutils.widgets.get("Folder")
File = dbutils.widgets.get("File")

file_path = "/"+Folder+"/"+File
df = spark.read.csv(file_path, header=True, inferSchema=True, sep=",")
df.createOrReplaceTempView("chat")

# COMMAND ----------

query_semitot="""SELECT *
FROM chat
WHERE INTERACTION_ID IN (
    SELECT INTERACTION_ID
    FROM chat
    GROUP BY INTERACTION_ID
    ORDER BY COUNT(*) DESC
    LIMIT 10000
)
ORDER BY INTERACTION_ID, EVENT_ID;"""

semitot_df = spark.sql(query_semitot)
semitot_df.createOrReplaceTempView("semitot_chat")

# COMMAND ----------

display(semitot_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modello

# COMMAND ----------

openaikey = dbutils.secrets.get(scope="",key="")
#openaiendpoint = dbutils.secrets.get(scope="")

client = AzureOpenAI(
    api_key=openaikey,  
    api_version="2023-10-01-preview",
    azure_endpoint = ''
    )

deployment_name='gpt4-1106-preview'

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

#prompt unico
sys_prompt_unique="""
Analizza una conversazione con il servizio clienti di un'azienda nel mercato del gioco pubblico legale. La conversazione avviene tra un cliente, una Live chat automatica e, non sempre, un operatore umano dell'azienda.
Dalla conversazione estrai nove feature con dei valori prestabiliti: risoluzione della richiesta del cliente, soddisfazione del cliente, emozione del cliente, emozione dell'operatore,  tono del cliente, tono dell'operatore, livello linguistico del cliente, riassunto della conversazione, argomento della conversazione.

### Istruzioni risoluzione richiesta ###
Ad ogni conversazione associa una feature che definisca se il problema è stato risolto o meno alla fine della conversazione. Può avere tre valori: Risolto, Non risolto, Non risolvibile. Il valore Non risolvibile deve essere applicato a quei casi in cui il problema presentato non è di competenza reale del servizio clienti, ad esempio un problema della banca, o ai casi in cui non c'è un vero problema espresso, ma solo delle lamentele generiche.

### Istruzioni soddisfazione ###
Ad ogni conversazione associa una feature che definisca la soddisfazione del cliente, deducendola dai suoi messaggi. Può avere cinque valori: Molto insoddisfatto, Insoddisfatto, Neutrale, Soddisfatto, Molto soddisfatto.

### Istruzioni emozioni ###
Ad ogni conversazione associa da una a tre emozioni provate dal cliente e da una a tre emozioni provate dall'operatore. Le possibili emozioni sono sei: Gioia, Rabbia, Sorpresa, Tristezza, Neutralità, Paura. Limitati esclusivamente a queste classi e non ad altre possibili emozioni. Le emozioni scelte devono essere quelle prevalenti nella conversazione, che traspaiono maggiormente dai messaggi.

### Istruzioni toni ###
Ad ogni conversazione associa il tono assunto dal cliente e il tono asssunto dall'operatore per comunicare durante la conversazione. Il tono può essere definito come una di queste quattro classi: Neutrale, Emotivo, Ironico, Sarcastico. Limitati esclusivamente a queste classi e non ad altri possibili toni. Il tono scelto deve essere quello prevalente nella conversazione, che quindi è maggiormente presente. 

### Istruzioni livello linguistico ###
Ad ogni conversazione associa il livello linguistico del cliente, ovvero il livello di conoscenza e applicazione delle regole della lingua italiana da parte del cliente nello scrivere i messaggi. Può avere tre valori: Scarso, Medio, Alto. Un cliente con un livello linguistico Scarso, costruisce male le frasi e non rispetta le regole grammaticali e la punteggiatura o si esprime con un dialetto regionale. Un cliente con un livello linguistico Medio, usa correttamente i costrutti della lingua italiana. Un cliente con un livello linguistico Alto, non solo usa correttamente le regole della lingua italiana, ma si esprime usando parole complesse e avverbi.

### Istruzioni Riassunto e  Argomento della conversazione###
Segui le seguenti istruzioni step by step per analizzare la conversazione e definire il topic della conversazione:

-Step 1 : Riassumi in un testo di 180 caratteri il contenuto della conversazione, concentrandoti sull'argomento di cui si è parlato.

-Step 2 : Analizza il riassunto dello Step 1 e identifica l'argomento principale del testo. Restituisci una keyword, composta da non più di due parole, che rappresenti il topic dell'intera conversazione. 

### Istruzioni per tutte le feature ###
Se non sai che valore associare ad una feature, etichettala come NA e in nessun altro modo. Ad esempio, se l'operatore umano non interviene mai nella conversazione ma c'è solo la Live Chat, l'emozione e il tono dell'operatore devono essere necessariamente NA. Se la conversazione si interrompe prima che il problema possa essere risolto, le feature sulla risoluzione della richiesta o sulla soddisfazione devono essere NA.

Formatta l'output finale come un JSON, in questo modo:{"Risoluzione_richiesta_cliente": "Classe", "Percezione_soddisfazione_cliente": "Classe", "Emozione_prevalente_cliente": "Emozione", "Emozione_prevalente_operatore": "Emozione", "Tono_conversazione_cliente": "Tono", "Tono_conversazione_operatore": "Tono", "Livello_linguistico_cliente": "Livello", "Riassunto": "riassunto", "Argomento_della_conversazione": "keyword"}. Tutte le feature devono essere scritte con la prima lettera maiuscola e sempre nello stesso modo. 

### Chat ###

"""

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sample

# COMMAND ----------

df = correct_df.toPandas()

# COMMAND ----------

df

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

selected_columns = ['INTERACTION_ID', 'USERNICK', 'USERTYPE', 'AN_MSGTEXT']
grouped_chats = df.groupby('INTERACTION_ID')[selected_columns[1:]].apply(lambda x: x.to_dict('records')).reset_index(name='chats')
grouped_chats['json_string'] = grouped_chats['chats'].apply(lambda x: convert_to_json(x))

# COMMAND ----------

from tqdm import tqdm
tqdm.pandas()

grouped_chats['result_column'] = tqdm(grouped_chats['json_string'].progress_apply(lambda x: chat_completion(sys_prompt_unique,x)))

grouped_chats[['result_column', 'success']] = grouped_chats['result_column'].apply(lambda x: parse_json_or_return_default(x)).apply(pd.Series).rename(columns={0: 'result_column', 1: 'success'})

# COMMAND ----------

df_final = pd.concat([grouped_chats, json_normalize(grouped_chats['result_column'])], axis=1)

# COMMAND ----------

columns_to_drop = ['chats', 'json_string']
df_final = df_final.drop(columns=columns_to_drop, errors='ignore')

df_final = df_final.apply(lambda x: x.astype(str))

#df_final['result_column'] = df_final["result_column"].astype('str')

df_final = spark.createDataFrame(df_final)
display(df_final)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scrivo i risultati in un CSV

# COMMAND ----------

output_path = f"/Running_v2/"

#sample
df_final.coalesce(1).write.mode("overwrite").option("header", "true").option("sep", ";").mode("overwrite").csv(output_path+ 'Feature_GPT4_turbo_pt3')
