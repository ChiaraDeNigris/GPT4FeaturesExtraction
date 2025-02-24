# Databricks notebook source
# MAGIC %md
# MAGIC # Preprocessing

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

# COMMAND ----------

#adding a text widget to select input folder
dbutils.widgets.text(name = "Input Folder Name", defaultValue = "cleaned_output/cleaned_data")
#adding a text widget to select input file
dbutils.widgets.text(name = "Input File Name", defaultValue = "")

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
# MAGIC # Vecchio metodo

# COMMAND ----------

# MAGIC %md
# MAGIC ## Searching for sensible data

# COMMAND ----------

from pyspark.sql.functions import col, regexp_extract
from functools import reduce

#regex dictionary
patterns = {
    'CODICE_FISCALE': r'\b(?:[A-Za-z][AEIOUaeiou][AEIOUXaeioux]|[AEIOUaeiou]X{2}|[B-DF-HJ-NP-TV-Zb-df-hj-np-tv-z]{2}[A-Za-z]){2}(?:[\dLMNP-Vlmnp-v]{2}(?:[A-EHLMPR-Ta-ehlmpr-t](?:[04LQ][1-9MNP-Vlmnp-v]|[15MR][\dLMNP-Vlmnp-v]|[26NS][0-8LMNP-Ulmnp-u])|[DHPSdhps][37PTpt][0L]|[ACELMRTacelmr-t][37PTpt][01LM]|[AC-EHLMPR-Tac-ehlmpr-t][26NSns][9Vv])|(?:[02468LNQSUlnqsu][048LQU048lqu]|[13579MPRTV13579mprtv][26NS26ns])B[26NS26ns][9Vv])(?:[A-MZa-mz][1-9MNP-Vlmnp-v][\dLMNP-Vlmnp-v]{2}|[A-Ma-m][0L](?:[1-9MNP-Vlmnp-v][\dLMNP-Vlmnp-v]|[0L][1-9MNP-Vlmnp-v]))[A-Za-z]\b',
    'IBAN': r'\b[A-Za-z]{2}[0-9]{2}(?:[ ]?[0-9]{4}){4}(?!(?:[ ]?[0-9]){3})(?:[ ]?[0-9]{1,2})?\b',
    'VISA': r'\b4[0-9]{12}(?:[0-9]{3})?\b',
    'MASTERCARD': r'\b(5[1-5][0-9]{14}|2(22[1-9][0-9]{12}|2[3-9][0-9]{13}|[3-6][0-9]{14}|7[0-1][0-9]{13}|720[0-9]{12}))\b',
    'AMEX':r'\b3[47][0-9]{13}\b',
    'MAESTRO':r'\b(5018|5020|5038|6304|6759|6761|6763)[0-9]{8,15}\b',
    'VISA_MASTERCARD':r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14})\b',
    'PHONE_NUMBER': r'\b[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}\b',
    'LINK':r'(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])'
}

#create a df with a flag column for each pattern found
df_with_sensitive_data = reduce(lambda df, col_name: df.withColumn(col_name, regexp_extract(col('MSGTEXT'), patterns[col_name], 0) != ''), patterns.keys(), df)

df_with_sensitive_data.createOrReplaceTempView("sensitive")

# COMMAND ----------

display(df_with_sensitive_data)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM sensitive
# MAGIC WHERE PHONE_NUMBER=TRUE

# COMMAND ----------

# MAGIC %md
# MAGIC ## Counting sensitive data found

# COMMAND ----------

#number of senstive data found
query_sensitive = " SELECT COUNT(*)FROM sensitive WHERE CODICE_FISCALE = TRUE OR IBAN = TRUE OR VISA = TRUE OR MASTERCARD = TRUE OR AMEX = TRUE OR MAESTRO = TRUE OR VISA_MASTERCARD = TRUE OR PHONE_NUMBER = TRUE OR EMAIL = TRUE"
n_sens = spark.sql(query_sensitive)
count_sens = n_sens.collect()[0][0]

#number of credit card number found
query_creditcard = " SELECT COUNT(*)FROM sensitive WHERE  VISA = TRUE OR MASTERCARD = TRUE OR AMEX = TRUE OR MAESTRO = TRUE OR VISA_MASTERCARD = TRUE"
n_cc = spark.sql(query_creditcard)
count_cc = n_cc.collect()[0][0]

#number of phone numbers found
query_phone = " SELECT COUNT(*)FROM sensitive WHERE PHONE_NUMBER = TRUE"
n_phone = spark.sql(query_phone)
count_phone = n_phone.collect()[0][0]

#number of emails found
query_email = " SELECT COUNT(*)FROM sensitive WHERE EMAIL = TRUE"
n_email = spark.sql(query_email)
count_email = n_email.collect()[0][0]

#number of IBAN found
query_iban = " SELECT COUNT(*)FROM sensitive WHERE IBAN = TRUE"
n_iban = spark.sql(query_iban)
count_iban = n_iban.collect()[0][0]

#numebr of codici fiscali found
query_cf = " SELECT COUNT(*)FROM sensitive WHERE CODICE_FISCALE = TRUE"
n_cf = spark.sql(query_cf)
count_cf = n_cf.collect()[0][0]

#numebr of links found
query_l = " SELECT COUNT(*)FROM sensitive WHERE LINK = TRUE"
n_l = spark.sql(query_l)
count_l = n_l.collect()[0][0]

# COMMAND ----------

print(f"\033[1mNumber of senstive data:\033[0m {count_sens}")
print(f"\033[1mNumber of credit card numbers:\033[0m {count_cc}")
print(f"\033[1mNumber of phone numbers:\033[0m {count_phone}")
print(f"\033[1mNumber of IBAN:\033[0m {count_iban}" )
print(f"\033[1mNumber of emails:\033[0m {count_email}" )
print(f"\033[1mNumber of codici fiscali:\033[0m {count_cf}" )
print(f"\033[1mNumber of links:\033[0m {count_l}" )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Creating anonymized DF

# COMMAND ----------

#replace MSGTEXT column with an anonymized version
df_anonymized = spark.sql("""
    SELECT DATE_TIME, DATE, TIME, INTERACTION_ID, USER_ID,TIMESHIFT, VISIBILITY,EVENT_ID, PERSON_ID,USERNICK,USERTYPE,PROTOCOLTYPE,TZOFFSET, CLIENTVERSION, ASKERID,REASON,
    CASE 
        WHEN CODICE_FISCALE = TRUE THEN 'Codice Fiscale'
        WHEN IBAN = TRUE THEN 'IBAN'
        WHEN VISA = TRUE THEN 'Numero carta di credito'
        WHEN MASTERCARD = TRUE THEN 'Numero carta di credito'
        WHEN AMEX = TRUE THEN 'Numero carta di credito'
        WHEN MAESTRO = TRUE THEN 'Numero carta di credito'
        WHEN VISA_MASTERCARD = TRUE THEN 'Numero carta di credito'
        WHEN PHONE_NUMBER = TRUE THEN 'Numero di telefono'
        WHEN EMAIL = TRUE THEN 'Email'
        WHEN LINK = TRUE THEN 'Link'
        ELSE MSGTEXT 
    END AS AN_MSGTEXT   
    FROM sensitive
""")

df_anonymized.createOrReplaceTempView("anonymized")
#display(df_anonymized)

# COMMAND ----------

an_count=df_anonymized.count()
print(f"Number of lines in the anonimized df: {an_count}")

# COMMAND ----------

output_path = f""

#data anonymized
df_anonymized.coalesce(1).write.mode("overwrite").option("header", "true").option("sep", ";").mode("overwrite").csv(output_path+ 'Data Anonymized_new')

# COMMAND ----------

# MAGIC %md
# MAGIC # Nuova prova

# COMMAND ----------

from pyspark.sql.functions import col, when, regexp_replace

# Define the sensitive data patterns
CODICE_FISCALE = r'\b(?:[A-Za-z][AEIOUaeiou][AEIOUXaeioux]|[AEIOUaeiou]X{2}|[B-DF-HJ-NP-TV-Zb-df-hj-np-tv-z]{2}[A-Za-z]){2}(?:[\dLMNP-Vlmnp-v]{2}(?:[A-EHLMPR-Ta-ehlmpr-t](?:[04LQ][1-9MNP-Vlmnp-v]|[15MR][\dLMNP-Vlmnp-v]|[26NS][0-8LMNP-Ulmnp-u])|[DHPSdhps][37PTpt][0L]|[ACELMRTacelmr-t][37PTpt][01LM]|[AC-EHLMPR-Tac-ehlmpr-t][26NSns][9Vv])|(?:[02468LNQSUlnqsu][048LQU048lqu]|[13579MPRTV13579mprtv][26NS26ns])B[26NS26ns][9Vv])(?:[A-MZa-mz][1-9MNP-Vlmnp-v][\dLMNP-Vlmnp-v]{2}|[A-Ma-m][0L](?:[1-9MNP-Vlmnp-v][\dLMNP-Vlmnp-v]|[0L][1-9MNP-Vlmnp-v]))[A-Za-z]\b'
EMAIL = r'\b(?!.*lottomatica)[\w\-\.]+@([\w-]+\.)+[\w-]{2,}\b'
IBAN = r'\b[A-Za-z]{2}[0-9]{2}(?:[ ]?[0-9]{4}){4}(?!(?:[ ]?[0-9]){3})(?:[ ]?[0-9]{1,2})?\b'
VISA = r'\b4[0-9]{12}(?:[0-9]{3})?\b'
MASTERCARD = r'\b(5[1-5][0-9]{14}|2(22[1-9][0-9]{12}|2[3-9][0-9]{13}|[3-6][0-9]{14}|7[0-1][0-9]{13}|720[0-9]{12}))\b'
AMEX = r'\b3[47][0-9]{13}\b'
MAESTRO = r'\b(5018|5020|5038|6304|6759|6761|6763)[0-9]{8,15}\b'
VISA_MASTERCARD = r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14})\b'
PHONE_NUMBER = r'\b[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}\b'
LINK = r'(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])'

# Replace sensitive content with a generic string and add a "Pattern" column
df = df.withColumn("AN_MSGTEXT", col("MSGTEXT"))
df = df.withColumn(
    "Pattern",
    when(col("MSGTEXT").rlike(CODICE_FISCALE), "Codice Fiscale")
    .when(col("MSGTEXT").rlike(EMAIL), "Email")
    .when(col("MSGTEXT").rlike(IBAN), "IBAN")
    .when(col("MSGTEXT").rlike(VISA) | col("MSGTEXT").rlike(MASTERCARD) | col("MSGTEXT").rlike(AMEX) | col("MSGTEXT").rlike(MAESTRO) | col("MSGTEXT").rlike(VISA_MASTERCARD), "Numero carta di credito")
    .when(col("MSGTEXT").rlike(PHONE_NUMBER), "Numero di telefono")
    .when(col("MSGTEXT").rlike(LINK), "Link")
    .otherwise('False')
)

# Iterate through sensitive data patterns and replace them with generic strings
patterns = [
    (CODICE_FISCALE, "Codice Fiscale"),
    (EMAIL, "Email"),
    (IBAN, "IBAN"),
    (VISA, "Numero carta di credito"),
    (MASTERCARD, "Numero carta di credito"),
    (AMEX, "Numero carta di credito"),
    (MAESTRO, "Numero carta di credito"),
    (VISA_MASTERCARD, "Numero carta di credito"),
    (PHONE_NUMBER, "Numero di telefono"),
    (LINK, "Link")
]

for pattern, replacement in patterns:
    df = df.withColumn("AN_MSGTEXT", regexp_replace(col("AN_MSGTEXT"), pattern, replacement))

# COMMAND ----------

display(df)

# COMMAND ----------

df.createOrReplaceTempView('an')

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT AN_MSGTEXT, MSGTEXT, Pattern
# MAGIC FROM an
# MAGIC WHERE Pattern != 'No Pattern Found'

# COMMAND ----------

from pyspark.sql.functions import col

column_name = "Pattern"
value_counts = df.groupBy(column_name).count()

value_counts.show()

# COMMAND ----------

df = df.drop("Pattern")

# COMMAND ----------

output_path = f"./cleaned_output/"

#data anonymized
df.coalesce(1).write.mode("overwrite").option("header", "true").option("sep", ";").mode("overwrite").csv(output_path+ 'Data Anonymized_new')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Con Pandas

# COMMAND ----------

import pandas as pd
from tqdm import tqdm

df = df.toPandas()
df = df.apply(lambda x: x.astype(str))

#Pattern per trovare i dati sensibili
CODICE_FISCALE = r'\b(?:[A-Za-z][AEIOUaeiou][AEIOUXaeioux]|[AEIOUaeiou]X{2}|[B-DF-HJ-NP-TV-Zb-df-hj-np-tv-z]{2}[A-Za-z]){2}(?:[\dLMNP-Vlmnp-v]{2}(?:[A-EHLMPR-Ta-ehlmpr-t](?:[04LQ][1-9MNP-Vlmnp-v]|[15MR][\dLMNP-Vlmnp-v]|[26NS][0-8LMNP-Ulmnp-u])|[DHPSdhps][37PTpt][0L]|[ACELMRTacelmr-t][37PTpt][01LM]|[AC-EHLMPR-Tac-ehlmpr-t][26NSns][9Vv])|(?:[02468LNQSUlnqsu][048LQU048lqu]|[13579MPRTV13579mprtv][26NS26ns])B[26NS26ns][9Vv])(?:[A-MZa-mz][1-9MNP-Vlmnp-v][\dLMNP-Vlmnp-v]{2}|[A-Ma-m][0L](?:[1-9MNP-Vlmnp-v][\dLMNP-Vlmnp-v]|[0L][1-9MNP-Vlmnp-v]))[A-Za-z]\b'
EMAIL = r'\b(?!.*lottomatica)[\w\-\.]+@([\w-]+\.)+[\w-]{2,}\b'
IBAN= r'\b[A-Za-z]{2}[0-9]{2}(?:[ ]?[0-9]{4}){4}(?!(?:[ ]?[0-9]){3})(?:[ ]?[0-9]{1,2})?\b'
VISA= r'\b4[0-9]{12}(?:[0-9]{3})?\b'
MASTERCARD= r'\b(5[1-5][0-9]{14}|2(22[1-9][0-9]{12}|2[3-9][0-9]{13}|[3-6][0-9]{14}|7[0-1][0-9]{13}|720[0-9]{12}))\b'
AMEX= r'\b3[47][0-9]{13}\b'
MAESTRO= r'\b(5018|5020|5038|6304|6759|6761|6763)[0-9]{8,15}\b'
VISA_MASTERCARD= r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14})\b'
PHONE_NUMBER= r'\b[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}\b'
LINK= r'(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])'

#funzione per rimpiazzare i dati sensibili con una stringa generica in base al pattern
def replace_sensitive(row):
    if pd.Series(row['MSGTEXT']).str.contains(CODICE_FISCALE, regex=True).any():
        return pd.Series(row['MSGTEXT']).str.replace(CODICE_FISCALE, 'Codice Fiscale', regex=True).values[0]
    elif pd.Series(row['MSGTEXT']).str.contains(EMAIL, regex=True).any():
        return pd.Series(row['MSGTEXT']).str.replace(EMAIL, 'Email', regex=True).values[0]
    elif pd.Series(row['MSGTEXT']).str.contains(IBAN, regex=True).any():
        return pd.Series(row['MSGTEXT']).str.replace(IBAN, 'IBAN', regex=True).values[0]
    elif pd.Series(row['MSGTEXT']).str.contains(PHONE_NUMBER, regex=True).any():
        return pd.Series(row['MSGTEXT']).str.replace(PHONE_NUMBER, 'Numero di telefono', regex=True).values[0]
    elif pd.Series(row['MSGTEXT']).str.contains(LINK, regex=True).any():
        return pd.Series(row['MSGTEXT']).str.replace(LINK, 'Link', regex=True).values[0]
    elif pd.Series(row['MSGTEXT']).str.contains(VISA, regex=True).any():
        return pd.Series(row['MSGTEXT']).str.replace(VISA, 'Numero carta di credito', regex=True).values[0]
    elif pd.Series(row['MSGTEXT']).str.contains(MASTERCARD, regex=True).any():
        return pd.Series(row['MSGTEXT']).str.replace(MASTERCARD, 'Numero carta di credito', regex=True).values[0]
    elif pd.Series(row['MSGTEXT']).str.contains(AMEX, regex=True).any():
        return pd.Series(row['MSGTEXT']).str.replace(AMEX, 'Numero carta di credito', regex=True).values[0]
    elif pd.Series(row['MSGTEXT']).str.contains(MAESTRO, regex=True).any():
        return pd.Series(row['MSGTEXT']).str.replace(MAESTRO, 'Numero carta di credito', regex=True).values[0]
    elif pd.Series(row['MSGTEXT']).str.contains(VISA_MASTERCARD, regex=True).any():
        return pd.Series(row['MSGTEXT']).str.replace(VISA_MASTERCARD, 'Numero carta di credito', regex=True).values[0]
    else:
        return row['MSGTEXT']

tqdm.pandas()
#applico la funzione
df['AN_MSGTEXT'] = tqdm(df.progress_apply(replace_sensitive, axis=1))

#creo una colonna nel df che indica che dati sensibili sono stati trovati nei messaggi
df['Pattern'] = df['MSGTEXT'].apply(lambda x:
    'Codice Fiscale' if 'Codice Fiscale' in x else
    'Email' if 'Email' in x else
    'IBAN' if 'IBAN' in x else
    'Telefono' if 'Numero di telefono' in x else
    'Link' if 'Link' in x else
    'Carta di credito' if 'Numero carta di credito' in x else
    False)
