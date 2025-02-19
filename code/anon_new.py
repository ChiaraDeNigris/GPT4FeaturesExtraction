from pyspark.sql.functions import col, when, regexp_replace

#Define the sensitive data patterns using Regex
CODICE_FISCALE = r'\b(?:[A-Za-z][AEIOUaeiou][AEIOUXaeioux]|[AEIOUaeiou]X{2}|[B-DF-HJ-NP-TV-Zb-df-hj-np-tv-z]{2}[A-Za-z]){2}(?:[\dLMNP-Vlmnp-v]{2}(?:[A-EHLMPR-Ta-ehlmpr-t](?:[04LQ][1-9MNP-Vlmnp-v]|[15MR][\dLMNP-Vlmnp-v]|[26NS][0-8LMNP-Ulmnp-u])|[DHPSdhps][37PTpt][0L]|[ACELMRTacelmr-t][37PTpt][01LM]|[AC-EHLMPR-Tac-ehlmpr-t][26NSns][9Vv])|(?:[02468LNQSUlnqsu][048LQU048lqu]|[13579MPRTV13579mprtv][26NS26ns])B[26NS26ns][9Vv])(?:[A-MZa-mz][1-9MNP-Vlmnp-v][\dLMNP-Vlmnp-v]{2}|[A-Ma-m][0L](?:[1-9MNP-Vlmnp-v][\dLMNP-Vlmnp-v]|[0L][1-9MNP-Vlmnp-v]))[A-Za-z]\b'
EMAIL = r'\b[\w\-\.]+@([\w-]+\.)+[\w-]{2,}\b'
IBAN = r'\b[A-Za-z]{2}[0-9]{2}(?:[ ]?[0-9]{4}){4}(?!(?:[ ]?[0-9]){3})(?:[ ]?[0-9]{1,2})?\b'
VISA = r'\b4[0-9]{12}(?:[0-9]{3})?\b'
MASTERCARD = r'\b(5[1-5][0-9]{14}|2(22[1-9][0-9]{12}|2[3-9][0-9]{13}|' \
             r'[3-6][0-9]{14}|7[0-1][0-9]{13}|720[0-9]{12}))\b'
AMEX = r'\b3[47][0-9]{13}\b'
MAESTRO = r'\b(5018|5020|5038|6304|6759|6761|6763)[0-9]{8,15}\b'
VISA_MASTERCARD = r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14})\b'
PHONE_NUMBER = r'\b[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}\b'
LINK = r'(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])'

#Assuming df is a Spark DataFrame. Replace sensitive content found in the column MSGTEXT with a generic string a write them in a new column called AN_MSGTEXT 
df = df.withColumn("AN_MSGTEXT", col("MSGTEXT"))

#Add a Pattern column which defines which pattern has been found
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

#Define a list containing tuples with the pattern name and the replacement
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

#Iterate through sensitive data patterns and replace them with generic strings
for pattern, replacement in patterns:
    df = df.withColumn("AN_MSGTEXT", regexp_replace(col("AN_MSGTEXT"), pattern, replacement))