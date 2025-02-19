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
t_prompt="""
Analizza una conversazione tra un cliente e un servizio clienti di un'azienda nel mercato del gioco pubblico legale.
Estrai otto feature con valori prestabiliti: Risoluzione_richiesta (Risolto/Non risolto/Non risolvibile), Soddisfazione_cliente (Molto insoddisfatto/Insoddisfatto/Neutrale/Soddisfatto/Molto soddisfatto), Emozioni_cliente (Gioia/Rabbia/Sorpresa/Tristezza/Neutralità/Paura), Emozioni_operatore (Gioia/Rabbia/Sorpresa/Tristezza/Neutralità/Paura), Tono_cliente (Neutrale/Emotivo/Ironico/Sarcastico), Tono_operatore (Neutrale/Emotivo/Ironico/Sarcastico) Livello_linguistico_cliente (Scarso/Medio/Alto), Argomento_conversazione (Registrazione/Assistenza giochi/Operatore/Prelievo/Conto gioco/Ricarica/Bonus/Gioco responsabile/Altro). Estrai anche una feature con valore libero: Riassunto_conversazione.

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

### Istruzioni Riassunto e Argomento_conversazione###
Segui le seguenti istruzioni step by step per analizzare la conversazione e definire il topic della conversazione:

-Step 1 : Sintetizza il contenuto della conversazione in un testo di massimo 50 parole, concentrandoti  sull'argomento di cui si è parlato.

-Step 2: In base al riassunto dello Step 1 restituisci una coppia di classi che identifichino gli argomenti della conversazione. La prima classe è più generale, mentre la seconda sotto classe specifica in modo più preciso la prima. La classificazione deve attenersi al contenuto di questo dizionario con i seguenti valori : 
{ Registrazione : [Registrazione_ComeRegistrarsi, Registrazione_GestioneDocumenti, Registrazione_ProblemiRegistrazione, Registrazione_InvioDocumenti, Registrazione_ModificaRecapiti], 
Assistenza giochi : [ AssistenzaGiochi_Scommesse, AssistenzaGiochi_AltriGiochi, AssistenzaGiochi_SessioniCasinoBloccate],
Operatore: [], 
Prelievo: [Prelievo_InfoStato, Prelievo_DissociaMetodiPrelievo, Prelievo_InfoPrelievo],
Conto gioco : [ContoGioco_Sblocco, ContoGioco_RecuperoConto, ContoGioco_Stato, ContoGioco_Chiusura, inserimento_ContoGioco],
Ricarica: [Ricarica_Problemi, Ricarica_Info, Ricarica_FareRicarica],
Bonus: [Bonus_Info, Bonus_MancataErogazione],
Gioco responsabile : [GiocoResponsabile_AutoEsclusione, GiocoResponsabile_LimitiGioco],
Altro: [Altro_RicevitoriaVicina, Altro_ModificaAnagrafica, Altro_RimborsoVoucher, Altro_Prelevare, Altro_ProblemiAccessoSito, Altro_RiattivazioneContoInAutoesclusione, Altro_InfoSaldo, Altro_TrasferimentoContoGioco, Altro_SmarrimentoBolletta, Altro_AnnullamentoPrelievo, Altro_LimiteSpesaSettimanale]}
Associa alla conversazione fino a due coppie di classi. Non generare altri possibili valori, se non sai che valore associare alla feature usa l'etichetta Altro.

### Istruzioni generali ###
Se non sai che valore associare ad una feature, etichettala come NA e in nessun altro modo. Ad esempio, se l'operatore umano non interviene mai nella conversazione ma c'è solo la Live Chat, l'emozione e il tono dell'operatore devono essere necessariamente NA. Se la conversazione si interrompe prima che il problema possa essere risolto, le feature sulla risoluzione della richiesta o sulla soddisfazione devono essere NA.

Formatta l'output finale come un JSON. Tutte le feature devono essere scritte con la prima lettera maiuscola. 

### Chat ###

"""


# COMMAND ----------

#prompt nuova classificazione a 2 livelli e nuova feature
class_prompt_2lv="""
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

-Step 3: In base al riassunto dello Step 1 restituisci una coppia di classi che identifichino gli argomenti della conversazione. La prima classe è più generale, mentre la seconda sotto classe specifica in modo più preciso la prima. La classificazione deve attenersi al contenuto di questo dizionario con i seguenti valori : 

{'Bonus': ['Bonus_Conto Gioco', 'Bonus_Mancato Accredito Bonus', 'Bonus_Utilizzo saldo bonus'], 'Campagna Outbound': ['Campagna Outbound_Contatto'], 'Contatto non completato': ['Contatto non completato_Chat', 'Contatto non completato_Email', 'Contatto non completato_Interazione', 'Contatto non completato_Telefono'], 'Gestione Conto Gioco': ['Gestione Conto Gioco_Abilitazione CG', 'Gestione Conto Gioco_Accesso al CG', 'Gestione Conto Gioco_Affiliazione a PV', 'Gestione Conto Gioco_Chiusura CG', 'Gestione Conto Gioco_Deposito', 'Gestione Conto Gioco_Disconoscimento', 'Gestione Conto Gioco_Documentazione', 'Gestione Conto Gioco_Gioco Responsabile', 'Gestione Conto Gioco_Prelievo', 'Gestione Conto Gioco_Privacy', 'Gestione Conto Gioco_Registrazione'], 'Giochi': ['Giochi_10&Lotto', 'Giochi_Abilitazione al gioco', 'Giochi_Amusnet', 'Giochi_Bingo', 'Giochi_Casinò Gan', 'Giochi_Casinò Nyx/360', 'Giochi_Casino', 'Giochi_Consulab', 'Giochi_Contestazione', 'Giochi_Cristaltec', 'Giochi_EGT', 'Giochi_Evolution (Casinò Live)', 'Giochi_G&V on line', 'Giochi_Games Global', 'Giochi_Gioca on Live (Casinò)', 'Giochi_IGT (Casinò)', 'Giochi_Ippica', 'Giochi_Light&Wonder', 'Giochi_Lotto', 'Giochi_Medialive (Casinò)', 'Giochi_Microgame (Casinò)', 'Giochi_Million Day', 'Giochi_Playtech (Casinò)', 'Giochi_Poker', 'Giochi_Scommesse', 'Giochi_Skill Games', 'Giochi_SuperEnalotto', 'Giochi_Totocalcio', 'Giochi_Tuko (Casinò)', 'Giochi_Virtual Betting', 'Giochi_Worldmatch(Casinò)', 'Giochi_Playngo', 'Giochi_WMG ', 'Giochi_BetPoint', 'Giochi_Betpoint'], 'Loyalty': ['Loyalty_Informazioni'], 'Malfunzionamento': ['Malfunzionamento_App', 'Malfunzionamento_Casinò Nyx/360 ', 'Malfunzionamento_Consulab (Casinò)', 'Malfunzionamento_EGT (Casinò)', 'Malfunzionamento_Prelievi', 'Malfunzionamento_Problema Generale', 'Malfunzionamento_Supernalotto', 'Malfunzionamento_Worldmatch (Casinò)']}

Non generare altri possibili valori, se non sai che valore associare alla feature usa l'etichetta Altro.

### Istruzioni generali ###
Se non sai che valore associare ad una feature, etichettala come NA e in nessun altro modo. Ad esempio, se l'operatore umano non interviene mai nella conversazione ma c'è solo la Live Chat, l'emozione e il tono dell'operatore devono essere necessariamente NA. Se la conversazione si interrompe prima che il problema possa essere risolto, le feature sulla risoluzione della richiesta o sulla soddisfazione devono essere NA.

Formatta l'output finale come un JSON. Tutte le feature devono essere scritte con la prima lettera maiuscola. 

### Chat ###

"""

# COMMAND ----------

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

{'Bonus': {'Bonus_Conto Gioco': ['Conto Gioco_Inibizione Bonus', 'Conto Gioco_Richiesta Bonus', 'Conto Gioco_Richiesta Bonus Caring'], 'Bonus_Mancato Accredito Bonus': ['Mancato Accredito Bonus_10&Lotto', 'Mancato Accredito Bonus_Affiliato Web (BTAG)', 'Mancato Accredito Bonus_Benvenuto', 'Mancato Accredito Bonus_Benvenuto (Fidelity)', 'Mancato Accredito Bonus_Bingo', 'Mancato Accredito Bonus_Bonus Park', 'Mancato Accredito Bonus_Casinò', 'Mancato Accredito Bonus_Casinò Live', 'Mancato Accredito Bonus_Free Spin', 'Mancato Accredito Bonus_G&V on line', 'Mancato Accredito Bonus_Ippica', 'Mancato Accredito Bonus_Lotto', 'Mancato Accredito Bonus_Million Day', 'Mancato Accredito Bonus_Poker', 'Mancato Accredito Bonus_Scommesse', 'Mancato Accredito Bonus_Skill Games', 'Mancato Accredito Bonus_SuperEnalotto', 'Mancato Accredito Bonus_Virtual Betting'], 'Bonus_Utilizzo saldo bonus': ['Utilizzo saldo bonus_Affiliato Web (BTAG)', 'Utilizzo saldo bonus_Bingo', 'Utilizzo saldo bonus_Casinò', 'Utilizzo saldo bonus_Casino Live', 'Utilizzo saldo bonus_Free Spin', 'Utilizzo saldo bonus_G&V on line', 'Utilizzo saldo bonus_Ippica', 'Utilizzo saldo bonus_Lotterie', 'Utilizzo saldo bonus_Play Bonus Casinò', 'Utilizzo saldo bonus_Poker', 'Utilizzo saldo bonus_Skill Games', 'Utilizzo saldo bonus_Sport', 'Utilizzo saldo bonus_Virtual']}, 'Campagna Outbound': {'Campagna Outbound_Contatto': ['Contatto_Esito ok']}, 'Contatto non completato': {'Contatto non completato_Chat': ['Chat_Sessione Interrotta'], 'Contatto non completato_Email': ['Email_utente non riconosciuto'], 'Contatto non completato_Interazione': ['Interazione_Offese'], 'Contatto non completato_Telefono': ['Telefono_Caduta Linea']}, 'Gestione Conto Gioco': {'Gestione Conto Gioco_Abilitazione CG': ['Abilitazione CG_Conto sospeso per mancato invio doc', 'Abilitazione CG_Informazione Invio e ricezione documenti', 'Abilitazione CG_Richiesta doc alternativo ad AT/BT'], 'Gestione Conto Gioco_Accesso al CG': ['Accesso al CG_Accesso non autorizzato', 'Accesso al CG_Conto sospeso per password errata', 'Accesso al CG_Domanda di sicurezza', 'Accesso al CG_Info stato conto di gioco', 'Accesso al CG_Informazioni sospensione conto gioco', 'Accesso al CG_Modifica cellulare', 'Accesso al CG_Modifica email', 'Accesso al CG_Password errata', 'Accesso al CG_Recupero password', 'Accesso al CG_Recupero username', 'Accesso al CG_Ricezione OTP', 'Accesso al CG_Risposta di sicurezza', 'Accesso al CG_Verifica Email/Cellulare'], 'Gestione Conto Gioco_Affiliazione a PV': ['Affiliazione a PV_Modalità di Affiliazione (Trasferimento)'], 'Gestione Conto Gioco_Chiusura CG': ['Chiusura CG_Informazione Recesso del Concessionario', 'Chiusura CG_Modalità di chiusura', 'Chiusura CG_Problemi di Chiusura', 'Chiusura CG_Richiesta Chiusura per Gioco Compulsivo'], 'Gestione Conto Gioco_Deposito': ['Deposito_Applepay', 'Deposito_Bollettino postale', 'Deposito_Bonifico bancario', 'Deposito_Cambio account paypal', 'Deposito_Cambio account Skrill', 'Deposito_Carte di credito', 'Deposito_Informazioni cluster finanziario', 'Deposito_Muchbetter', 'Deposito_MyBank', 'Deposito_Neteller', 'Deposito_Onshop', 'Deposito_Paypal', 'Deposito_Paysafecard', 'Deposito_Postepay', 'Deposito_Rapid Transfer', 'Deposito_Ricarica diretta', 'Deposito_Ricezione OTP', 'Deposito_Richiesta Dissociazione Carta Di Credito', 'Deposito_Richiesta estratto conto', 'Deposito_Skrill', 'Deposito_Superamento massimali provider', 'Deposito_Superamento massimali utente', 'Deposito_Voucher', 'Deposito_Voucher deposito scaduto'], 'Gestione Conto Gioco_Disconoscimento': ['Disconoscimento_Apertura conto', 'Disconoscimento_Depositi', 'Disconoscimento_Prelievi', 'Disconoscimento_Transazioni CG', 'Disconoscimento_Transazioni di Gioco'], 'Gestione Conto Gioco_Documentazione': ['Documentazione_Chiusura CG', 'Documentazione_Denuncia ad Autorità', 'Documentazione_Disconoscimento Apertura CG', 'Documentazione_Invio Titolarità', 'Documentazione_Modifica Cellulare', 'Documentazione_Modifica Codice IBAN', 'Documentazione_Modifica Dati Anagrafici', 'Documentazione_Modifica E-Mail', 'Documentazione_Nuovo Doc. Riconoscimento', 'Documentazione_Revoca Autoesclusione Permanente', 'Documentazione_Riattivazione CG sospeso per art.14', 'Documentazione_Richiesta info autorità PS/Giudiziaria'], 'Gestione Conto Gioco_Gioco Responsabile': ['Gioco Responsabile_Info autoesclusione', 'Gioco Responsabile_Info autolimitazione', 'Gioco Responsabile_Modalità Revoca Autoesclusione Permanente', 'Gioco Responsabile_Richiesta esclusione verticale di gioco', 'Gioco Responsabile_Richiesta riabilitazione verticale di gioco', 'Gioco Responsabile_Verifiche per Gioco Compulsivo'], 'Gestione Conto Gioco_Prelievo': ['Prelievo_Abilitazione voucher', 'Prelievo_Applepay', 'Prelievo_Bonifico bancario', 'Prelievo_Bonifico Domiciliato', 'Prelievo_Bonifico Istantaneo', 'Prelievo_Carte di credito', 'Prelievo_Deposito con debit card', 'Prelievo_Disabilitazione paypal', 'Prelievo_Disassociazione metodi di prelievo', 'Prelievo_Info per prelievo stornato da operatore', 'Prelievo_Info per prelievo stornato da sistema/ente', 'Prelievo_Informazione ricezione documentazione aggiuntiva', 'Prelievo_Informazione stato lavorazione prelievo', 'Prelievo_Informazioni bilanciamento', 'Prelievo_Informazioni cluster finanziario-Carte terzi', 'Prelievo_Informazioni cluster finanziario-Informazioni NO Voucher', 'Prelievo_Informazioni cluster finanziario-Opposite\\collusion', 'Prelievo_Informazioni cluster finanziario-Paypal terzi', 'Prelievo_Informazioni cluster finanziario-Titolarità account PayPal', 'Prelievo_Informazioni cluster finanziario-Titolarità carte', 'Prelievo_Informazioni cluster finanziario-Titolarità iban', 'Prelievo_Informazioni Invio/Ricezione Titolarità', 'Prelievo_Informazioni quantità prelievi giornalieri', 'Prelievo_Informazioni storno Voucher', 'Prelievo_Mancato pagamento voucher prelievo', 'Prelievo_Muchbetter', 'Prelievo_Neteller', 'Prelievo_Paypal', 'Prelievo_Paysafecard', 'Prelievo_Postepay', 'Prelievo_Prelievo stornato per documentazione mancante', 'Prelievo_Prelievo stornato per iban errato', 'Prelievo_Prelievo stornato per modifica metodo di pagamento', 'Prelievo_Prelievo stornato su richiesta del cliente', 'Prelievo_Rapid Transfer', 'Prelievo_Ricezione OTP', 'Prelievo_Skrill', 'Prelievo_Storno prelievo', 'Prelievo_Tempistiche accredito prelievo', 'Prelievo_Voucher', 'Prelievo_Voucher prelievo scaduto'], 'Gestione Conto Gioco_Privacy': ['Privacy_Contatto assistenza clienti', 'Privacy_Diritto a Oblio', 'Privacy_Diritto alla Cancellazione Dati', 'Privacy_Diritto alla Portabilità dei Dati', 'Privacy_Diritto di accesso ai dati', 'Privacy_Diritto di limitazione e opposizione del trattamento', 'Privacy_Diritto di rettifica dati', 'Privacy_Gestione consensi', 'Privacy_Informazioni Oblio', 'Privacy_Richiesta informazioni'], 'Gestione Conto Gioco_Registrazione': ['Registrazione_Conto Dormiente/CG Chiuso da Rinnovare', 'Registrazione_Modalità di Apertura', 'Registrazione_Necessaria modifica anagrafica', 'Registrazione_Problemi di Registrazione', 'Registrazione_Registrazione PV fallita']}, 'Giochi': {'Giochi_10&Lotto': ['10&Lotto_Accredito Vincite', '10&Lotto_Contestazione Giocata', '10&Lotto_Modalità di Gioco', '10&Lotto_Promozioni'], 'Giochi_Abilitazione al gioco': ['Abilitazione al gioco_Sblocco Pop-Up'], 'Giochi_Amusnet': ['Amusnet_Accredito Vincite', 'Amusnet_Blocco Sessioni con ID ADM', 'Amusnet_Blocco Sessioni senza ID ADM'], 'Giochi_BetPoint': ['BetPoint_Accredito Vincite', 'BetPoint_Blocco Sessioni con ID ADM'], 'Giochi_Betpoint': ['Betpoint_Blocco Sessioni senza ID ADM'], 'Giochi_Bingo': ['Bingo_Accredito Vincite', 'Bingo_Contestazione Giocata', 'Bingo_Modalità di Gioco', 'Bingo_Problemi Acquisto Cartelle', 'Bingo_Promozioni'], 'Giochi_Casino': ['Casino_Problema Accesso Gioco'], 'Giochi_Casinò Gan': ['Casinò Gan_Accredito Vincite', 'Casinò Gan_Blocco Sessione con ID ADM', 'Casinò Gan_Blocco Sessione senza ID ADM'], 'Giochi_Casinò Nyx/360': ['Casinò Nyx/360_Accredito Vincite', 'Casinò Nyx/360_Blocco Sessione con ID ADM', 'Casinò Nyx/360_Blocco Sessione senza ID ADM'], 'Giochi_Consulab': ['Consulab_Accredito Vincite', 'Consulab_Blocco Sessione con ID ADM', 'Consulab_Blocco Sessione senza ID ADM'], 'Giochi_Contestazione': ['Contestazione_RNG o Probabilità di Vincita (RTP)'], 'Giochi_Cristaltec': ['Cristaltec_Accredito Vincite', 'Cristaltec_Blocco Sessioni con ID ADM', 'Cristaltec_Blocco Sessioni senza ID ADM'], 'Giochi_EGT': ['EGT_Accredito Vincite', 'EGT_Blocco Sessione con ID ADM', 'EGT_Blocco Sessione senza ID ADM'], 'Giochi_Evolution (Casinò Live)': ['Evolution (Casinò Live)_Abilitazione', 'Evolution (Casinò Live)_Accredito Vincite', 'Evolution (Casinò Live)_Blocco Sessione con ID ADM', 'Evolution (Casinò Live)_Blocco Sessione senza ID ADM'], 'Giochi_G&V on line': ['G&V on line_Accredito Vincite', 'G&V on line_Contestazione Giocata', 'G&V on line_Modalità di Gioco', 'G&V on line_Promozioni'], 'Giochi_Games Global': ['Games Global_Accredito Vincite', 'Games Global_Blocco Sessioni con ID ADM', 'Games Global_Blocco Sessioni senza ID ADM'], 'Giochi_Gioca on Live (Casinò)': ['Gioca on Live (Casinò)_Accredito Vincite', 'Gioca on Live (Casinò)_Blocco Sessione con ID ADM', 'Gioca on Live (Casinò)_Blocco Sessione senza ID ADM'], 'Giochi_IGT (Casinò)': ['IGT (Casinò)_Accredito Vincite', 'IGT (Casinò)_Blocco Sessione con ID ADM', 'IGT (Casinò)_Blocco Sessione senza ID ADM'], 'Giochi_Ippica': ['Ippica_Accredito Vincite', 'Ippica_Giocata Senza ID ADM', 'Ippica_Informazione refertazione ticket', 'Ippica_Informazioni mercati', 'Ippica_Informazioni palinsesto', 'Ippica_Rimborsi'], 'Giochi_Light&Wonder': ['Light&Wonder_Accredito Vincite', 'Light&Wonder_Blocco Sessioni con ID ADM', 'Light&Wonder_Blocco Sessioni senza ID ADM'], 'Giochi_Lotto': ['Lotto_Accredito Vincite', 'Lotto_Contestazione Giocata', 'Lotto_Modalità di Gioco', 'Lotto_Promozioni'], 'Giochi_Medialive (Casinò)': ['Medialive (Casinò)_Accredito Vincite', 'Medialive (Casinò)_Blocco Sessione con ID ADM', 'Medialive (Casinò)_Blocco Sessione senza ID ADM'], 'Giochi_Microgame (Casinò)': ['Microgame (Casinò)_Accredito Vincite', 'Microgame (Casinò)_Blocco Sessione con ID ADM', 'Microgame (Casinò)_Blocco Sessione senza ID ADM'], 'Giochi_Million Day': ['Million Day_Accredito Vincite', 'Million Day_Contestazione Giocata', 'Million Day_Modalità di Gioco', 'Million Day_Promozioni'], 'Giochi_Playngo': ['Playngo_Accredito Vincite', 'Playngo_Blocco Sessioni con ID ADM', 'Playngo_Blocco Sessioni senza ID ADM'], 'Giochi_Playtech (Casinò)': ['Playtech (Casinò)_Accredito Vincite', 'Playtech (Casinò)_Blocco Sessione con ID ADM', 'Playtech (Casinò)_Blocco Sessione senza ID ADM'], 'Giochi_Poker': ['Poker_Abilitazione', 'Poker_Accesso alla Lobby', 'Poker_Accredito Vincite', 'Poker_Blocco Sessioni Cash', 'Poker_Blocco Tornei', 'Poker_Classifiche', 'Poker_Disconnessione', 'Poker_Informazioni Tornei Twister', 'Poker_Installazione', 'Poker_Promozioni', 'Poker_Requisiti Device', 'Poker_Storico Mani'], 'Giochi_Scommesse': ['Scommesse_Accredito Vincite', 'Scommesse_Avvenimento Sospeso', 'Scommesse_Avvenimento Void', 'Scommesse_Bet Scanner', 'Scommesse_Contestazione', 'Scommesse_Informazione Refertazione Avvenimenti', 'Scommesse_Informazioni Cashout', 'Scommesse_Informazioni mercati', 'Scommesse_Informazioni Storno Vincita', 'Scommesse_Mancata Refertazione', 'Scommesse_Prenotazione Giocata', 'Scommesse_Promozioni', 'Scommesse_Pubblicazione Palinsesti-Avvenimenti', 'Scommesse_Scommessa accettata parzialmente', 'Scommesse_Scommessa Senza IA ADM', 'Scommesse_Scommesse placed', 'Scommesse_Scommesse rifiutate', 'Scommesse_Scommessa Senza ID ADM'], 'Giochi_Skill Games': ['Skill Games_Accredito Vincite', 'Skill Games_Blocco Sessione', 'Skill Games_Blocco Tornei', 'Skill Games_Contestazione Giocata', 'Skill Games_Modalità di Gioco', 'Skill Games_Modifica Nickname', 'Skill Games_Promozioni'], 'Giochi_SuperEnalotto': ['SuperEnalotto_Contestazione Giocata', 'SuperEnalotto_Modalità di Gioco', 'SuperEnalotto_Promozioni', 'SuperEnalotto_Regolamenti'], 'Giochi_Totocalcio': ['Totocalcio_Accredito Vincite', 'Totocalcio_Contestazione Giocata', 'Totocalcio_Modalità di Gioco', 'Totocalcio_Promozioni'], 'Giochi_Tuko (Casinò)': ['Tuko (Casinò)_Accredito Vincite', 'Tuko (Casinò)_Blocco Sessione con ID ADM', 'Tuko (Casinò)_Blocco Sessione senza ID ADM'], 'Giochi_Virtual Betting': ['Virtual Betting_Accredito Vincite', 'Virtual Betting_Giocata Senza ID ADM', 'Virtual Betting_Informazione Refertazione Avvenimenti', 'Virtual Betting_Informazioni mercati', 'Virtual Betting_Prenotazione Giocata', 'Virtual Betting_Promozioni', 'Virtual Betting_Pubblicazione Palinsesti-Avvenimenti', 'Virtual Betting_Scommessa accettata parzialmente', 'Virtual Betting_Scommesse rifiutate'], 'Giochi_WMG ': ['WMG _Accredito Vincite', 'WMG _Blocco Sessioni con ID ADM', 'WMG _Blocco Sessioni senza ID ADM'], 'Giochi_Worldmatch(Casinò)': ['Worldmatch(Casinò)_Accredito Vincite', 'Worldmatch(Casinò)_Blocco Sessione con ID ADM', 'Worldmatch(Casinò)_Blocco Sessione senza ID ADM']}, 'Loyalty': {'Loyalty_Informazioni': ['Informazioni_Funzionalità']}, 'Malfunzionamento': {'Malfunzionamento_10&Lotto': ['10&Lotto_Visualizzazione Parziale'], 'Malfunzionamento_App': ['App_Bingo Android', 'App_Bingo IOS', 'App_Casinò Android', 'App_Casinò IOS', 'App_Poker Android', 'App_Poker IOS', 'App_Sport Android', 'App_Sport IOS', 'App_Virtual Android', 'App_Virtual IOS'], 'Malfunzionamento_Casinò Gan': ['Casinò Gan_Blocco Sessione'], 'Malfunzionamento_Casinò Nyx/360 ': ['Casinò Nyx/360 _Blocco Sessione'], 'Malfunzionamento_Consulab (Casinò)': ['Consulab (Casinò)_Blocco Sessione'], 'Malfunzionamento_Conto Gioco': ['Conto Gioco_Accesso Sito Mobile', 'Conto Gioco_Accesso Sito Web'], 'Malfunzionamento_Deposito': ['Deposito_Bollettino postale', 'Deposito_Bonifico bancario', 'Deposito_Carte di credito', 'Deposito_Muchbetter', 'Deposito_Neteller', 'Deposito_Onshop', 'Deposito_Paypal', 'Deposito_Postepay', 'Deposito_Rapid Transfer', 'Deposito_Ricarica diretta', 'Deposito_Skrill', 'Deposito_Voucher'], 'Malfunzionamento_EGT (Casinò)': ['EGT (Casinò)_Blocco Sessione'], 'Malfunzionamento_Evolution (Casinò Live)': ['Evolution (Casinò Live)_Blocco Sessione'], 'Malfunzionamento_Gioca on Live (Casinò)': ['Gioca on Live (Casinò)_Blocco Sessione'], 'Malfunzionamento_IGT (Casinò)': ['IGT (Casinò)_Blocco Sessione'], 'Malfunzionamento_Ippica': ['Ippica_Assenza quote', 'Ippica_Assenza Schedine', 'Ippica_Problema streaming'], 'Malfunzionamento_Lotto': ['Lotto_Visualizzazione Parziale'], 'Malfunzionamento_Microgame (Casinò)': ['Microgame (Casinò)_Blocco Sessione'], 'Malfunzionamento_Million Day': ['Million Day_Visualizzazione Parziale'], 'Malfunzionamento_Playtech (Casinò)': ['Playtech (Casinò)_Blocco Sessione'], 'Malfunzionamento_Poker': ['Poker_Accesso alla Lobby', 'Poker_Iscrizione Torneo'], 'Malfunzionamento_Prelievi': ['Prelievi_ApplePay', 'Prelievi_Bonifico bancario', 'Prelievi_Bonifico Domiciliato', 'Prelievi_Carte di credito', 'Prelievi_Muchbetter', 'Prelievi_Neteller', 'Prelievi_Paypal', 'Prelievi_Postepay', 'Prelievi_Rapid Transfer', 'Prelievi_Skrill', 'Prelievi_Voucher'], 'Malfunzionamento_Problema Generale': ['Problema Generale_Pegasos', 'Problema Generale_Ricezione OTP', 'Problema Generale_Verticale Casino', 'Problema Generale_Verticale Poker', 'Problema Generale_Verticale Scommesse'], 'Malfunzionamento_Scommesse': ['Scommesse_Assenza Quote App', 'Scommesse_Assenza Quote Mobile', 'Scommesse_Assenza Quote Web', 'Scommesse_Disallineamento Quote', 'Scommesse_Discrepanza quota coupon da quota biglietto', 'Scommesse_Errore Refertazione Avvenimenti', 'Scommesse_Errori quote', 'Scommesse_Gioco Chiuso', 'Scommesse_Problema streaming', 'Scommesse_Scommesse doppie', 'Scommesse_Scommesse placed'], 'Malfunzionamento_Skill Games': ['Skill Games_Blocco Sessione'], 'Malfunzionamento_Supernalotto': ['Supernalotto_Accettazione Giocata', 'Supernalotto_Accredito Vincita', 'Supernalotto_Errata Visualizzazione'], 'Malfunzionamento_Tuko (Casinò)': ['Tuko (Casinò)_Blocco Sessione'], 'Malfunzionamento_Virtual Betting': ['Virtual Betting_Assenza Quote App', 'Virtual Betting_Assenza Quote Mobile', 'Virtual Betting_Assenza Quote Web', 'Virtual Betting_Discrepanza quota coupon da quota biglietto', 'Virtual Betting_Errore Refertazione Avvenimenti', 'Virtual Betting_Errori quote', 'Virtual Betting_Gioco Chiuso', 'Virtual Betting_Problema streaming', 'Virtual Betting_Scommesse doppie', 'Virtual Betting_Scommesse placed'], 'Malfunzionamento_Worldmatch (Casinò)': ['Worldmatch (Casinò)_Blocco Sessione']}}

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

df_final = spark.read.csv("abfss://botchat@adlsvoicebotdev.dfs.core.windows.net/Sample/SampleAnnotati/GPT_Annotations_FS_17_01/part-00000-tid-8700051930863589699-675cde2e-641b-4a56-a56b-5c462635dc74-163-1-c000.csv", header=True, inferSchema=True, sep=";")

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