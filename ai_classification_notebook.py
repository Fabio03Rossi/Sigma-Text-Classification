#%% md
# # SetFit Few-Shot Classification
# 
# In questo notebook vengono mostrati i segmenti di codice e i procedimenti eseguiti per effettuare l'allenamento e l'inferenza del modello di classificazione.
# 
# L'obiettivo è quello di classificare le note di chiusura degli interventi dei tecnici in base alle aree di guasto possibili della macchina. Queste comprendono: **CASSETTE, CT, NE, NF, NV e SHUTTER**.
# Il sistema deve poter categorizzare questi elementi testuali in maniera più o meno affidabile attraverso IA, includendo anche più di un'area per selezione.
# 
# La task che quindi dobbiamo eseguire si ricongiunge a una classificazione di testo multi-label.
# 
# L'approccio utilizzato per la risoluzione consiste nel [**Few-Shot Learning**](https://www.ibm.com/think/topics/few-shot-learning), dove un modello IA di embedding viene allenato su un numero N di esempi reali per ogni singola label. Richiede più tempo e risorse hardware ma permette di ottenere risultati migliori in ambienti con pochi dati di allenamento e soprattutto categorizzazioni complesse come nel nostro caso.
# 
# Utilizziamo diverse librerie di HuggingFace come [**Sentence-Transformers**](https://sbert.net/) e [**SetFit**](https://huggingface.co/docs/setfit/main/en/index) per la definizione e l'allenamento del modello IA, insieme a [**Datasets**](https://huggingface.co/docs/hub/en/datasets) per la manipolazione dei dati necessari agli step di training, validation e test.
# 
# 
# Nella prima cella definiamo la variabile d'ambiente per poter utilizzare la GPU durante l'allenamento.
# 
#%%
import os
from pandas.core.interchange.dataframe_protocol import DataFrame

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.system("pip install setfit datasets sentence-transformers pandas numpy")
#%% md
# Come primo step si vanno a importare i 3 dataset necessari a completare l'operazione di training.
# Consistono nel dataset di training che presenta una varietà di dati labelizzati su cui il modello eseguirà l'allenamento, il dataset di validation che contiene altri elementi labelizzati e conosciuti per validare l'accuratezza del modello in casi controllati, e infine abbiamo il dataset di test contenente una grande collezione di note di chiusura senza labelizzazioni revisionate in cui il modello verrà messo contro le selezioni dei tecnici.
# 
# I dataset sono estratti in `DataFrame` pandas da dei file Excel, e successivamente convertiti in `Dataset` filtrando le colonne ritenute non rilevanti per l'allenamento.
# 
# Fare attenzione ai nomi delle colonne delle label da valutare, devono necessariamente corrispondere a `"{label} ground-truth"` per evitare problematiche.
#%%
import pandas as pd
from datasets import Dataset

df_training: DataFrame = pd.read_excel("docs/training_docs/DummyClosingNotes-training.xlsx") # Path del file Excel
df_validation: DataFrame = pd.read_excel("docs/training_docs/DummyClosingNotes-validation.xlsx")
df_test: DataFrame = pd.read_excel("docs/training_docs/DummyClosingNotes-test.xlsx")

# Rimozione di colonne non rilevanti per indici, singoli o per sequenza
train_dataset: Dataset = Dataset.from_pandas(df_training.iloc[:, 9:16])
validation_dataset: Dataset = Dataset.from_pandas(df_validation.iloc[:, 9:16])
test_dataset: Dataset = Dataset.from_pandas(df_test.iloc[:, [0,1,2,3,4,5, 32, 49]])

features = train_dataset.column_names
features.remove("Closing Note")
features
#%% md
# Eseguiamo ulteriori elaborazioni sui nostri dataset correnti, in cui rimappiamo le colonne delle singole label in una singola colonna `"labels"` contenente un vettore delle singole selezioni in formato binario.
# 
# Successivamente vengono rimosse tutte le righe in cui le note di chiusura risultano vuote.
#%%
def clean_dataset(ds):
    ds = ds.map(lambda entry: {"labels": [entry[label] for label in features]})
    ds = ds.map(lambda entry: {"text": entry["Closing Note"]})
    ds = Dataset.from_pandas(ds.to_pandas().dropna())
    ds = Dataset.from_pandas(ds.to_pandas().replace(r'^\s*$', "Empty", regex=True))
    return ds
#%%
train_dataset
#%%
validation_dataset
#%%
test_dataset
#%%
train_dataset = clean_dataset(train_dataset)

validation_dataset = clean_dataset(validation_dataset)

test_dataset = clean_dataset(test_dataset)
#%%
train_dataset
#%%
validation_dataset
#%%
test_dataset
#%% md
# _`get_templated_dataset`_ è un metodo di SetFit che permette di generare dati sintetici a seconda delle necessità.
# Date le label e impostando il parametro `multi_label = True`, possiamo decidere un numero di esempi sintetici che verranno aggiunti per label seguendo un template uguale per tutte. Risulta essere metodologia semplice, ma aiuta comunque per la classificazione.
#%%
from setfit import get_templated_dataset

train_dataset = get_templated_dataset(train_dataset, candidate_labels=features, sample_size=5, label_column="labels", multi_label=True, template="Il problema è del {}")

train_dataset
#%% md
# ## Fine-Tuning
# 
# Si definisce una funzione per l'inizializzazione del modello che verrà usato dal `Trainer`.
# 
# In questo caso si sta usando [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5).
# 
# Vi è anche la possibilità di passare direttamente la variabile del modello, ma la funzione è vantaggiosa per motivi di versatilità, e la possibilità di implementare alcune funzionalità aggiuntive che la richiedono.
# 
# Si notano delle parametrizzazioni di base applicate tra cui:
#     <p>- la temperatura (rappresenta la creatività del modello ed è impostata a 0);</p>
#     <p>- il numero e la lista di label con cui deve rispondere;</p>
#     <p>- la strategia di selezione della label (nel nostro caso `"multi-output"`, ma anche `"one-vs-rest"`...).</p>
#%%
import torch
from setfit import SetFitModel

os.environ["WANDB_DISABLED"] = "true"

def model_init() -> SetFitModel:
    params = {"device": torch.device("cuda"), 'out_features': 6, 'temperature': 0}
    return SetFitModel.from_pretrained("BAAI/bge-small-en-v1.5",
                                       multi_target_strategy="multi-output", params=params, labels=features)
#%% md
# Successivamente si specificano nel dettaglio i parametri a cui il `Trainer` sarà sottoposto, questo significa che ogni elemento che si definisce di seguito riguarderà la fase di allenamento e non sarà quindi necessario tenerne traccia in qualsiasi altro utlizzo post-allenamento.
# 
# Qui è anche possibile specificare dei parametri per il debugging.
# 
# Gli elementi che più interessano in questo caso sono: <br>
#     <p>- **`body_learning_rate`** (definisce "quanto" il modello dovrà imparare a ogni step dell'allenamento. Se troppo alto rischia di causare più facilmente "over-training");</p>
#     <p>- **`num_epochs`** (rappresenta il numero di volte che il dataset viene attraversato nella sua interezza);</p>
#     <p>- **`batch_size`** (il numero di sample processato per ogni step, un "chunk");</p>
#     <p>- **`warmup_proportion`** (influisce sul `learning_rate` nei primi step di allenamento, mantenendo un basso valore prima di passare al valore definito. Dovrebbe aiutare ad aumentare l'attenzione del modello);</p>
#     <p>- **`sampling_strategy`** (riguarda il bilanciamento del numero di comparazioni per label. `"unique"` in questo caso non bilancia il peso delle label, garantendo comunque che vengano effettuate tutte le comparazioni senza duplicazioni).</p>
#%%
from setfit import TrainingArguments

args = TrainingArguments(
    # Parametri di training opzionali:
    body_learning_rate=1.9859376752033417e-05,
    num_epochs=2,
    batch_size=6,
    warmup_proportion=0.2,
    sampling_strategy="unique",
    # Parametri di debugging:
    logging_strategy="steps",
    logging_steps=1000,
    eval_strategy="steps",
    logging_first_step=True,
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    run_name="finetune-setfit",
    load_best_model_at_end=True
)
#%% md
# Si inizializza poi il `Trainer` specificando i dataset di training e validation, gli argomenti e la funzione di inizializzazione del modello precedentemente definiti, avviando lo step di allenamento.
#%%
from setfit import Trainer

trainer = Trainer(
    model_init=model_init,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    column_mapping={"text": "text", "labels": "label"},
)

trainer.train()
#%% md
# Si verifica l'accuratezza generale del modello allenato con una semplice metrica, contro il set di validation.
#%%
metrics = trainer.evaluate()
print(metrics)
#%% md
# A seguito del completamento del Fine-Tuning, si salva il modello su disco in modo tale da poter farne riuso.
#%%
model = trainer.model
model.save_pretrained("models/DummyModel") # Path del modello
#%%
import numpy
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import multilabel_confusion_matrix

## Funzioni di utilità

def export_as_excel(filename, preds, test_dataset):
    preds = map(lambda i: i.numpy().astype(numpy.int64).tolist(), preds)
    df_out = Dataset.to_pandas(test_dataset)

    cassette, ct, ne, nf, nv, shutter = [], [], [], [], [], []

    for k in preds:
        cassette.append(int(k[0]))
        ct.append(int(k[1]))
        ne.append(int(k[2]))
        nf.append(int(k[3]))
        nv.append(int(k[4]))
        shutter.append(int(k[5]))

    df_out = pd.concat([df_out, pd.DataFrame({
                    "CASSETTE Model": cassette,
                    "CT Model": ct,
                    "NE Model": ne,
                    "NF Model": nf,
                    "NV Model": nv,
                    "SHUTTER Model": shutter
        })], axis=1)

    df_out.to_excel(filename, index=False)


# Associa una predizione alla rispettiva label selezionata
def parse_pred(i):
    result, count = [], 0
    for k in i:
        if k == 1:
            result.append(str(features[count]))
        count += 1
    return result


# Suddivisione della matrice per label
# Output per label: TN, FP, FN, TP
def confusion_matrix(w_dataset, preds):
    res = multilabel_confusion_matrix(w_dataset['labels'], preds).ravel().tolist()
    res = [res[i:i + 4] for i in range(0, len(res), 4)]
    res = {"CASSETTE": res[0], "CT": res[1], "NE": res[2], "NF": res[3], "NV": res[4], "SHUTTER": res[5]}
    return res


def parse_balanced_accuracy(truth_list, pred_list):
    return str(round(balanced_accuracy_score(truth_list, pred_list), 2))


# Stampa in output i risultati delle predizioni del modello in maniera strutturata
def print_results(w_dataset, preds, pred_proba):
    ac_cassette, ac_ct, ac_nf, ac_ne, ac_nv, ac_shutter = [], [], [], [], [], []
    ac_cassette_pred, ac_ct_pred, ac_nf_pred, ac_ne_pred, ac_nv_pred, ac_shutter_pred = [], [], [], [], [], []

    array = []

    for idx, p in enumerate(preds):
        p = p.numpy().astype(numpy.int64).tolist()
        matching = w_dataset['labels'][idx]
        print(str(parse_pred(matching)) + '\n' + str(parse_pred(p)) + '\n' + str(pred_proba[idx]) + '\n')
        array.append(matching == p)

        # Teniamo conto delle predizioni per label per il calcolo della singola accuratezza
        ac_cassette.append(matching[0])
        ac_cassette_pred.append(p[0])
        ac_ct.append(matching[1])
        ac_ct_pred.append(p[1])
        ac_ne.append(matching[2])
        ac_ne_pred.append(p[2])
        ac_nf.append(matching[3])
        ac_nf_pred.append(p[3])
        ac_nv.append(matching[4])
        ac_nv_pred.append(p[4])
        ac_shutter.append(matching[5])
        ac_shutter_pred.append(p[5])


    result = "Total: " + str(round(accuracy_score(w_dataset["labels"], preds), 2)) + " - " + str(len(array)) + "\n"
    cassette_result = "CASSETTE: " + parse_balanced_accuracy(ac_cassette, ac_cassette_pred) + " - \n"
    ct_result = "CT: " + parse_balanced_accuracy(ac_ct, ac_ct_pred) + " - \n"
    ne_result = "NE: " + parse_balanced_accuracy(ac_ne, ac_ne_pred) + " - \n"
    nf_result = "NF: " + parse_balanced_accuracy(ac_nf, ac_nf_pred) + " - \n"
    nv_result = "NV: " + parse_balanced_accuracy(ac_nv, ac_nv_pred) + " - \n"
    shutter_result = "SHUTTER: " + parse_balanced_accuracy(ac_shutter, ac_shutter_pred) + " - \n"

    print(result + cassette_result + ct_result + ne_result + nf_result + nv_result + shutter_result)

    print("TN, FP, FN, TP \n")
    print(str(confusion_matrix(w_dataset, preds)) + '\n')
#%% md
# ## Inferenza
# 
# Quando si vuole utilizzare un modello esterno già presente su disco ad esempio, basta richiamare il metodo _`SetFitModel.from_pretrained`_ con il path e nome del modello.
# 
# Per eseguire una predizione basta richiamare il metodo predict fornendo la lista di stringhe che si devono testare. In questo caso utilizziamo la colonna `"text"` del dataset di testing che contiene tutte le note dei tecnici.
# 
# Si può anche utilizzare il metodo _`predict_proba`_ che fornisce le percentuali di confidenza sulle scelte che il modello ha preso per le predizioni.
#%%
model = SetFitModel.from_pretrained("models/DummyModel")

preds = model.predict(test_dataset['text'])
pred_proba = model.predict_proba(test_dataset['text'])
#%% md
# Si salvano le predizioni all'interno di un nuovo file Excel per una più facile consultazione e analisi.
#%%
export_as_excel('closing_notes_output/ClosingNotesResults.xlsx', preds, test_dataset)
#%% md
# Infine si va a stampare in output ogni singola predizione e le sue probabilità, accodando ai risultati svariati valori di accuratezza tra cui la _balanced accuracy_ di ogni label e la _subset accuracy_ del totale.
# 
# Si forniscono anche le informazioni di training relative al modello allenato in questo caso, per poi fornire un completo report di classificazione tramite **sklearn**, approfondendo nel dettaglio altre metriche di valutazione che potrebbero risultare rilevanti per ulteriori procedure di fine-tuning.
#%%
print_results(test_dataset, preds, pred_proba)

print("body_learning_rate: " + str(args.body_learning_rate))
print("num_epochs: " + str(args.num_epochs))
print("batch_size: " + str(args.batch_size))
print("warmup_proportion: " + str(args.warmup_proportion))

print(classification_report(test_dataset['labels'], preds, target_names=features, zero_division=0))
#%%
preds