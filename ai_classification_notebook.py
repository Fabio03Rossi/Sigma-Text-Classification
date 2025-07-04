import os

import numpy
import pandas as pd
import torch
from datasets import Dataset
from pandas import DataFrame
from setfit import SetFitModel, get_templated_dataset, TrainingArguments, Trainer
from sklearn.metrics import multilabel_confusion_matrix, balanced_accuracy_score, accuracy_score, classification_report


## Funzioni di utilità

def clean_dataset(ds):
    ds = ds.map(lambda entry: {"labels": [entry[label] for label in features]})
    ds = ds.map(lambda entry: {"text": entry["Closing Note"]})
    ds = Dataset.from_pandas(ds.to_pandas().dropna())
    ds = Dataset.from_pandas(ds.to_pandas().replace(r'^\s*$', "Empty", regex=True))
    return ds

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

def model_init() -> SetFitModel:
    params = {"device": torch.device("cuda"), 'out_features': 6, 'temperature': 0}
    return SetFitModel.from_pretrained("BAAI/bge-small-en-v1.5",
                                       multi_target_strategy="multi-output", params=params, labels=features)


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_DISABLED"] = "true"

os.system("pip install setfit datasets sentence-transformers pandas numpy")

df_training: DataFrame = pd.read_excel("docs/training_docs/DummyClosingNotes-training.xlsx")  # Path del file Excel
df_validation: DataFrame = pd.read_excel("docs/training_docs/DummyClosingNotes-validation.xlsx")
df_test: DataFrame = pd.read_excel("docs/training_docs/DummyClosingNotes-test.xlsx")

# Rimozione di colonne non rilevanti per indici, singoli o per sequenza
train_dataset: Dataset = Dataset.from_pandas(df_training.iloc[:, 9:16])
validation_dataset: Dataset = Dataset.from_pandas(df_validation.iloc[:, 9:16])

test_dataset: Dataset = Dataset.from_pandas(df_test.iloc[:, [0, 1, 2, 3, 4, 5, 32, 49]])
features = train_dataset.column_names
features.remove("Closing Note")

train_dataset = clean_dataset(train_dataset)

validation_dataset = clean_dataset(validation_dataset)

test_dataset = clean_dataset(test_dataset)

train_dataset = get_templated_dataset(train_dataset, candidate_labels=features, sample_size=5, label_column="labels",
                                      multi_label=True, template="Il problema è del {}")


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

trainer = Trainer(
    model_init=model_init,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    column_mapping={"text": "text", "labels": "label"},
)

trainer.train()

metrics = trainer.evaluate()
print(metrics)

model = trainer.model
model.save_pretrained("models/DummyModel")  # Path del modello

model = SetFitModel.from_pretrained("models/DummyModel")

preds = model.predict(test_dataset['text'])
pred_proba = model.predict_proba(test_dataset['text'])

export_as_excel('closing_notes_output/ClosingNotesResults.xlsx', preds, test_dataset)

print_results(test_dataset, preds, pred_proba)

print("body_learning_rate: " + str(args.body_learning_rate))
print("num_epochs: " + str(args.num_epochs))
print("batch_size: " + str(args.batch_size))
print("warmup_proportion: " + str(args.warmup_proportion))

print(classification_report(test_dataset['labels'], preds, target_names=features, zero_division=0))
