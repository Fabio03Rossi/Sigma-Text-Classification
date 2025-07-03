### Programma per la valutazione totale di un numero N di campioni di report dei tecnici.
### Utilizzo di Thread da provare ma improbabile.
### Per il momento tutti i test useranno la tecnica Zero-Shot Classification, in seguito non solo.

## TODO: Implementare questa soluzione con supporto RAG e testarne gli effetti
## TODO: Riprovare l'uso della SmartLLMChain

from transformers import pipeline, Pipeline

from hybrid_text_classification import prompt_question

## Parametri aggiustabili   ------------------------------------------------------------------------

# Percentuale minima per considerare valida una classificazione
AREA_MIN_CONFIDENCE = 0.1
ACTION_MIN_CONFIDENCE = 0.05

CT = ["Cash Transport", "CT", "piatti", "LT", "ST", "PT"]
NE = ["Note Escrow", "NE", "nastro", "PE", "leva 10", "ME", "precas"]
NF = ["Note Feeder", "NF", "bocchetta", "leva 1", "sfogliatore", "SF", "PF", "MF", "stacker", "upper unit"]
NV = ["Node Validator", "NV", "leva 7", "n validator", "SNV", "UNV", "thickness", "Feeder"]
CASSETTE = ["Cassetto", "Cassetti", "RC", "AC", "DC", "MC"]
SHUTTER = ["Shutter"]

areas = CT + NE + NF + NV + CASSETTE + SHUTTER

NO_ACTION = ["nessuna azione"]
REPAIRED = ["riparazione", "riposizionat"]
REPLACED = ["sostituzione"]
CALIBRATED = ["calibrazione"]
CLEANED = ["pulizia", "rimozione"]

operations = NO_ACTION + REPAIRED + REPLACED + CALIBRATED + CLEANED

# Label da sistemare, anche il prompt poichè non affidabile ancora
state = ["Guasto presente", "fuori servizio", "Guasto non presente", "sconosciuto"]

## Definizione dei modelli  ------------------------------------------------------------------------

#MODEL = "Jiva/xlm-roberta-large-it-mnli"
#MODEL = "MoritzLaurer/bge-m3-zeroshot-v2.0"
#MODEL = "models/models--tasksource--deberta-small-long-nli/snapshots/deberta-small-long-nli/"

classifier: Pipeline = None

## Funzioni ------------------------------------------------------------------------

def classify_text(text, labels, prompt: str = "si parla di {}"):
    if classifier is not None:
        result = classifier(text, labels, hypothesis_template=prompt)
        return {label: score for label, score in zip(result["labels"], result["scores"])}
    else:
        raise Exception("Classifier not initialized.")


def parse_higher_result(classification):
    return max(classification.keys(), key=(lambda key: classification[key]))

def parse_multiple_results(classification, n_results: 1):
    evaluation = []

    for i in range(n_results):
        evaluation.append(list(classification.values())[i])

    return evaluation


# rimuoviamo tutti gli elementi sotto la soglia minima
def parse_results(classification, min_value):
    removals = []
    for entry in classification:
        if classification[entry] < min_value:
            removals.append(entry)
    for entry in removals:
        classification.pop(entry)
    return classification


def identify_area(text):
    classification = classify_text(text, areas,"La componente di cui si parla è {}")
    evaluation = parse_results(classification, AREA_MIN_CONFIDENCE)

    if len(evaluation) == 0:
        return ["UNK"]

    dict = {}

    for i in range(len(evaluation)):
        if i == 2:
            break
        match list(evaluation)[i]:
            case x if x in CT:
                dict.update({"CT": evaluation[x]})
            case x if x in NE:
                dict.update({"NE": evaluation[x]})
            case x if x in NF:
                dict.update({"NF": evaluation[x]})
            case x if x in NV:
                dict.update({"NV": evaluation[x]})
            case x if x in CASSETTE:
                dict.update({"CASSETTE": evaluation[x]})
            case x if x in SHUTTER:
                dict.update({"SHUTTER": evaluation[x]})

    return dict


def identify_action(text):
    classification = classify_text(text, operations, "è stata effettuata una {}")
    evaluation = parse_results(classification, ACTION_MIN_CONFIDENCE)
    if len(evaluation) == 0:
        return ["UNK"]

    return evaluation


def identify_state(text):
    classification = classify_text(text, state,"Il tecnico afferma un {}")
    evaluation = parse_higher_result(classification)

    if evaluation in "Guasto presente" or evaluation in "fuori servizio":
        return "Guasto presente", classification[evaluation]

    return "Guasto non presente", classification[evaluation]

def initialize(model, tokenizer):
    global classifier

    if tokenizer is None:
        classifier = pipeline(
            "zero-shot-classification",
            model=model,
            device="cpu",
            use_fast=True
        )
    else:
        classifier = pipeline(
            "zero-shot-classification",
            model=model,
            device="cpu",
            use_fast=True,
            tokenizer=tokenizer
        )

def complete_evaluation(text, area_confidence, action_confidence):
    #print(text)
    global AREA_MIN_CONFIDENCE, ACTION_MIN_CONFIDENCE

    AREA_MIN_CONFIDENCE = area_confidence
    ACTION_MIN_CONFIDENCE = action_confidence

    # Controlliamo la presenza di guasto
    state_result = identify_state(text)
    #print("-- " + state_result)
    #print(confidence1)

    # Controlliamo le aree affette
    area_result = identify_area(text)
    #print("-- " + area_result)
    #print(confidence2)

    # Controlliamo le azioni effettuate
    action_result = identify_action(text)
    #print("-- " + action_result)
    #print(confidence3)
    prompt_question(text)

    return text, state_result, area_result, action_result