# Zero-Shot Text Classification

Riprendendo la task definita già nell'approccio SetFit, si definisce una metodologia di classificazione delle 6 aree di guasto attraverso pipeline di classificazione di **Transformers**.

Con questa metodologia non si vanno a costruire precedentemente dei dataset, ma la task di classificazione è operativa fin da subito. Si utilizza la libreria **Transformers** per instanziare una pipeline definita per questa task, e attraverso l'uso di modelli di piccole dimensioni si esegue una classificazione di testo senza esempi impiegati.

Dato un input infatti il modello si porrà una "ipotesi" la cui risposta corrisponderà alla classificazione della label. Questa ipotesi può essere personalizzata.


```
import os

os.system("pip install transformers")
```




    0



Definiamo il modello che verrà usato per la classificazione e un valore di confidenza minimo per poter considerare valida una classificazione. Questo perchè il risultato che otterremo non consiste in delle vere e proprie scelte del modello, ma in un valore di confidenza per ogni label definita ordinati in ordine decrescente. Spetta a noi decidere cosa è valido o meno.


```
# Modello italiano valido
MODEL = "Jiva/xlm-roberta-large-it-mnli"

# Percentuale minima per considerare valida una classificazione
MIN_CONFIDENCE = 0.13
```

Si definiscono delle liste di label assegnate ad ogni area, includendo tutte quelle label che risultano essere sinonimi di quella determinata area. Si va anche a definire una lista completa che daremo al modello.


```
# Lista di label associata a ciascun modulo
CT = ["Cash Transport", "CT", "piatti", "LT", "ST", "PT"]
NE = ["Note Escrow", "NE", "nastro", "PE", "leva 10", "ME", "precas"]
NF = ["Note Feeder", "NF", "bocchetta", "leva 1", "sfogliatore", "SF", "PF", "MF", "stacker"]
NV = ["Node Validator", "NV", "leva 7", "n validator", "SNV", "UNV", "thickness", "Feeder"]
CASSETTE = ["Cassett", "RC", "AC", "DC", "MC"]
SHUTTER = ["Shutter"]

areas = CT + NE + NF + NV + CASSETTE + SHUTTER
```

Qui di seguito si definiscono quelle che sono le funzioni per ogni operazione che si andrà a svolgere.

_`classify_text`_ è la funzione che si occupa di effettuare la vera e propria classificazione ed il parsing immediato del risultato ottenuto. Prende come parametro opzionale il valore di "ipotesi" che il modello userà per la classificazione, che se correttamente definito può portare a risultati migliori.

_`parse_higher_result`_ invece estrapola il risultato con confidenza maggiore, e tutti i risultati la cui confidenza è maggiore di quella definita precedentemente.
Delimita inoltre la classificazione alle sole aree base mappando la label riconosciuta con la rispettiva lista di appartenenza.

_`elaborate_prompt`_ richiama le funzioni descritte ed effettua il parsing del risultato in formato JSON.


```
# Zero-Shot Text Classification
def classify_text(text, labels, prompt: str = "si parla di {}"):
    result = classifier(text, labels, hypothesis_template=prompt)
    return {label: score for label, score in zip(result["labels"], result["scores"])}

# Elaborazione del risultato
def parse_higher_result(classification, text):
    evaluation = max(classification.keys(), key=(lambda key: classification[key]))

    print(str(classification))

    # Se il modello non è abbastanza sicuro, si considera sconosciuto
    if classification[evaluation] < MIN_CONFIDENCE:
        return "UNK"

    if evaluation in CT:
        return "CT"
    elif evaluation in NE:
        return "NE"
    elif evaluation in NF:
        return "NF"
    elif evaluation in NV:
        return "NV"
    elif evaluation in CASSETTE:
        return "CASSETTE"
    elif evaluation in SHUTTER:
        return "SHUTTER"

    return "UNK"

# Esecuzione totale del prompt
def elaborate_prompt(text):
    if text.strip() == "" or text.strip() is None or text.strip() == '' or text.strip() == '0':
        return """{"selection": "UNK"}"""

    classified = classify_text(text, areas, "La componente di cui si parla è {}")

    result = '{"selection": "' + parse_higher_result(classified, text) + '"}' + "\n"
    return result
```

Per l'inferenza si definisce l'oggetto classificatore attraverso la pipeline di **Transformers**.
Si specifica la task che in questo caso è zero-shot-classification, e altri parametri come il modello e la specifica dell'utilizzo della GPU.


```
from transformers import pipeline

# Rimane di gran lunga il metodo più affidabile in generale, il metodo riflessivo è troppo confuso
classifier = pipeline(
    "zero-shot-classification",
    model=MODEL,
    device="cuda",
    use_fast=True
)
```

    Device set to use cuda
    

Infine semplicemente richiamiamo la funzione elaborate_prompt che dato un testo da classificare, ritornerà la selezione parsata.


```
elaborate_prompt("Riscontrata banconota inceppata nel CT. Rimozione e test ok")
```

    {'CT': 0.16946330666542053, 'Cash Transport': 0.04899828881025314, 'LT': 0.03893483802676201, 'Note Escrow': 0.03788568824529648, 'PT': 0.037771668285131454, 'Cassett': 0.03605511784553528, 'bocchetta': 0.03446636348962784, 'SNV': 0.03099001757800579, 'NE': 0.02866402640938759, 'NV': 0.02690412662923336, 'MF': 0.02689838595688343, 'precas': 0.026827262714505196, 'UNV': 0.026514099910855293, 'piatti': 0.02645251527428627, 'ST': 0.024740444496273994, 'thickness': 0.024396777153015137, 'NF': 0.024251079186797142, 'MC': 0.0228794626891613, 'PE': 0.022639406844973564, 'PF': 0.021710975095629692, 'RC': 0.021312743425369263, 'SF': 0.02100963704288006, 'stacker': 0.02020789310336113, 'Note Feeder': 0.019959069788455963, 'Node Validator': 0.019108954817056656, 'sfogliatore': 0.018979569897055626, 'Feeder': 0.01850590482354164, 'Shutter': 0.018345283344388008, 'DC': 0.016499020159244537, 'leva 1': 0.01591811329126358, 'n validator': 0.015762493014335632, 'AC': 0.015050403773784637, 'leva 7': 0.01154065690934658, 'ME': 0.01110775675624609, 'leva 10': 0.010315688326954842, 'nastro': 0.008932951837778091}
    




    '{"selection": "CT"}\n'


