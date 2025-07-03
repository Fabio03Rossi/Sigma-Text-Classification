from langchain.output_parsers import RegexParser, BooleanOutputParser
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_core.prompts import PromptTemplate
from llama_cpp import LlamaGrammar
from transformers import pipeline, TextStreamer, AutoTokenizer

import json
from hybrid_text_classification import prompt_question

#nli_model = XLMRobertaModel.from_pretrained('joeddav/xlm-roberta-large-xnli')
#tokenizer = XLMRobertaTokenizer.from_pretrained('joeddav/xlm-roberta-large-xnli')

# PARAMETRI

#MODEL = "reddgr/zero-shot-prompt-classifier-bart-ft"
#MODEL = "reddgr/MoritzLaurer/bge-m3-zeroshot-v2.0"
#MODEL = "joeddav/xlm-roberta-large-xnli"

# Modello italiano valido
MODEL = "Jiva/xlm-roberta-large-it-mnli"
#tokenizer = AutoTokenizer.from_pretrained(MODEL)


# Percentuale minima per considerare valida una classificazione
MIN_CONFIDENCE = 0.13

# Lista di label associata a ciascun modulo
CT = ["Cash Transport", "CT", "piatti", "LT", "ST", "PT"]
NE = ["Note Escrow", "NE", "nastro", "PE", "leva 10", "ME", "precas"]
NF = ["Note Feeder", "NF", "bocchetta", "leva 1", "sfogliatore", "SF", "PF", "MF", "stacker"]
NV = ["Node Validator", "NV", "leva 7", "n validator", "SNV", "UNV", "thickness", "Feeder"]
CASSETTE = ["Cassett", "RC", "AC", "DC", "MC"]
SHUTTER = ["Shutter"]

areas = CT + NE + NF + NV + CASSETTE + SHUTTER

# -----------------------------------------------------

#NO_ACTION = ["No Action"]
#REPAIRED = ["ripara"]
#REPLACED = ["sostitu"]
#CALIBRATED = ["calibra", "taratura"]
#CLEANED = ["pulizia", "rimo"]

NO_ACTION = ["nessuna azione"]
REPAIRED = ["riparazione"]
REPLACED = ["sostituzione"]
CALIBRATED = ["calibrazione"]
CLEANED = ["pulizia", "rimozione"]

operations = NO_ACTION + REPAIRED + REPLACED + CALIBRATED + CLEANED

# Label da sistemare assieme a prompt poichè non affidabile ancora
guasto = ["Guasto presente", "Guasto non presente"]

# ESECUZIONE

# Rimane di gran lunga il metodo più affidabile in generale, il metodo riflessivo è troppo confuso
classifier = pipeline(
    "zero-shot-classification",
    model=MODEL,
    device="cpu",
    use_fast=True
)

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
        return json.loads(prompt_question(text))['selection']

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
    action = classify_text(text, operations, "è stata effettuata una {}")
    for score in action.keys():
        if action[score] >= MIN_CONFIDENCE:
            print(score + ': ' + str(action[score]))

    result = '{"selection": "' + parse_higher_result(classified, text) + '"}' + "\n" + str(action)
    return result

def action_elaboration(text):
    #temp_prompt = PromptTemplate(
    #    template="Il tuo compito è identificare esclusivamente se ci sono state sostituzioni nel seguente report. Rispondi secco con Si o No. Rispondi Si se ci sono state sostituzioni, No altrimenti. \n Report: {input}",
    #    input_variables=["input"]
    #)
    #is_broken = (temp_prompt | smart_llm | BooleanOutputParser(false_val="No", true_val="Si")).invoke({"input": text})
    #return is_broken
    return classify_text(text, guasto, """Il tecnico afferma un {}""")

# Noi vogliamo categorizzare la presenza di un guasto o meno, se vi è stato quali aree sono affette,
# e le azioni prese per risolvere il problema.