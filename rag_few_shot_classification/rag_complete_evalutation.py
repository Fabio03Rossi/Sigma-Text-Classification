import time

import numpy as np
import pandas as pd
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

from rag_few_shot_classification.Classifier import Classifier
from rag_few_shot_classification.RAGDocument import RAGDocument

## TODO: Implementare questa soluzione con supporto RAG e testarne gli effetti

## Parametri aggiustabili   ------------------------------------------------------------------------

CT = ["Cash Transport", "CT", "piatti", "LT", "ST", "PT"]
NE = ["Note Escrow", "NE", "nastro", "PE", "leva 10", "ME", "precas", "escrow"]
NF = ["Note Feeder", "NF", "bocchetta", "leva 1", "sfogliatore", "SF", "PF", "MF", "stacker", "upper unit"]
NV = ["Note Validator", "NV", "leva 7", "n validator", "SNV", "UNV", "thickness", "Feeder"]
CASSETTE = ["Cassetto", "Cassetti", "RC", "AC", "DC", "MC"]
SHUTTER = ["Shutter"]

areas = CT + NE + NF + NV + CASSETTE + SHUTTER + ["Sconosciuto"]

NO_ACTION = ["nessuna azione"]
REPAIRED = ["riparazione", "riposizionat"]
REPLACED = ["sostituzione"]
CALIBRATED = ["calibrazione"]
CLEANED = ["pulizia", "rimozione"]

operations = NO_ACTION + REPAIRED + REPLACED + CALIBRATED + CLEANED

# Label da sistemare, anche il prompt poichè non affidabile ancora
state = ["ATM Guasto sì", "ATM Guasto no", "ATM Fuori Servizio", "ATM in errore"]

## Funzioni ------------------------------------------------------------------------

def file_as_string(filename):
    df = pd.read_excel(filename, index_col=None, na_values=['NA'])
    return df.to_string(index=False)

def get_string_docs(text):
    string = ""
    for item in result.vectorstore.max_marginal_relevance_search(text):
        string += item.page_content + "\n"
    return "<context>" + string + "</context>"


def rag_retriever(documents, embedding):
    """Create the vector database and the retriever to execute RAG."""

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding
    )
    retriever = vector_store.as_retriever()

    return retriever


def parse_higher_result(classification):
    return max(classification.keys(), key=(lambda key: classification[key]))


def identify_area(text, classifier, area_confidence):
    classification = classifier.classify(text, areas, True, "Hey! Si parla di {}")
    evaluation = classifier.parse_results(classification, area_confidence)

    if len(evaluation) == 0:
        return {"UNK"}

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


def identify_action(text, classifier, action_confidence):
    classification = classifier.classify(text, operations, True, "Hey! Stai attento! L'azione {} è stata effettuata")
    evaluation = classifier.parse_results(classification, action_confidence)
    if len(evaluation) == 0:
        return {"UNK"}

    return evaluation


def identify_state(text, classifier):
    #classification = classifier.classify(text, state, file_as_string("../docs/ClosingNotes - Copy.xlsx") + "\nIl tecnico afferma un {}")
    classification = classifier.classify(text, state, False)
    evaluation = parse_higher_result(classification)
    if classification[evaluation] < 0.0:
        return {"UNK"}

    return {evaluation: classification[evaluation]}

    """
    from llama_cpp import Llama

    llm = Llama.from_pretrained(
        repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        filename="Meta-Llama-3.1-8B-Instruct-Q4_0_8_8.gguf",
    )

    smart_llm = LlamaCpp(
        model_path="../models/Meta-Llama-3.1-8B-Instruct-Q4_0_4_4.gguf",
        temperature=0.1,
        n_ctx=65636,
        verbose=False,
        max_tokens=500,
        stop=["--!"]
    )

    template = ""
    Tell me if in the following input the mentioned machine was functioning or not before the report.
    The text is in italian and terms like FS refer to "fuori servizio".
    <input>{input}</input>
    ""
    prompt = PromptTemplate(
        template=template,
        input_variables=["input"],
    )

    chain = prompt | smart_llm
    return chain.invoke({"input": text})"""


def complete_evaluation(text, classifier, area_confidence, action_confidence):

    # Controlliamo la presenza di guasto
    state_result = identify_state(text, classifier)

    # Controlliamo le aree affette
    area_result = identify_area(text, classifier, area_confidence)

    # Controlliamo le azioni effettuate
    action_result = identify_action(text, classifier, action_confidence)

    return text, state_result, area_result, action_result


def print_and_w(file, text):
    text = str(text)
    print(text)
    print(text, file=file)


def run_model(classifier, df, logname, area_confidence, action_confidence):
    count = 1
    with open(logname + ".txt", 'w') as f:
        print_and_w(f, "AREA_CONF: " + str(area_confidence))
        print_and_w(f, "ACTION_CONF: " + str(action_confidence))
        for row_name, row in df.iterrows():
            st1 = row['Closing Note']

            print_and_w(f, "\n-------------------------------")
            print_and_w(f, f"\n++{count}++\n")

            start_time = time.time()
            text, state_result, area_result, action_result = complete_evaluation(str(st1), classifier, area_confidence, action_confidence)
            end_time = time.time()

            print_and_w(f, text + "\n\n" + str(state_result) + "\n" + str(area_result) + "\n" + str(action_result))

            print_and_w(f, f"Time taken: {end_time - start_time:.2f} seconds")
            count += 1
            if count > 10:
                break

        print_and_w(f, "\n-------------------------------\n")

        text, state_result, area_result, action_result = complete_evaluation("""
                ATM fuori servizio. Errore 5306 permane dopo cambio sfogliatore inferiore per cinghia rotto. 
                Sostituto nf superiore, anomalia permane, sostituita scheda upper e anomalia permane, infine provato a sostituire la scheda lower e l'anomalia Ã© rimasta. 
                Trovato cavo connettore sfogliatore superiore tranciato. Recuperato cablaggio superiore, sostituito cablaggio, prove massive da tool con esito positivo. 
                ATM in servizio con saltuarie anomalie di lettura carta per problema giÃ  noto dell'host intesa sanpaolo. FunzionalitÃ  atm ripristinate.
                """, classifier, area_confidence, action_confidence)

        print_and_w(f, text + "\n\n" + str(state_result) + "\n" + str(area_result) + "\n" + str(action_result))

        print_and_w(f, "\n-------------------------------\n")

        text, state_result, area_result, action_result = complete_evaluation("""
                Gr versamento e prelievo fuori servizio causa cinghie disallineate..ripristinato allineamento in sede cinghie del note escrow. verifica di tutte le parti, sostituito n.1 piatto sul cash transport. Eseguite prove funzionamento con esito positivo
                """, classifier, area_confidence, action_confidence)

        print_and_w(f, text + "\n\n" + str(state_result) + "\n" + str(area_result) + "\n" + str(action_result))

        print_and_w(f, "\n-------------------------------\n")

        text, state_result, area_result, action_result = complete_evaluation("""
                ATM trovato con errore nei versamenti, trovato errore cassetto pieno a livello software, controllati cassetti e verificato funzionamento NV, nessuna anomalia riscontrata, cassetti non pieni, 
                eseguito controllo log riscontrato un errore per cassetto pieno in 5gg di funzionamento, riavviato PC ATM, eseguite diverse prove con esito positivo, ATM in regolare funzionamento.
                """, classifier, area_confidence, action_confidence)

        print_and_w(f, text + "\n\n" + str(state_result) + "\n" + str(area_result) + "\n" + str(action_result))

        print_and_w(f, "\n-------------------------------\n")

        text, state_result, area_result, action_result = complete_evaluation("""
                Apparato trovato spento, il direttore l'ha spento in modo brutale, compromettendo la configurazone del sistema, inoltre il monitor esterno non veniva rilevato. 
                Rieseguete le configurazioni e ripristino del monitor. 
                Una volta eseguito il reset anche sul dispensatore non sisono rilevati problemi. Test funzionali con esito positivo. Eseguiti prelievi e versamenti.
                """, classifier, area_confidence, action_confidence)

        print_and_w(f, text + "\n\n" + str(state_result) + "\n" + str(area_result) + "\n" + str(action_result))


document = RAGDocument("../docs/ClosingNotes - Copy.xlsx").get_documents()

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    cache_folder="../models/"
)

result = rag_retriever(document, embeddings)


def main():
    while True:
        selection = input("Seleziona un modello:\n" +
                          "1 - Jiva/xlm-roberta-large-it-mnli\n" +
                          "2 - MoritzLaurer/bge-m3-zeroshot-v2.0\n" +
                          "3 - tasksource/deberta-small-long-nli\n")
        try:
            selection = int(selection)
        except ValueError:
            print("Valore non valido")
            continue
        if selection - 1 in range(3):
            break

    match selection:
        case 1:
            model = ["Jiva/xlm-roberta-large-it-mnli", 0.45, 0.9]
        case 2:
            model = ["MoritzLaurer/bge-m3-zeroshot-v2.0", 0.15, 0.09]
        case 3:
            model = ["tasksource/deberta-small-long-nli", 0.01, 0.01]
        case _:
            print("Selezionando Jiva come base")
            model = "Jiva/xlm-roberta-large-it-mnli"

    classifier = Classifier(model[0])

    df = pd.read_excel("../docs/ReportFeedbacks_2025-05-12_11-25.xlsx", index_col=None, na_values=['NA'])
    df.replace(np.nan, 0, inplace=True)

    run_model(classifier, df, "general", model[1], model[2])

if __name__ == "__main__":
    main()

