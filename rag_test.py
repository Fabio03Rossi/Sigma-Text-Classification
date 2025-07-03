import time

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader, UnstructuredExcelLoader, HuggingFaceModelLoader
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_community.vectorstores import Chroma
from langchain_experimental.smart_llm import SmartLLMChain
from langchain_huggingface import HuggingFaceEmbeddings
from llama_cpp import LlamaGrammar
from llama_cpp.server.types import max_tokens_field
from transformers import pipeline, PreTrainedModel, AutoModelForSequenceClassification


# Usa il loader corretto in base al tipo di file
def get_loader(location):
    if location.endswith(".csv"):
        return CSVLoader(location)
    elif location.endswith(".xlsx"):
        return UnstructuredExcelLoader(location)
    return None


# Funzione per la stampa e l'elaborazione di un prompt con uso di RAG
def prompt_question(string):
    if string.strip() == "" or string.strip() is None or string.strip() == '':
        return """{ "selection": "UNK"}"""
    start_time = time.time()
    #result = chain.invoke({"input": string, "context": retriever.invoke(string)})
    result = chain.invoke({"input": string})
    print(result)
    end_time = time.time()

    print(f"Question: {""}")
    print(f"Answer: {result['resolution']}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    #print("\nSource documents:")
    #for i, doc in enumerate(result["context"]):
    #    print(f"Document {i + 1}:")
    #    print(f"Content: {doc.page_content[:500]}.\n")
    return result['resolution'].strip()

doc_location = "docs/ReportFeedbacks_2025-05-12_11-25.xlsx"

# Spezziamo il contenuto dei documenti
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=500,
    length_function=len
)

# Carichiamo il dataset da interrogare in base alla colonna del file Excel (molteplici colonne o documenti possono essere aggiunti)
loader = get_loader(doc_location)
documents = []
documents.extend(loader.load_and_split(text_splitter))

# Inizializziamo i modelli LLM per l'inferenza e gli embedding
llm = LlamaCpp(
    model_path="models/Llama-3.2-3B-Instruct-Q4_0.gguf",
    verbose=False,
    n_gpu_layers=-1,
    temperature=0.1,
    n_ctx=65636,
    grammar=LlamaGrammar.from_json_schema("""{
          "$schema": "https://json-schema.org/draft/2020-12/schema",
          "type": "object",
          "properties": {
            "selection": {
              "type": "string",
              "enum": ["UNK", "NV", "NF", "CT", "NE", "CASSETTE", "SHUTTER"]
            }
          },
          "required": ["selection"]
        }"""
                                          ),
    max_tokens=50,
    stop=["--!"]
)

idea_llm = LlamaCpp(
    model_path="models/Llama-3.2-3B-Instruct-Q4_0.gguf",
    verbose=False,
    n_gpu_layers=-1,
    temperature=0.8,
    n_ctx=65636,
    grammar=LlamaGrammar.from_json_schema("""{
          "$schema": "https://json-schema.org/draft/2020-12/schema",
          "type": "object",
          "properties": {
            "selection": {
              "type": "string",
              "enum": ["UNK", "NV", "NF", "CT", "NE", "CASSETTE", "SHUTTER"]
            }
          },
          "required": ["selection"]
        }"""
                                          ),
    max_tokens=50,
    stop=["--!"]
)

smart_llm = LlamaCpp(
    model_path="models/llama-3.1-8b-instruct-q4_0.gguf",
    temperature=0.1,
    n_ctx=65636,
    verbose=False,
    max_tokens=500,
    stop=["--!"]
)

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    cache_folder="models/"
)

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings
)

# ----------------------------------------------------------------

# Definiamo la struttura del prompt RAG per iniettare la informazioni di contesto
# Questo prompt è riuscito a passare tutti i semplici test iniziali che gli sono stati proposti
template = """
You are a Text Classificator. Your job is to select the best classification type on damage reports using the selections: UNK, CASSETTE, CT, NE, NF, NV, SHUTTER.
CT stands for "Cash Transport"
NE stands for "Node Escrow"
NF stands for "Node Feeder"
NV stands for "Node Validator"
Based on the following context, classify the following note by first summarizing it efficiently, and then decide based on its content.
If the input is empty or doesn't contain expected words, you HAVE TO return UNK.
Let's think step by step, but only keep a minimum draft for each step, with 5 words at most.
It's extremely important.

Context:
{context}

Keep in mind the keywords in question have synonyms and you must use these to answer the question.
Before evaluating the classification, look up each synonym as single words to decide.
In case of conflicting inputs, read carefully and correctly decide the classification.
Remember, try to understand the core issue of the report before answering.

Synonyms per selection:
CASSETTE = ["cassetto", "cassetti", "cassette" "RC", "AC", "DC", "MC"]
then
CT = ["ct", "cash transport", "cash", "piatto", "piatti", "LT", "ST", "PT", "transport", "TRASPORT"]
then
NE = ["ne", "LEVA 7", "PRECASSA", "PREGASSA", "NASTRO", "escrow", "ecrow"]
then
NF = ["nf", "sfogliatore", "leva 1", "FEEDER", "FEEDEER", "BOCCHETTA"]
then
NV = ["nv", "SNV", "UNV", "leva 7", "VALIDATOR", "LETTORE", "THICKNESS", "validatore"]
then
SHUTTER = ["shutter"]
then
UNK = []

Note: 'banconote' does not equal to 'cash transport' in this case.

Input: "MTA IN SERVIZIO. Anomalia risolta in autonomia dalla filiale. Effettuate verifiche hardware e controllo LOG. Funzionalità ok."
Summary: "verifiche hardware e software"
Thinking: Doesn't contain any matching word or synonym.
Output: UNK

Input: "Sono Luigi e mi piace il cioccolato."
Thinking: The type input is not expected.
Output: UNK

Input: "Ritrovato CT con cinghia rotta causa usura. Sostituito Ct. Test ok. ATM in servizio"
Summary: "CT rotto e sostituito"
Thinking: it contains the synonym 'Ct'.
Output: CT

Input: "sostituzione validatore banconote, reset e test ok"
Summary: "validatore sostituito"
Thinking: it contains the synonym 'validatore'.
Output: NV

Input: "Trovato dispensatore bloccato con banconote all'interno e errore su nv. Sostituito validatore ed effettuati test, numerose banconote scartate che arrivavano storte. Effettuata pulizia link transport e ricalibrazione. Test dispensazione e versamento ok. Lasciata macchina da gestire per numerose banconote in ac"
Summary: "nv in errore con dispensatore bloccato. validatore sostituito"
Thinking: it contains the synonym 'nv'.
Output: NV

Input: "ATM fuori servizio, nessuna banconota inceppata, errore 332f errore sui sensori dell'nv. Sostituito nv, eseguita pulizia analisi log atm bloccato da questo errore. Eseguite prove con clientela intesa sanpaolo con risultati positivi. ATM in regolare servizio"
Summary: "sensori nv in errore. nv sostituito"
Thinking: it contains the synonym 'nv'.
Output: NV

Input: "Trovato atm con versamento e dispensazione banconote fuori servizio, riscontrato grg con modulo escrow in errore: nastro fuoriuscito e cinghie usurate, sostituito modulo escrow, configurato periferica, rimesso in servizio atm e verificato funzionamento con operazioni clienti banca"
Summary: "modulo escrow in errore e sostituito"
Thinking: it contains the synonym 'escrow'.
Output: NE

Input: "G.R. dispensatore banconote fuori servizio rimosso elastico dalle cinghie dell escrow ripristinate cinghie. Eseguiti controlli sostituito piatto cassetto AC Eseguiti controlli e test finali con esito positivo macchina nuovamente funzionante"
Summary: "modulo escrow in errore e sostituito"
Thinking: it contains the synonym 'escrow'.
Output: NE

Input: "Banconote €20 incastrata nel NF superiore. Rinvenuta moneta nei rulli dello sfogliatore. Rimossa banconota e corpo estraneo. Test di funzionamento ATM con clientela, ok."
Summary: "NF con banconota incastrata"
Thinking: it contains the synonym 'NF'.
Output: NF

Input: "All'arrivo in filiale si riscontra sfogliatore inferiore con cinghia usurata ."
Summary: "sfogliatore usurato"
Thinking: it contains the synonym 'sfogliatore'.
Output: NF

Input: "Sostituzione cassetti DC fit e unfit. Test ok"
Summary: "cassetti DC sostituiti"
Thinking: it contains the synonyms 'cassetti' and 'DC'.
Output: CASSETTE

Input: "Gr: ATM trovato in servizio. Trovate banconote accartocciate nel cassetto AC. Sostituito cassetto AC ed effettuate prove di dispensazione e versamento andate a buon fine. Rimesso in servizio ATM e fatte prove con cassiere positive. "
Summary: "cassetto AC sostituito"
Thinking: it contains the synonyms 'AC' and 'cassetto'.
Output: CASSETTE

Input: "{input}"
"""
prompt = PromptTemplate(
    template=template,
    input_variables=["input", "context"],
)

# ----------------------------------------------------------------

alt_template = """
UNK stands for "Unknown"
CASSETTE stands for "Cassette"
CT stands for "Cash Transport"
NE stands for "Node Escrow"
NF stands for "Node Feeder"
NV stands for "Node Validator"
SHUTTER stands for "Shutter"
Classify the following note based on its content like this:
-- UNK --!

If unsure, look for the most possible answer.
It's extremely important for my career.

Keep in mind the keywords in question have synonyms and you must use these to answer the question.
Before evaluating the classification, look up each synonym as single words to decide.
In case of conflicting inputs, read carefully and correctly decide the classification.

Synonyms per selection:
CASSETTE = ["cassetto", "cassetti", "cassette" "RC", "AC", "DC", "MC"]
then
CT = ["ct", "cash transport", "cash", "piatto", "piatti", "LT", "ST", "PT", "transport", "TRASPORT"]
then
NE = ["ne", "LEVA 7", "PRECASSA", "PREGASSA", "NASTRO", "escrow", "ecrow"]
then
NF = ["nf", "sfogliatore", "leva 1", "FEEDER", "FEEDEER", "BOCCHETTA"]
then
NV = ["nv", "SNV", "UNV", "leva 7", "VALIDATOR", "LETTORE", "THICKNESS", "validatore"]
then
SHUTTER = ["shutter"]
then
UNK = []

Input: "Sono Luigi e mi piace il cioccolato."
Thinking: The type input is not expected.
Output: UNK

Input: "Ritrovato CT con cinghia rotta causa usura. Sostituito Ct. Test ok. ATM in servizio"
Summary: "CT rotto e sostituito"
Thinking: it contains the synonym 'Ct'.
Output: CT

Input: "sostituzione validatore banconote, reset e test ok"
Summary: "validatore sostituito"
Thinking: it contains the synonym 'validatore'.
Output: NV

Input: "All'arrivo in filiale si riscontra sfogliatore inferiore con cinghia usurata ."
Summary: "sfogliatore usurato"
Thinking: it contains the synonym 'sfogliatore'.
Output: NF

Input: "Sostituzione cassetti DC fit e unfit. Test ok"
Summary: "cassetti DC sostituiti"
Thinking: it contains the synonyms 'cassetti' and 'DC'.
Output: CASSETTE

Input: "{input}"
Output:
"""

alt_prompt = PromptTemplate(
    template=alt_template,
    input_variables=["input"],
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={'k': 3, 'fetch_k': 25}
)

chain = SmartLLMChain(
    llm=llm,
    critique_llm=smart_llm,
    ideation_llm=idea_llm,
    prompt=alt_prompt,
    n_ideas=5,
    verbose=True
)

combine_docs_chain = create_stuff_documents_chain(llm, prompt)
rag_pipeline = create_retrieval_chain(retriever, combine_docs_chain)

#chain.prep_inputs(rag_pipeline)
