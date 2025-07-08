#%% md
# # Classificazione di testo con LLM e LangChain
# 
# Riprendendo la task definita già nell'approccio SetFit, si definisce una metodologia di classificazione delle 6 aree di guasto attraverso inferenza di **LLM** e l'uso di **Prompt Engineering**.
# 
# Con questa metodologia non si vanno a costruire precedentemente dei dataset, ma la task di classificazione è operativa fin da subito. Attraverso l'uso diretto di LLM Open Source si effettua l'inferenza manipolando il prompt affinchè il modello risponda alle esigenze da soddisfare.
# 
# Questo significa che i modelli utilizzati sono più grandi e tipicamente sono quantizzati per un utilizzo di risorse minore cercando di mantenerne i benefici.
# 
# Un approccio anche più semplice da applicare ma con risultati non ottimali in task più complesse simili a quelle definite.
# 
# Il primo step è inizializzare il modello (in questo caso [HuggingFaceTB/SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct)), definirne la task di generazione di testo, e poi aggiungere altri parametri aggiuntivi.
# 
# _Nota_: in questo caso si fa uso della classe _`HuggingFacePipeline`_ per ottenere la pipeline del modello, ma nelle sperimentazioni si è usata la classe _`LlamaCpp`_ con LLM quantizzati scaricati localmente.
#%%
from langchain_huggingface import HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 100}
)
#%% md
# Successivamente si costruisce la struttura del prompt che verrà impiegata ad ogni inferenza. Sarà qui che si guiderà il modello allo svolgimento della task.
# 
# Il prompt consiste in una lunga stringa di testo che definisce tutte le label con relativi sinonimi e istruisce il modello sulla task da svolgere in inglese (tipicamente porta a risultati migliori), a cui poi viene aggiunto il parametro di input dove verranno dinamicamente inserite le note di chiusura per la valutazione.
# 
# La manipolazione del prompt è estremamente flessibile per via della sua natura, e per questo è possibile applicare numerose tecniche per ridurre allucinazioni e ragionamenti fallati che il modello potrebbe derivare. Le tecniche principali utilizzate durante la sperimentazione sono:
# * **Few-Shot Prompting** (consiste nel proporre al modello diversi esempi prima del vero input)
# * **Self-Critique** (3 step: il modello inizialmente deriva 3 risposte, che lui stesso poi valuta e confronta all'input originale. Infine deriva una risposta dalle considerazioni fatte)
# * **Retrieval-augmented generation (RAG)** (sistema a supporto del prompt che attraverso l'input interroga delle informazioni date in precedenza, in base alla similarità, e le aggiunge come fonte di informazione)
# * **Zero-shot Chain-Of-Thoughts Prompting** (si aggiungono espressioni di ragionamento al modello per aumentarne la logica, ad esempio "pensiamo passo per passo" oppure ragionamenti specifici per la task di applicazione)
# 
# Le tecniche di Prompt Engineering sono molte, di cui la gran parte documentate qui: [Prompt Engineering Guide](https://www.promptingguide.ai/techniques)
# 
#%%
from langchain_experimental.smart_llm import SmartLLMChain
from langchain_core.prompts import PromptTemplate

template = """
CASSETTE stands for "Cassette", "Cassetto", "AC", "RC"
then
CT stands for "Cash Transport", "Piatto", "Piatti", "LT", "ST", "PT"
then
NE stands for "Node Escrow", "precassa", "nastro"
then
NF stands for "Node Feeder", "bocchetta", "leva 1", "sfogliatore"
then
NV stands for "Node Validator", "leva 7", "validatore"
then
SHUTTER stands for "Shutter"

Your job is to classify the following input with the labels above defined.
You only answer with one single phrase of 50 words and the resulting classification, no more no less.
If you're unsure, return UNK.
After the answer write "end".

Input:
{input}

Response:
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["input"],
)
#%% md
# Ora si va a definire la catena LangChain per eseguire poi l'inferenza.
# 
# Le catene sono dei tool definiti da LangChain per poter effettuare manipolazioni precise e concatenabili l'una con l'altra sul prompting del modello.
# In questo caso utilizziamo la catena sperimentale _`SmartLLMChain`_ che in breve si occupa di applicare la tecnica di prompting Self-Critique.
# 
# Quando infatti questa catena verrà invocata verranno sempre eseguiti i 3 step precedentemente descritti. Possiamo definire diversi parametri per personalizzare questo comportamento tra cui il numero di idee preliminari, eventuali diversi LLM per i singoli step ed altro ancora.
#%%
smart_chain = SmartLLMChain(
    llm=llm,
    prompt=prompt,
    n_ideas=2,
    verbose=True
)
#%% md
# Non rimane che costruire la funzione per l'inferenza invocando la catena.
#%%
def prompt_question(string):
    return smart_chain.invoke({"input": string})
#%% md
# Ed infine effettuare la vera e propria inferenza su un esempio di nota. Da notare anche che possiamo manipolare la grammatica di output di un modello llama_cpp attraverso LangChain, aprendo la possibilità di avere output che seguono Schema JSON o altro.
#%%
prompt_question("Riscontrata banconota inceppata nel CT. Rimozione e test ok")