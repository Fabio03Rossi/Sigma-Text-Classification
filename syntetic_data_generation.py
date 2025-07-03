import random

import pandas as pd
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_core.runnables import RunnableConfig
from llama_cpp import LlamaGrammar, Llama

def get_model():
    return Llama.from_pretrained(
        repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        filename="Meta-Llama-3.1-8B-Instruct-IQ2_M.gguf",
        verbose=False,
        n_gpu_layers=-1,
        temperature=0,
        n_ctx=65636,
        max_tokens=50,
        seed=random.randint(1, 100),
        stop=["Instruction:", "Report:"],
        streaming=True
    )

"""
model = pipeline(
                model="Jiva/xlm-roberta-large-it-mnli",
                device="cuda",
                use_fast=True
            )
"""

df = pd.read_excel("docs/training_docs/ClosingNotes-sample-20250623-training.xlsx")

string = """
    root   ::= value
    value  ::= cassette | ct | ne | nf | nv | shutter
    cassette ::= "CASSETTE"
    ct ::= "CT"
    ne ::= "NE"
    nf ::= "NF"
    nv ::= "NV"
    shutter ::= "SHUTTER"
    """

label_llm = Llama.from_pretrained(
    repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
    filename="Meta-Llama-3.1-8B-Instruct-IQ2_M.gguf",
    verbose=False,
    n_gpu_layers=-1,
    temperature=0,
    grammar=LlamaGrammar.from_string(string),
    streaming=True
)

note_prompt = """
Instruction: Forget everything.
You are an italian technician who works with ATMs.
Describe exclusively in one phrase under 80 words your last encountered issue.
These can include these areas = "CASSETTE" | "CT" | "NE" | "NF" | "NV" | "SHUTTER".
Do not repeat yourself, just follow the instructions.
It is extremely important for my career.

Report:
Eseguito sostituzione cassetto per molteplici errori. 
Riallineamento della parte lower,prove di movimentazione banconote con esito positivo 

Report:
Riscontrato MF inferiore con cinghia rotta causa inserimento corpo estraneo. Sostituzione NF inferiore. Test ok.

Report:
Nastro NE fuori posizione a seguito di un inceppamento. Riposizionato nastro NE. Riscontrato piatto CT RC2 con parti danneggiate, sostituito. Durante prove con tool, continue anomalie sul piatto del cassetto RC3. Sostituito cassetto RC3, anomalia rientrata. Prove massive di prelievo e versamento con tool, ok. Test di funzionamento ATM con clientela, ok.

Report:
"""

for i in range(0, 5):
    llm = get_model()

    output = llm.create_chat_completion(
      messages = [
          {
              "role": "system",
              "content": note_prompt
          }
        ],
      max_tokens=50,
      stop=["Instruction:", "Report:", "\n"],
    )
    print(output['choices'][0]['message']['content'])
    label = label_llm(str(output['choices'][0]['message']['content']),
    max_tokens = 32,  # Generate up to 32 tokens, set to None to generate up to the end of the context window
    stop = ["Instruction:", "Report:", "\n"],  # Stop generating just before the model would generate a new question
    echo = True  # Echo the prompt back in the output
    )

    print(label['choices'][0]['text'])
    cassette, ct, ne, nf, nv, shutter = 0, 0, 0, 0, 0, 0
    match label:
        case "CASSETTE":
            cassette, ct, ne, nf, nv, shutter = 1, 0, 0, 0, 0, 0
        case "CT":
            cassette, ct, ne, nf, nv, shutter = 0, 1, 0, 0, 0, 0
        case "NE":
            cassette, ct, ne, nf, nv, shutter = 0, 0, 1, 0, 0, 0
        case "NF":
            cassette, ct, ne, nf, nv, shutter = 0, 0, 0, 1, 0, 0
        case "NV":
            cassette, ct, ne, nf, nv, shutter = 0, 0, 0, 0, 1, 0
        case "SHUTTER":
            cassette, ct, ne, nf, nv, shutter = 0, 0, 0, 0, 0, 1


    new_row = pd.DataFrame({
        'Closing Note': output,
        'CASSETTE ground-truth': cassette,
        'CT ground-truth': ct,
        'NE ground-truth': ne,
        'NV ground-truth': nv,
        'NF ground-truth': nf,
        'SHUTTER ground-truth': shutter
    }, index=range(0, 1))

    df = pd.concat([df, new_row])

df.to_excel("docs/generated/ClosingNotesGeneratedSamples-3006.xlsx")

#for chunk in llm.stream(prompt):
#    print(chunk, end="", flush=True)
