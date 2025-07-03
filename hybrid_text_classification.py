import time

from langchain.chains.llm import LLMChain
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_experimental.fallacy_removal.base import FallacyChain
from langchain_experimental.fallacy_removal.models import LogicalFallacy
from langchain_experimental.smart_llm import SmartLLMChain
from llama_cpp import LlamaGrammar
from transformers import pipeline

candidate_labels = [
    "not a report", "Node Validator", "Note Feeder", "Cash Transport", "Note Escrow", "Cassette", "Shutter",
    "CT", "NE", "NF", "NV", "SNV", "sfogliatore", "leva 1", "leva 7", "leva 10", "validator", "bocchetta",
    "RC", "AC", "DC", "piatti", "LT", "ST", "PT", "nastro", "Cassetti", "Cassetto", "plate", "UNV",
    "SNV", "SF", "PF", "MF", "CL", "stacker", "thickness", "precas"
]

#model = "reddgr/zero-shot-prompt-classifier-bart-ft"
#model = "reddgr/MoritzLaurer/bge-m3-zeroshot-v2.0"
#model = "joeddav/xlm-roberta-large-xnli"

model = "facebook/bart-large-mnli"

classifier = pipeline("zero-shot-classification", model=model, device="cpu")

# Zero-Shot Text classification
def classify_text(text):
    result = classifier(text, candidate_labels)
    return {label: score for label, score in zip(result["labels"], result["scores"])}

# Approccio Smart LLM
def prompt_question(string):
    result = smart_chain.invoke({"input": string, "fallacy_revision_request": "Give an answer that meets better criteria."})

    #fallacy_prompt_question(string)

    #result = chain.invoke({"input": string})
    #print(result)
    #result = chain.invoke({"input": string + "\n\nIdea:\n" + result + "\n Does this answer meet the criteria? Give an answer that meets the criteria above."})
    #print(result)
    #result = json_chain.invoke({"input": string + "\n\nIdea:\n" + result})


    #print(f"Answer: {result['resolution']}")
    #return result['resolution'].strip()
    print(f"Answer: {result}")
    return result

def fallacy_prompt_question(string):
    fallacy = fallacy_chain.invoke({"input": string, "fallacy_revision_request": "Give an answer that meets better criteria."})

    print(fallacy)

    print("Fallacy")
    print(f"Answer: {fallacy}")
    return fallacy.strip()

def parse_higher_result(classification, text):
    evaluation = max(classification.keys(), key=(lambda key: classification[key]))

    print(str(classification))

    if classification[evaluation] < 0.13:
        #return prompt_question(text)
        return '{"selection": "UNK"}'

    if any(x in evaluation for x in ["Cash Transport", "CT", "piatti", "LT", "ST", "PT"]):
        return '{"selection": "CT"}'
    elif any(x in evaluation for x in ["Note Escrow", "NE", "nastro", "PE", "leva 10", "ME", "precas"]):
        return '{"selection": "NE"}'
    elif any(x in evaluation for x in ["Note Feeder", "NF", "bocchetta", "leva 1", "sfogliatore", "SF", "PF", "MF", "stacker"]):
        return '{"selection": "NF"}'
    elif any(x in evaluation for x in ["Node Validator", "NV", "leva 7", "n validator", "SNV", "UNV", "thickness"]):
        return '{"selection": "NV"}'
    elif any(x in evaluation for x in ["Cassette", "Cassetti", "Cassetto", "RC", "AC", "DC", "MC"]):
        return '{"selection": "CASSETTE"}'
    elif any(x in evaluation for x in ["Shutter"]):
        return '{"selection": "SHUTTER"}'
    return '{"selection": "UNK"}'


# Elaborazione totale del prompt
def elaborate_prompt(text):
    if text.strip() == "" or text.strip() is None or text.strip() == '' or text.strip() == '0':
        return """{"selection": "UNK"}"""

    start_time = time.time()
    classified = classify_text(text)
    result = parse_higher_result(classified, text)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    return result


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
    stop=["end"],
    streaming=True
)

idea_llm = LlamaCpp(
    model_path="models/Llama-3.1-8b-ITA-Q4_0.gguf",
    verbose=False,
    n_gpu_layers=-1,
    temperature=0.5,
    n_ctx=65636,
    max_tokens=300,
    stop=["end"],
    streaming=True
)

smart_llm = LlamaCpp(
    model_path="models/Llama-3.1-8b-ITA-Q4_0.gguf",
    temperature=0,
    n_ctx=65636,
    verbose=False,
    max_tokens=50,
    repeat_penalty=10,
    stop=["end"],
    streaming=True
)

"""
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

You must classify in a phrase the following note based on the above definitions.
Your final answer must match the definitions CASSETTE, CT, NE, NF, NV, SHUTTER, UNK. Anything different is wrong.
In case you're unsure return UNK.

Stay relevant to your task. Focus on word matching. GIVE ONLY 1 LIKELY ANSWER IN 50 WORDS MAX.
Use your own words.
When finished, write "End Idea"

Think step by step.
This is extremely important for my career.

Input:
{input}
"""

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
You only answer with one sinlge phrase of 50 words and the resulting classification, no more no less.
If you're unsure, return UNK.
After the answer write "end".

Input:
{input}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["input"],
)

chain = prompt | smart_llm

json_chain = prompt | llm

smart_chain = SmartLLMChain(
    llm=llm,
    critique_llm=smart_llm,
    ideation_llm=idea_llm,
    prompt=prompt,
    n_ideas=2,
    verbose=True
)

fallacy_chain = FallacyChain.from_llm(
    llm=smart_llm,
    chain=LLMChain(prompt=prompt, llm=smart_llm),
    logical_fallacies=[
        LogicalFallacy(
            fallacy_critique_request="Tell if this answer meets criteria.",
            fallacy_revision_request="Give an answer that meets better criteria."
        )
    ],
    verbose=True
)

