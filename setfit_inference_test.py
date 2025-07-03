import numpy
import torch
from datasets import Dataset
from setfit import SetFitModel
from setfit_training import df_test
from setfit_training import remap_dataset

features = ["CASSETTE ground-truth", "CT ground-truth", "NE ground-truth",
            "NF ground-truth", "NV ground-truth", "SHUTTER ground-truth",
            "CRM9250 Excluded ground-truth"]

mappings = ["CASSETTE", "CT", "NE", "NF", "NV", "SHUTTER", "CRM9250 Excluded"]

state = ["Guasto", "Funzionante"]

def elaborate_prediction(i):
    result = []
    count = 0
    for k in i:
        if k == 1:
            result.append(str(mappings[count]))
        count += 1
    return result

def single_model_prediction(model, string_list):
    pred = model.predict(string_list)
    proba = model.predict_proba(string_list)
    return pred, proba

def execute_prediction(inputs):
    string_list = inputs.split("--")
    result1, proba1 = single_model_prediction(model1, string_list)
    #count = 0
    #for i in result1:
    #    i.append(proba1[count])
    #    count += 1

    return str(result1), str(proba1)

params = {"device": torch.device("cuda"), 'out_features': 2, 'temperature': 0.5}
model1 = SetFitModel.from_pretrained("hello/stateModel3", params=params)

labels = model1.labels
print(labels)

#test_dataset = remap_dataset(Dataset.from_pandas(df_test), features)
#test_dataset = Dataset.from_pandas(df_test)
#pred1 = model1.predict(test_dataset['Closing Note'])

#for i in pred1:
#    print(str(i))