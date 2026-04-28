import json
import numpy as np
from datasets import load_dataset
from pprint import pprint

# DatasetDict({
# test: Dataset({
        # features: ['id', 'document', 'extractive_keyphrases', 'abstractive_keyphrases'],
        # num_rows: 2304
# })
# })
def load_krapiving():
    ds = load_dataset("midas/krapivin", "generation")

    print(ds) 
    print(ds["test"][0]["document"])
    print(ds["test"][0]["extractive_keyphrases"])

# clusters, sentences, ner, relations, doc_key
def load_SCIERC():
    path = "/home/dust/Downloads/processed_data/json/train.json"
    
    output = []
    
    with open(path, 'r') as f:
        for line in f.readlines():
            obj = json.loads(line)
            ner = obj["ner"]
            sentences_flat = [item for sublist in obj["sentences"] for item in sublist]
            
            cl = []
            for key in ner:
                for k in key:
                    cl.append(sentences_flat[k[0]:k[1]+1])
            
            output.append({
                "sentences": obj["sentences"],
                "entities": cl
            })
    
    with open("output.json", "w") as f:
        json.dump(output, f, indent=2)

load_SCIERC()

