from dotenv import load_dotenv
from anthropic import Anthropic
from pathlib import Path
import json
import csv
import re
import ast

def call_antrophic_api(section_name: str, section_text: str):
# {{{
    client = Anthropic()
    TEMPLATE = 'Instruction: From the provided text give me a list of ALL of the concepts FROM THE TEXT that are relevant to the domain of "information retrieval" in the context of "' + section_name + '". Formatting rules: Respond only with a comma AND 1 space SEPARATED list of the relevant concepts. Do not include any additional information or formatting to your response. Input text: ' + section_text
    message = client.messages.create(
        max_tokens=1024,
        messages=[{"role": "user", "content": TEMPLATE}],
        model=MODEL,
    )

    print("API CALL SUCCESSFUL")
    print(message.content)
    return message.content[0].text

    # response = client.messages.count_tokens(
    # model="claude-haiku-4-5-20251001",
    # messages=[
    #     {"role": "user", "content": "Hello, Claude. Explain attention in 3 sentences."}
    # ],
    # )

    # print(response.input_tokens)
# }}}

def load_and_experiment_ds():
# {{{
    # section_number, section_name, text, source_url
    # path = "/home/dust/Downloads/prereq/scripts/IIR/iir_book.json"
    path = Path("~/Downloads/prereq/scripts/IIR/iir_book.json").expanduser()
    output = []

    with open(path, "r") as f:
        data = json.load(f)
    for obj in data:
        section_number = obj["section_number"]
        section_name = obj["section_name"]
        section_text = obj["text"]
        source_url = obj["source_url"]
        print(section_number)
        print(section_name)
        print(source_url)

        response = call_antrophic_api(section_name, section_text)
        output.append({
            "section_number": section_number,
            "section_name": section_name,
            "model": MODEL,
            "extracted_concepts": response
        })

        print()

    with open("output.json", "w") as f:
        json.dump(output, f, indent=2)
# }}}

def evaluate(predicted, ground_truth):
# {{{
    pred_set = set(predicted)
    gt_set = set(ground_truth)
    
    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) else 0.0)
    
    return precision, recall, f1
# }}}

def load_IIR_annotations():
# {{{
    # iir-{section_number}.csv
    IIR_folder = Path("~/Downloads/prereq/datasets/IIR-dataset/annotation").expanduser()
    output_file = Path("~/Downloads/prereq/scripts/IIR/output.json").expanduser()
    pattern = r"iir-(\d+(?:\.\d+)*)\.csv"
    files = []
    
    for file in IIR_folder.iterdir():
        if file.is_file():
            reg = re.match(pattern, file.name)

            if reg:
                if reg.group(1) in ["1", "1.1", "1.2", "1.3", "1.4"]:
                    files.append(str(file))

    files.sort(key=lambda f: tuple(int(x) for x in re.match(pattern, Path(f).name).group(1).split(".")))
    # print(files)
    
    #  "section_number", "section_name", "model", "extracted_concepts"
    with open(output_file, 'r') as o:
        data = json.load(o)

    for obj, file in zip(data, files):
        # predictions
        concepts = [c.lower() for c in obj["extracted_concepts"].split(", ")]
        # ground truth
        concepts_IIR = []
        
        with open(file, 'r') as f:
            reader = csv.reader(f)
            next(reader)

            for row in reader:
                synonyms = ast.literal_eval(row[0])
                concepts_IIR.extend(s.strip().lower() for s in synonyms)
        
        # print(obj["section_number"])
        # print("predicted:", concepts)
        # print("ground truth:", concepts_IIR)
        prec, rec, f1 = evaluate(concepts, concepts_IIR)
        print(f"{obj['section_number']}: p={prec:.3f} r={rec:.3f} f1={f1:.3f}")
        print()
# }}}

MODEL = "claude-opus-4-6"
print("MODEL = ", MODEL)

load_dotenv()
load_and_experiment_ds()
load_IIR_annotations()

