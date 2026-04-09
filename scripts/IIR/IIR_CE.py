from dotenv import load_dotenv
from anthropic import Anthropic
from pathlib import Path
from datetime import datetime
import json
import csv
import re
import ast

def call_antrophic_api(section_name: str, section_text: str):
# {{{
    client = Anthropic()
    # Zero-shotting:
    # TEMPLATE = 'Instruction: From the provided text give me a list of ALL of the concepts FROM THE TEXT that are relevant to the domain of "information retrieval" in the context of "' + section_name + '". Formatting rules: Respond only with a comma AND 1 space SEPARATED list of the relevant concepts. Do not include any additional information or formatting to your response. Input text: ' + section_text

    # TEMPLATE = 'Instruction: From the provided text give me a list of ALL of the concepts FROM THE TEXT that are relevant to the domain of "information retrieval" in the context of "' + section_name + '". Formatting rules: Respond only with a comma AND 1 space SEPARATED list of the relevant concepts. If the concept itself contains any commas, write out the concept without the commas. Do not include any additional information or formatting to your response. Input text: ' + section_text

    # TEMPLATE = 'Instruction: From the provided text give me a list of ALL of the CONCEPTS that are IN THE PROVIDED TEXT that are relevant to the domain of "Information Retrieval (IR)" in the sub-context of "' + section_name + '". Formatting rules: Respond only with a comma AND 1 space SEPARATED list of the relevant concepts. If the concept itself contains any commas, write out the concept without the commas. Do not include any additional information or formatting to your response. Input text: ' + section_text

    # One-shotting
    # TEMPLATE = '<instructions> From the provided text give me a list of ALL OF THE CONCEPTS that are IN THE PROVIDED TEXT that are relevant to the domain of "Information Retrieval (IR)" in the sub-context of "' + section_name + '".</instructions> <example> For example, consider this sentence: "Tokenization is the task of chopping it up into pieces, called tokens, perhaps at the same time throwing away certain characters, such as punctuation". In this example, Tokenization and tokens are domain concepts, but characters and punctuation are not. </example> <instructions> Formatting rules: Respond only with a comma AND 1 space SEPARATED list of the relevant concepts. If the concept itself contains any commas, write out the concept without the commas. Do not include any additional information or formatting to your response. </instructions> <input> ' + section_text + '</input>'
    TEMPLATE = '<instructions> From the provided text give me a list of ALL OF THE CONCEPTS that are IN THE PROVIDED TEXT that are relevant to the domain of "Information Retrieval (IR)" in the sub-context of "' + section_name + '".</instructions> <example> For example, consider this sentence: "Tokenization is the task of chopping it up into pieces, called tokens, perhaps at the same time throwing away certain characters, such as punctuation". In this example, Tokenization and tokens are domain concepts, but characters and punctuation are not. </example> <instructions> Formatting rules: Respond only with a comma AND 1 space SEPARATED list of the relevant concepts. If the concept itself contains any commas, write out the concept without the commas. Do not include any additional information or formatting to your response. </instructions> <input> ' + section_text + '</input>'
    # TEMPLATE = '<instructions> From the provided text give me a list of ALL of the CONCEPTS that are IN THE PROVIDED TEXT that are relevant to the domain of "Information Retrieval (IR)" in the sub-context of "' + section_name + '". Also don\'t be stingy with the list of CONCEPTS. </instructions> <example> For example, consider this sentence: "Tokenization is the task of chopping it up into pieces, called tokens, perhaps at the same time throwing away certain characters, such as punctuation". In this example, Tokenization and tokens are domain concepts, but characters and punctuation are not. </example> <instructions> Formatting rules: Respond only with a comma AND 1 space SEPARATED list of the relevant concepts. If the concept itself contains any commas, write out the concept without the commas. Do not include any additional information or formatting to your response. </instructions> <input> ' + section_text + '</input>'

    message = client.messages.create(
        # max_tokens=1024,
        # max_tokens=2048,
        max_tokens=4096,
        system="You are a highly specialized expert in Information Retrieval (IR) and very skilful at extracting concepts from texts in the aforementioned domain.",
        messages=[{"role": "user", "content": TEMPLATE}],
        model=MODEL,
    )

    print("API CALL SUCCESSFUL !!!")
    print()
    # print(message.content)
    return message.content[0].text

    # response = client.messages.count_tokens(
    # model="claude-haiku-4-5-20251001",
    # messages=[
    #     {"role": "user", "content": "Hello, Claude. Explain attention in 3 sentences."}
    # ],
    # )
    # print(response.input_tokens)
# }}}

def load_and_predict_IIR():
# {{{
    # section_number, section_name, section_text, source_url
    # path = "/home/dust/Downloads/prereq/scripts/IIR/iir_book.json"
    predictions = []

    with open(IIR_path, "r") as f:
        data = json.load(f)
    for obj in data:
        section_number = obj["section_number"]
        section_name = obj["section_name"]
        section_text = obj["section_text"]
        # source_url = obj["source_url"]

        if section_number in SECTIONS or SECTIONS == [0]:
            print(section_number)
            print(section_name)
            # print(source_url)

            response = call_antrophic_api(section_name, section_text)
            predictions.append({
                "section_number": section_number,
                "section_name": section_name,
                "model": MODEL,
                "extracted_concepts": response
            })

        write_predictions_json(predictions)
# }}}

def write_predictions_json(predictions):
# {{{
    with open(MODEL + '_' + TYPE + '_' + TIMESTAMP + '_' + "predictions.json", "w") as f:
        json.dump(predictions, f, indent=2)
# }}}

def load_and_evaluate_IIR_annotations():
# {{{
    # iir-{section_number}.csv
    IIR_folder = Path("~/Downloads/prereq/datasets/IIR-dataset/annotation").expanduser()
    output_filename = MODEL + '_' + TYPE + '_' + TIMESTAMP + '_' + "predictions.json"
    output_file = Path("~/Downloads/prereq/scripts/IIR/" + str(output_filename)).expanduser()
    pattern = r"iir-(\d+(?:\.\d+)*)\.csv"
    files = []
    
    for file in IIR_folder.iterdir():
        if file.is_file():
            reg = re.match(pattern, file.name)

            if reg:
                if reg.group(1) in SECTIONS or SECTIONS == [0]:
                    files.append(str(file))

    files.sort(key=lambda f: tuple(int(x) for x in re.match(pattern, Path(f).name).group(1).split(".")))
    # print(files)
    
    #  "section_number", "section_name", "model", "extracted_concepts"
    with open(output_file, 'r') as o:
        data = json.load(o)

    evaluation = []
    prec_total = 0.0
    rec_total = 0.0
    f1_total = 0.0
    count = 0

    for obj, file in zip(data, files):
        predictions = obj["extracted_concepts"].split(", ")
        predictions = normalize_words(predictions)
        ground_truth = []
        
        with open(file, 'r') as f:
            reader = csv.reader(f)
            next(reader)

            for row in reader:
                if row[1] and row[2] and row[3]:
                    if '1' in [row[1][0], row[2][0], row[3][0]]:
                        concept = ast.literal_eval(row[0])[0]
                        ground_truth.append(concept)
                        ground_truth = normalize_words(ground_truth)
        
        predictions = sorted(predictions)
        ground_truth = sorted(ground_truth)

        # print(obj["section_number"])
        print("predictions:", predictions)
        # print("predictions type: ", type(predictions))

        print("ground truth:", ground_truth)
        # print("ground truth type:", type(ground_truth))

        prec, rec, f1 = evaluate(predictions, ground_truth)
        count += 1
        prec_total += prec
        rec_total += rec
        f1_total += f1

        print(f"{obj['section_number']}: p={prec:.3f} r={rec:.3f} f1={f1:.3f}")
        print()

        evaluation.append({
                'section_number': obj['section_number'],
                'precision': round(prec, 3),
                'recall': round(rec, 3),
                'f1': round(f1, 3)
            })

    evaluation.append({
            'section_number': "total",
            'precision': round(prec_total / count, 3),
            'recall': round(rec_total / count, 3),
            'f1': round(f1_total / count, 3)
        })
    write_evaluation_json(evaluation)
# }}}

def normalize_words(words: list[str]) -> list[str]:
# {{{
    result: list[str] = []

    for word in words:
        cleaned = re.sub(r"[^\w\s]", " ", word)
        # cleaned = re.sub(r"\s+", " ", cleaned)
        # cleaned = cleaned.strip().lower()
        cleaned = cleaned.lower()
        result.append(cleaned)

    return result
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

def write_evaluation_json(evaluation):
# {{{
    with open(MODEL + '_' + TYPE + '_' +  TIMESTAMP + '_' + "evaluation.json", "w") as f:
        json.dump(evaluation, f, indent=2)
# }}}

IIR_path = Path("~/Downloads/prereq/scripts/IIR/iir_sections.json").expanduser()

# MODEL = "claude-opus-4-6"
# MODEL = "claude-sonnet-4-6"
MODEL = "claude-haiku-4-5"

TYPE = "DEFAULT"
TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# TIMESTAMP = "2026-04-10_01-56-58" # For testing
# SECTIONS = ['1']
# SECTIONS = ['1', '1.1', '1.2', '1.3', '1.4']
SECTIONS = [0] # All sections

print("TIMESTAMP = ", TIMESTAMP)
print("MODEL = ", MODEL)
print("TYPE = ", TYPE)

load_dotenv()
load_and_predict_IIR() # Makes API calls
load_and_evaluate_IIR_annotations()

