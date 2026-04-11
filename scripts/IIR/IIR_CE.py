from dotenv import load_dotenv
from anthropic import Anthropic
from pathlib import Path
from datetime import datetime
from nltk.stem import SnowballStemmer
from sentence_transformers import SentenceTransformer, util
import json
import csv
import re
import ast
import string
import simplemma
import torch

def call_antrophic_api(section_name: str, section_text: str) -> str:
# {{{
    client = Anthropic()
    # Zero-shotting:
    # TEMPLATE = 'Instruction: From the provided text give me a list of ALL of the CONCEPTS that are IN THE PROVIDED TEXT that are relevant to the domain of "Information Retrieval (IR)" in the sub-context of "' + section_name + '". Formatting rules: Respond only with a comma AND 1 space SEPARATED list of the relevant concepts. If the concept itself contains any commas, write out the concept without the commas. Do not include any additional information or formatting to your response. Input text: ' + section_text

    # One-shotting
    TEMPLATE = '<instructions> From the provided text give me a list of ALL OF THE CONCEPTS that are IN THE PROVIDED TEXT that are relevant to the domain of "Information Retrieval (IR)" in the sub-context of "' + section_name + '".</instructions> <example> For example, consider this sentence: "Tokenization is the task of chopping it up into pieces, called tokens, perhaps at the same time throwing away certain characters, such as punctuation". In this example, Tokenization and tokens are domain concepts, but characters and punctuation are not. </example> <instructions> Formatting rules: Respond only with a comma AND 1 space SEPARATED list of the relevant concepts. If the concept itself contains any commas, write out the concept without the commas. Do not include any additional information or formatting to your response. </instructions> <input> ' + section_text + '</input>'

    message = client.messages.create(
        max_tokens = int(MAX_TOKENS),
        system = 'You are a highly specialized expert in Information Retrieval (IR) and very skilful at extracting concepts from texts.',
        messages = [{'role': 'user', 'content': TEMPLATE}],
        model = MODEL,
    )

    print('API CALL SUCCESSFUL !!!')
    print()
    # print(message.content)
    return message.content[0].text

    # response = client.messages.count_tokens(
    # model='claude-haiku-4-5-20251001',
    # messages=[
    #     {'role': 'user', 'content': 'Hello, Claude. Explain attention in 3 sentences.'}
    # ],
    # )
    # print(response.input_tokens)
# }}}

def load_and_predict_IIR():
# {{{
    print('--- Getting predictions ---')

    # section_number, section_name, section_text, source_url
    with open(IIR_sections_json_path, 'r') as f:
        IIR_sections_obj = json.load(f)

    predictions = []
    IIR_gt_paths = get_IIR_ground_truth_paths()

    for section, gt_path in zip(IIR_sections_obj, IIR_gt_paths):
        section_number = section['section_number']
        
        if section_number in SECTIONS or SECTIONS == [0]:
            section_name = section['section_name']
            section_text = section['section_text']
            print(section_number, section_name)
            prediction = call_antrophic_api(section_name, section_text)
            prediction = sorted(normalize_words(prediction.split(', ')))
            ground_truth = []

            with open(gt_path, 'r') as f:
                reader = csv.reader(f)
                next(reader)

                for row in reader:
                    if row[1] and row[2] and row[3]:
                        if '1' in [row[1][0], row[2][0], row[3][0]]:
                            concept = ast.literal_eval(row[0])[0]
                            ground_truth.append(concept)

            ground_truth = sorted(normalize_words(ground_truth))
            # print('len(section_text) = ', len(section_text))

            predictions.append({
                'section_number': section_number,
                'section_name': section_name,
                'predicted_concepts (pc)': prediction,
                'pc_count': len(prediction),
                'pc_count_to_text_len_ratio (%)': round((len(prediction) / len(section_text)) * 100, 3),
                'ground_truth_concepts (gtc)': ground_truth,
                'gtc_count': len(ground_truth),
                'gtc_count_to_text_len_ratio (%)': round((len(ground_truth) / len(section_text)) * 100, 3),
                'section_text_count': len(section_text)
            })

    write_json(predictions, 'predictions')
# }}}

def write_json(predictions, type: str):
# {{{
    with open(PATH_PREFIX + type + '.json', 'w') as f:
        json.dump(predictions, f, indent=2)
# }}}

def evaluate_IIR():
# {{{
    print('--- Evaluations ---')
    predictions_filename = PATH_PREFIX + 'predictions.json'
    predictions_json_path = Path('~/Downloads/prereq/scripts/IIR/' + str(predictions_filename)).expanduser()

    # 'section_number', 'section_name', 'predicted_concepts (pc)', 'pc_count', 'pc_count_to_text_len_ratio (%)', 'ground_truth_concepts (gtc)', 'gtc_count', 'gtc_count_to_text_len_ratio (%)', 'section_text_count'
    with open(predictions_json_path, 'r') as o:
        predictions_obj = json.load(o)

    evaluation = []
    count = 0

    prec_total = 0.0
    rec_total = 0.0
    f1_total = 0.0

    prec_total_sent = 0.0
    rec_total_sent = 0.0
    f1_total_sent = 0.0

    prec_total_sim = 0.0
    rec_total_sim = 0.0
    f1_total_sim = 0.0

    for prediction in predictions_obj:
    # {{{
        section_number = prediction['section_number']

        if section_number in SECTIONS or SECTIONS == [0]:
            predictions = prediction['predicted_concepts (pc)']
            ground_truth = prediction['ground_truth_concepts (gtc)']

            # print('predictions:', predictions)
            # print('ground truth:', ground_truth)

            prec, rec, f1 = calc_statistical_metrics(predictions, ground_truth)

            print(f'{prediction['section_number']} {prediction['section_name']}: p={prec:.3f} r={rec:.3f} f1={f1:.3f}')
            prec_sent, rec_sent, f1_sent, prec_sim, rec_sim, f1_sim = calc_semantical_metrics(predictions, ground_truth)
            print()

            prec_total += prec
            rec_total += rec
            f1_total += f1

            prec_total_sent += prec_sent
            rec_total_sent += rec_sent
            f1_total_sent += f1_sent

            prec_total_sim += prec_sim
            rec_total_sim += rec_sim
            f1_total_sim += f1_sim

            count += 1

            evaluation.append({
                    'section_number': prediction['section_number'],

                    'precision': round(prec, 3),
                    'recall': round(rec, 3),
                    'f1': round(f1, 3),

                    'precision_sentence': round(prec_sent, 3),
                    'recall_sentence': round(rec_sent, 3),
                    'f1_sentence': round(f1_sent, 3),

                    'precision_sim': round(prec_sim, 3),
                    'recall_sim': round(rec_sim, 3),
                    'f1_sim': round(f1_sim, 3),

                    'pc_count': prediction['pc_count'],
                    'pc_count_to_text_len_ratio (%)': prediction['pc_count_to_text_len_ratio (%)'],
                    'gtc_count': prediction['gtc_count'],
                    'gtc_count_to_text_len_ratio (%)': prediction['gtc_count_to_text_len_ratio (%)'],
                    'section_text_count': prediction['section_text_count']
                })
    # }}}

    evaluation.append({
            'section_number': 'average',

            'precision': round(prec_total / count, 3),
            'recall': round(rec_total / count, 3),
            'f1': round(f1_total / count, 3),

            'precision_sentence': round(prec_total_sent / count, 3),
            'recall_sentence': round(rec_total_sent / count, 3),
            'f1_sentence': round(f1_total_sent / count, 3),

            'precision_sim': round(prec_total_sim / count, 3),
            'recall_sim': round(rec_total_sim / count, 3),
            'f1_sim': round(f1_total_sim / count, 3)
        })

    print(f'Average: prec_total={(prec_total / count):.3f} rec_total={(rec_total / count):.3f} f1_total:{(f1_total / count):.3f}')
    print(f'Average (sentence): prec_total_sent={(prec_total_sent / count):.3f} rec_total_sent={(rec_total_sent / count):.3f} f1_total_sent:{(f1_total_sent / count):.3f}')
    print(f'Average (sim): prec_total_sim={(prec_total_sim / count):.3f} rec_total_sim={(rec_total_sim / count):.3f} f1_total_sim:{(f1_total_sim / count):.3f}')
    write_json(evaluation, 'evaluation')
# }}}

def get_IIR_ground_truth_paths() -> list[str]:
# {{{
    # iir-{section_number}.csv
    pattern = r'iir-(\d+(?:\.\d+)*)\.csv'
    IIR_csvs = []
    
    for file in IIR_folder_path.iterdir():
        if file.is_file():
            reg = re.match(pattern, file.name)

            if reg:
                if reg.group(1) in SECTIONS or SECTIONS == [0]:
                    IIR_csvs.append(str(file))

    IIR_csvs.sort(key=lambda f: tuple(int(x) for x in re.match(pattern, Path(f).name).group(1).split('.')))
    return IIR_csvs
# }}}

def normalize_words(words: list[str]) -> list[str]:
# {{{
    result: list[str] = []

    for word in words:
        cleaned = ' '.join(re.sub(f'[{re.escape(string.punctuation)}]', ' ', word).lower().split())
        # lemmatized = ' '.join(simplemma.lemmatize(w, lang='en') for w in cleaned.split())
        # result.append(lemmatized)

        stemmed = ' '.join(stemmer.stem(w) for w in cleaned.split())
        result.append(stemmed)

    return result
# }}}

def calc_statistical_metrics(predicted: list[str], ground_truth: list[str]) -> tuple[float, float, float]:
# {{{
    pred_set = set(predicted)
    gt_set = set(ground_truth)
    
    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall) if (precision + recall) else 0.0)
    
    return precision, recall, f1
# }}}

def calc_semantical_metrics(predicted: list[str], ground_truth: list[str]):
# {{{
    predicted_concat  = [' '.join(predicted)]
    ground_truth_concat = [' '.join(ground_truth)]
    # print('predicted = ', predicted_concat)
    # print('ground_truth = ', ground_truth_concat)

    emb_pred_sent = SENTENCE_MODEL.encode(predicted_concat, normalize_embeddings = True, compression_ratio = 0.5)
    emb_gt_sent = SENTENCE_MODEL.encode(ground_truth_concat, normalize_embeddings = True, compression_ratio = 0.5)

    sim_sent = SENTENCE_MODEL.similarity(emb_pred_sent, emb_gt_sent)
    prec_sent = sim_sent.max(axis=1).values.mean().item()
    rec_sent = sim_sent.max(axis=0).values.mean().item()
    f1_sent = 2 * prec_sent * rec_sent / (prec_sent + rec_sent)
    print(f'Sentence similarity: p={prec_sent:.3f}  r={rec_sent:.3f}  f1={f1_sent:.3f}')

    emb_pred_sim = SENTENCE_MODEL.encode(predicted, normalize_embeddings = True, compression_ratio = 0.5)
    emb_gt_sim = SENTENCE_MODEL.encode(ground_truth, normalize_embeddings = True, compression_ratio = 0.5)

    # print('emb_pred.shape = ', emb_pred.shape)
    # print('emb_gt.shape = ', emb_gt.shape)
    # print('emb_pred = ', emb_pred)
    # print('emb_gt = ', emb_gt)
    
    sim_sim = SENTENCE_MODEL.similarity(emb_pred_sim, emb_gt_sim)
    prec_sim = sim_sim.max(axis=1).values.mean().item()
    rec_sim = sim_sim.max(axis=0).values.mean().item()
    f1_sim = 2 * prec_sim * rec_sim / (prec_sim + rec_sim)
    print(f'Greedy list similarity: p={prec_sim:.3f}  r={rec_sim:.3f}  f1={f1_sim:.3f}')

    return prec_sent, rec_sent, f1_sent, prec_sim, rec_sim, f1_sim
# }}}

# --- Paths ---
IIR_sections_json_path = Path('~/Downloads/prereq/scripts/IIR/iir_sections.json').expanduser()
IIR_folder_path = Path('~/Downloads/prereq/datasets/IIR-dataset/annotation').expanduser()

# --- Models ---
# MODEL = 'claude-opus-4-6'
# MODEL = 'claude-sonnet-4-6'
MODEL = 'claude-haiku-4-5'
MAX_TOKENS = '1024'
# MAX_TOKENS = 2048
# MAX_TOKENS = 4096

# --- Methods ---
METHOD = 'ONE-SHOT'

# --- Timestamps ---
TIMESTAMP = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
# TIMESTAMP = '2026-04-11_04-31-20' # For testing

# --- Sections ---
SECTIONS = ['1']
# SECTIONS = ['1', '1.1', '1.2', '1.3', '1.4']
# SECTIONS = [0] # All sections

# --- Other ---
# NORMALIZATION = 'LEMMATIZATION'
NORMALIZATION = 'STEMMED'
PATH_PREFIX = MODEL + '_' + MAX_TOKENS + '_' + METHOD + '_' + NORMALIZATION + '_' + TIMESTAMP + '_'
SENTENCE_MODEL = SentenceTransformer(
    'infgrad/Jasper-Token-Compression-600M',
    processor_kwargs={
        'dtype': torch.bfloat16,
        'attn_implementation': 'sdpa',
        'trust_remote_code': True,
    },
    trust_remote_code=True,
    device='cuda',
)
stemmer = SnowballStemmer('english')

print('TIMESTAMP = ', TIMESTAMP)
print('MODEL = ', MODEL)
print('MAX_TOKENS = ', MAX_TOKENS)
print('METHOD = ', METHOD)
print('NORMALIZATION = ', NORMALIZATION)
print()

load_dotenv()
load_and_predict_IIR() # Makes API calls
evaluate_IIR()
# calc_semantical_metrics(['How do I tie a shoelace?'], ['The GDP of Japan in 1987'])
# calc_semantical_metrics(['7x^2 + 3x - 5 = 0'], ['The dog barked loudly at the mailman'])
# calc_semantical_metrics(['Hello'], ['Hello'])

