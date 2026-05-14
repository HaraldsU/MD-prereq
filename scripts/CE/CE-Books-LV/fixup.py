import json
from pathlib import Path

def cleanup(in_path: Path, out_path: Path):
# {{{
    with open(in_path, 'r') as f:
        data = json.load(f)

    for obj in data:
        # if 'gtc' in obj:
            # obj['gtc'] = [
                # REWRITE_ITEMS.get(item, item).lower()
                # for item in obj['gtc']
                # if item not in REMOVE_ITEMS
            # ]
        if 'ground_truth_concepts (gtc)' in obj:
            obj['ground_truth_concepts (gtc)'] = [
                REWRITE_ITEMS.get(item, item).lower()
                for item in obj['ground_truth_concepts (gtc)']
                if item not in REMOVE_ITEMS
            ]

    with open(out_path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
# }}}

def deduplicate(in_path:Path, out_path:Path):
# {{{
    with open(in_path, 'r') as f:
        data = json.load(f)

    for obj in data:
        # if 'gtc' in obj:
            # obj['gtc'] = list(dict.fromkeys(obj['gtc']))
        if 'ground_truth_concepts (gtc)' in obj:
            obj['ground_truth_concepts (gtc)'] = list(dict.fromkeys(obj['ground_truth_concepts (gtc)']))

    with open(out_path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
# }}}

# in_path = Path('IIR-index.json')
# in_path = Path('google-gemini-3-flash-preview_1024_FEW-SHOT_FIVE-SHOT_RANDOM_TERMS_ENGLISH_DOMAIN-CONTEXT-YES_CONCEPT-DEFINITION-NO_SYSTEM-PROMPT-YES_-CONSENSUS_STEMMED_2026-04-30_23-55-44_predictions.json')
# in_path = Path('ds/chem_vsk_mg.json')
in_path = Path('outputs/p5_CHEM_KU7b/xiaomi-mimo-v2-flash_1024_FEW-SHOT_FIVE-SHOT_FIRST_GALVENĀS-FRĀZES_LATVIAN_DOMAIN-CONTEXT-YES_CONCEPT-DEFINITION-KEY_SYSTEM-PROMPT-YES_-CONSENSUS_STEMMED_2026-04-30_20-10-23_predictions.json')

# REMOVE_ITEMS = {'A', '1'}
# REWRITE_ITEMS = {
    # '-gram index': 'n-gram index',
    # 'B test': 'A/B test',
    # '0 loss': '1/0 loss',
# }

REMOVE_ITEMS = {'36,', '102,', '105,'}
REWRITE_ITEMS = {
    'membrānas, šūnu': 'šūnu membrānas',
}

cleanup(in_path, Path(in_path.stem + '_cleaned.json'))
deduplicate(Path(in_path.stem + '_cleaned.json'), Path(in_path.stem + '_deduped.json'))

