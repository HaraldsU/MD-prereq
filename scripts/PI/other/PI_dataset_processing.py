from pathlib import Path
import csv
import json

"""
It is assumed that if concept_A, concept_B has a is_PR: 1 then concept_A is a PR
of concept_B unlike the reverse notation in the given datasets.
"""

# Each line of the file is in a format of 'A\tB', representing that B is a prerequisite of A. 
def process_course_dataset():
# {{{
    save_path = Path('~/Downloads/prereq/datasets/Course/').expanduser()
    cs_pos_file = Path('~/Downloads/prereq/datasets/Course/CS_edges.csv').expanduser()
    cs_neg_file = Path('~/Downloads/prereq/datasets/Course/CS_edges_neg.csv').expanduser()
    math_pos_file = Path('~/Downloads/prereq/datasets/Course/MATH_edges.csv').expanduser()
    math_neg_file = Path('~/Downloads/prereq/datasets/Course/MATH_edges_neg.csv').expanduser()

    cs_obj = []
    add_to_course_obj(cs_pos_file, cs_obj, True)    
    print(len(cs_obj))
    add_to_course_obj(cs_neg_file, cs_obj, False)    
    print(len(cs_obj))
    write_json(save_path, 'CS_edges_full.json', cs_obj)

    math_obj = []
    add_to_course_obj(math_pos_file, math_obj, True)    
    print(len(math_obj))
    add_to_course_obj(math_neg_file, math_obj, False)    
    print(len(math_obj))
    write_json(save_path, 'MATH_edges_full.json', math_obj)
# }}}

def add_to_course_obj(file: Path, obj, is_pr: bool, delim = '\t'):
# {{{
    with open (file, 'r') as f:
        reader = csv.reader(f, delimiter = delim)

        for row in reader:
            obj.append({
                'concept_A': row[1],
                'concept_B': row[0],
                'is_PR': 1 if is_pr == True else 0
            })
# }}}

# The second concept (2nd column) is a prerequisite of the first concept (1st column).
def process_alcpl_dataset():
# {{{
    save_path = Path('~/Downloads/prereq/datasets/AL-CPL/').expanduser()

    data_mining_pos_file = Path('~/Downloads/prereq/datasets/AL-CPL/data_mining_pos.csv').expanduser()
    data_mining_all_file = Path('~/Downloads/prereq/datasets/AL-CPL/data_mining_all.csv').expanduser()

    geometry_pos_file = Path('~/Downloads/prereq/datasets/AL-CPL/geometry_pos.csv').expanduser()
    geometry_all_file = Path('~/Downloads/prereq/datasets/AL-CPL/geometry_all.csv').expanduser()
    
    physics_pos_file = Path('~/Downloads/prereq/datasets/AL-CPL/physics_pos.csv').expanduser()
    physics_all_file = Path('~/Downloads/prereq/datasets/AL-CPL/physics_all.csv').expanduser()

    precalculus_pos_file = Path('~/Downloads/prereq/datasets/AL-CPL/precalculus_pos.csv').expanduser()
    precalculus_all_file = Path('~/Downloads/prereq/datasets/AL-CPL/precalculus_all.csv').expanduser()
    
    # Data mining
    dmp_obj = set()
    get_alcpl_set(data_mining_pos_file, dmp_obj)

    dma_obj = set()
    get_alcpl_set(data_mining_all_file, dma_obj)

    dmn_obj = dma_obj - dmp_obj
    dm_obj = []

    add_to_alcpl_obj(dm_obj, dmp_obj, True)
    add_to_alcpl_obj(dm_obj, dmn_obj, False)
    write_json(save_path, 'data_mining_full.json', dm_obj)

    # Geometry
    gp_obj = set()
    get_alcpl_set(geometry_pos_file, gp_obj)
    ga_obj = set()
    get_alcpl_set(geometry_all_file, ga_obj)
    gn_obj = ga_obj - gp_obj
    g_obj = []
    add_to_alcpl_obj(g_obj, gp_obj, True)
    add_to_alcpl_obj(g_obj, gn_obj, False)
    write_json(save_path, 'geometry_full.json', g_obj)

    # Physics
    pp_obj = set()
    get_alcpl_set(physics_pos_file, pp_obj)

    pa_obj = set()
    get_alcpl_set(physics_all_file, pa_obj)

    pn_obj = pa_obj - pp_obj
    p_obj = []

    add_to_alcpl_obj(p_obj, pp_obj, True)
    add_to_alcpl_obj(p_obj, pn_obj, False)
    write_json(save_path, 'physics_full.json', p_obj)

    # Precalc
    pcp_obj = set()
    get_alcpl_set(precalculus_pos_file, pcp_obj)
    pca_obj = set()
    get_alcpl_set(precalculus_all_file, pca_obj)
    pcn_obj = pca_obj - pcp_obj
    pc_obj = []
    add_to_alcpl_obj(pc_obj, pcp_obj, True)
    add_to_alcpl_obj(pc_obj, pcn_obj, False)
    write_json(save_path, 'precalculus_full.json', pc_obj)
# }}}

def get_alcpl_set(file: Path, obj: set, delim = ','):
# {{{
    with open (file, 'r') as f:
        reader = csv.reader(f, delimiter = delim)

        for row in reader:
            obj.add((row[0], row[1]))
# }}}

def add_to_alcpl_obj(res, obj, is_pr: bool):
# {{{
    for a, b in obj: 
        res.append({
            'concept_A': b,
            'concept_B': a,
            'is_PR': 1 if is_pr == True else 0
        })
# }}}

# Each line "<Concept_A>,<Concept_B>" represents that B is a prerequisite of A.
def process_ucd_dataset():
# {{{
    save_path = Path('~/Downloads/prereq/datasets/UCD/').expanduser()
    ucd_pos_file = Path('~/Downloads/prereq/datasets/UCD/ucd_pos.csv').expanduser()
    ucd_all_file = Path('~/Downloads/prereq/datasets/UCD/ucd_all.csv').expanduser()

    ucdp_obj = set()
    get_alcpl_set(ucd_pos_file, ucdp_obj)
    print(len(ucdp_obj))

    ucda_obj = set()
    get_alcpl_set(ucd_all_file, ucda_obj, '\t')
    print(len(ucda_obj))

    ucdn_obj = ucda_obj - ucdp_obj
    print(len(ucdn_obj))

    ucd_obj = []
    add_to_alcpl_obj(ucd_obj, ucdp_obj, True)
    add_to_alcpl_obj(ucd_obj, ucdn_obj, False)
    write_json(save_path, 'ucd_full.json', ucd_obj)
# }}}

def write_json(save_path: Path, file_name: str, obj):
# {{{
    with open(str(save_path) + '/' + file_name, 'w') as f:
        json.dump(obj, f, indent = 2)
# }}}

# process_course_dataset()
process_alcpl_dataset()
# process_ucd_dataset()

