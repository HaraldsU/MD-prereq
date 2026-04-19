import csv

course_path = "/home/dust/Downloads/prereq/datasets/Course/"
positive_edges_path = course_path + "CS_edges.csv"
negative_edges_path = course_path + "CS_edges_neg.csv"

def make_course_union_csv(pp, np, output):
# {{{
    data = []

    with open(pp, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            data.append(row + ['1'])

    with open(np, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            data.append(row + ['0'])

    with open(output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['concept_a', 'concept_b', 'label'])
        writer.writerows(data)
# }}}

make_course_union_csv(positive_edges_path, negative_edges_path, 'Course_union.csv')

