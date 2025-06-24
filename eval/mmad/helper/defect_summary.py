import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def caculate_accuracy_mmad(answers_json_path, normal_flag='good', show_overkill_miss=True):
    if os.path.exists(answers_json_path):
        with open(answers_json_path, "r") as file:
            content = file.read().strip()
            if content.startswith('['):
                all_answers_json = json.loads(content)
            else:
                all_answers_json = [json.loads(line) for line in content.splitlines() if line.strip()]

    index_tuples = set()
    type_list = set()

    for answer in all_answers_json:
        dataset_name = answer['image'].split('/')[0]
        defect_name = answer['image'].split('/')[3]
        question_type = answer['question_type']
        if question_type in ["Object Structure", "Object Details"]:
            question_type = "Object Analysis"
        index_tuples.add((dataset_name, defect_name))
        type_list.add(question_type)

    index_tuples = sorted(list(index_tuples))
    type_list = sorted(list(type_list))

    question_stats = {}
    detection_stats = {}

    for ds, defect in index_tuples:
        if ds not in question_stats:
            question_stats[ds] = {}
            detection_stats[ds] = {}
        question_stats[ds][defect] = {}
        detection_stats[ds][defect] = {
            'normal': {'total': 0, 'correct': 0},
            'abnormal': {'total': 0, 'correct': 0}
        }
        for qt in type_list:
            question_stats[ds][defect][qt] = {'total': 0, 'correct': 0, 'answers': {}, 'correct_answers': {}}

    for answer in all_answers_json:
        dataset_name = answer['image'].split('/')[0]
        defect_name = answer['image'].split('/')[3]
        question_type = answer['question_type']
        if question_type in ["Object Structure", "Object Details"]:
            question_type = "Object Analysis"

        gpt_answer = answer['gpt_answer'].strip().split('.')[0].strip().upper()
        correct_answer = answer['correct_answer'].strip().split('.')[0].strip().upper()

        if gpt_answer not in ['A', 'B', 'C', 'D', 'E'] or correct_answer not in ['A', 'B', 'C', 'D', 'E']:
            continue

        stat = question_stats[dataset_name][defect_name][question_type]
        stat['total'] += 1
        if gpt_answer == correct_answer:
            stat['correct'] += 1

        stat['answers'][gpt_answer] = stat['answers'].get(gpt_answer, 0) + 1
        stat['correct_answers'][correct_answer] = stat['correct_answers'].get(correct_answer, 0) + 1

        if question_type == 'Anomaly Detection':
            label = 'normal' if normal_flag in answer['image'] else 'abnormal'
            detection_stats[dataset_name][defect_name][label]['total'] += 1
            if gpt_answer == correct_answer:
                detection_stats[dataset_name][defect_name][label]['correct'] += 1

    accuracy_rows = []
    for ds, defect in index_tuples:
        row = {'dataset': ds, 'defect_name': defect}
        for qt in ["Anomaly Detection", "Defect Analysis", "Defect Classification", "Defect Description",
                   "Defect Localization", "Object Analysis", "Object Classification"]:
            if qt not in question_stats[ds][defect]:
                row[qt] = 0.0
                continue
            total = question_stats[ds][defect][qt]['total']
            correct = question_stats[ds][defect][qt]['correct']
            accuracy = correct / total * 100 if total != 0 else 0.0
            row[qt] = round(accuracy, 3)

        n_stat = detection_stats[ds][defect]['normal']
        a_stat = detection_stats[ds][defect]['abnormal']
        n_acc = n_stat['correct'] / n_stat['total'] if n_stat['total'] else 0.0
        a_acc = a_stat['correct'] / a_stat['total'] if a_stat['total'] else 0.0
        row['Anomaly Detection'] = round((n_acc + a_acc) / 2 * 100, 3)

        if show_overkill_miss:
            row['Overkill'] = round((1 - n_acc) * 100, 3)
            row['Miss'] = round((1 - a_acc) * 100, 3)

        accuracy_cols = [col for col in row if col not in ['dataset', 'defect_name']]
        row['Average'] = round(sum(row[col] for col in accuracy_cols if isinstance(row[col], float)) / len(accuracy_cols), 3)
        accuracy_rows.append(row)

    accuracy_df = pd.DataFrame(accuracy_rows)
    avg_row = accuracy_df.drop(columns=['dataset', 'defect_name']).mean(numeric_only=True).round(3)
    avg_row['dataset'] = 'Average'
    avg_row['defect_name'] = ''
    accuracy_df = pd.concat([accuracy_df, pd.DataFrame([avg_row])], ignore_index=True)

    accuracy_path = answers_json_path.replace('.jsonl', '_accuracy_defect.csv').replace('.json', '_accuracy_defect.csv')
    accuracy_df.to_csv(accuracy_path, index=False)
    return accuracy_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--answers_json_path', type=str, required=True)
    parser.add_argument('--normal_flag', type=str, default='good')
    parser.add_argument('--show_overkill_miss', type=bool, default=False)
    args = parser.parse_args()

    caculate_accuracy_mmad(args.answers_json_path, args.normal_flag, args.show_overkill_miss)
