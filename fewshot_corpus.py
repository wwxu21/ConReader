import json
import argparse
import os
import shutil
import random
import re
random.seed(33)
def sample_train_CCE(args):
    train_f = os.path.join(args.indir,"train" + ".json")
    all_data = json.load(open(train_f, encoding="utf-8"))
    capacity = {}
    for i_d, data_one in enumerate(all_data['data'][0]['paragraphs'][0]['qas']):
        question_text = data_one['question']
        question_text = question_text.replace("of this contract ", "").replace(" that should be reviewed by a lawyer", "")
        label = re.findall(
            r'(?<=Highlight the parts \(if any\) related to ").+(?=".)',
            question_text)[0].strip()
        if label not in capacity:
            capacity[label] = 0

    label_count = len(capacity)
    example_count = len(all_data['data'])

    sample_index = list(range(example_count))
    random.shuffle(sample_index)
    if args.fewshot > 1:
        fewshot_number = int(args.fewshot)
    else:
        fewshot_number = int(args.fewshot * example_count)
    fewshot_data = {}
    fewshot_data['version'] = all_data['version'] + "fewshot"
    fewshot_data['data'] = []
    for i_c,i in enumerate(sample_index[:fewshot_number]):
        examples = all_data['data'][i]
        qas = examples['paragraphs'][0]['qas']
        for qa_one in qas:
            question_text = qa_one['question']
            question_text = question_text.replace("of this contract ", "").replace(
                " that should be reviewed by a lawyer", "")
            label = re.findall(
                r'(?<=Highlight the parts \(if any\) related to ").+(?=".)',
                question_text)[0].strip()
            capacity[label] = capacity[label] + len(qa_one['answers'])

        fewshot_data['data'].append(examples)
    print('total size of fewshot examples are {}'.format(len(fewshot_data['data'])))
    print("capacity", capacity)
    return fewshot_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--fewshot",
        default=0.1,
        type=float,
        required=False,
        help="how many fewshot examples select for each category",
    )
    args = parser.parse_args()

    args.dataset = '3full'
    task = 'cce'

    args.indir = os.path.join( "Data", args.dataset)
    args.outdir = os.path.join( "Data", args.dataset + "-few" + str(args.fewshot))

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
        # directly copy test set
        shutil.copy2(os.path.join(args.indir,"test" + ".json"), os.path.join(args.outdir, "test" + ".json"))

    fewshot_train = sample_train_CCE(args)

    # with open(os.path.join(args.outdir, "train" + ".json"), 'w') as writer:
    #     json.dump(fewshot_train, writer, ensure_ascii=False, sort_keys=True, indent=2)

