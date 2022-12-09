import json
import re

def read_json(addr):
    with open(addr) as reader:
        f = json.load(reader)
    return f

def preprocess(tsq_f, tsq_ref):
    for i, _ in enumerate(tsq_f['data']):
        answer1 = tsq_f['data'][i]['paragraphs'][0]['context']
        answer2 = tsq_ref['data'][i]['paragraphs'][0]['context']
        if answer1 != answer2:
            tsq_f['data'][i]['paragraphs'][0]['context'] = tsq_ref['data'][i]['paragraphs'][0]['context']
    for i_c, contract in enumerate(tsq_f['data']):
        definition = contract.pop('definition')
        data_one = contract['paragraphs'][0]
        data_one['definition'] = definition
    for i_c, contract in enumerate(tsq_f['data']):
        definition = contract['paragraphs'][0]['definition']
        context = contract['paragraphs'][0]['context']
        definition_list = []
        for key in definition:
            value = definition[key]
            if key != "" and key[0].isdigit():
                clean_key = re.sub("[0-9]+", "", key)
            else:
                clean_key = key
            clean_key = re.sub("shall", "", clean_key)
            clean_key = clean_key.strip()
            if len(clean_key) < 2 or clean_key not in context or (len(clean_key) < 3 and not clean_key[0].isupper()) or (
                        len(clean_key) >= 3 and not (
                            clean_key[0].isupper() or clean_key[1].isupper() or clean_key[2].isupper())) or value == '[***]':
                print(clean_key, ": cannot find any location in contract", i_c)
                continue
            try:
                new_definition = {}
                key_position = [m.start() for m in re.finditer(clean_key, context)]
                value_position = context.index(value)
                new_definition['key'] = clean_key
                new_definition['key_position'] = key_position
                new_definition['value'] = value
                new_definition['value_position'] = value_position
                definition_list.append(new_definition)
            except:
                print(clean_key, ": cannot find any location in contract", i_c)
        contract['paragraphs'][0]['definition'] = definition_list

    max_key = []
    max_ix = 0
    max_iy = 0
    for ix, x in enumerate(tsq_f['data']):
        for iy, y in enumerate(x['paragraphs'][0]['definition']):
            if len(y['key']) > 50:
                max_key.append((ix, iy, len(y['key'])))
    max_key = sorted(max_key, key=lambda x: x[1], reverse=True)
    for ix, x in enumerate(max_key):
        tsq_f['data'][x[0]]['paragraphs'][0]['definition'].pop(x[1])

    return tsq_f
if __name__ == "__main__":
    cuadv1_f = 'Data/CUADv1.json'
    tsq_f1_ref = 'Data/full/train.json'
    tsq_f2_ref = 'Data/full/test.json'
    tsq_f1 = 'Data/full/train.json'
    tsq_f2 = 'Data/full/test.json'
    tsq_f1 = read_json(tsq_f1)
    tsq_f2 = read_json(tsq_f2)
    tsq_f1_ref = read_json(tsq_f1_ref)
    tsq_f2_ref = read_json(tsq_f2_ref)
    # preprocess raymond data
    tsq_f1 = preprocess(tsq_f1, tsq_f1_ref)
    tsq_f2 = preprocess(tsq_f2, tsq_f2_ref)
    cuad_train_test = {'version': 'v2.0', 'data': tsq_f1['data']}
    with open('Data/3full/train.json', 'w') as writer:
        json.dump(cuad_train_test, writer)
    cuad_train_test = {'version': 'v2.0', 'data': tsq_f2['data']}
    with open('Data/3full/test.json', 'w') as writer:
        json.dump(cuad_train_test, writer)




