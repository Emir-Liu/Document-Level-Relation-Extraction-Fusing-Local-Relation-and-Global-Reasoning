import numpy as np
import ujson as json


def read_file(file_path,tokenizer,rel2id):
    print('file path {}'.format(file_path))
    with open(file_path,'r') as f:
        data = json.load(f)
    #print(data)
    feature = []

    for sample in data:
        entities = sample['vertexSet']
        mentions_start,mentions_end = [],[]
        for entity in entities:
            for mention in entity:
                mentions_start.append((mention['sent_id'],mention['pos'][0]))
                mentions_end.append((mention['sent_id'],mention['pos'][1]-1))
        #print('start mention {}'.format(mentions_start))
        #print('end mention {}'.format(mentions_end))

        sents = []
        sents_map = []
        for idx_sent,sent in enumerate(sample['sents']):
            sent_map = []# 每个句子中，token和token_piece开始位置 map
            # 其中需要注意的是，如果在实体前加的*,将*作为开始的位置。看图。最后一位没有用
            sent_map.append(len(sents))
            for idx_token,token in enumerate(sent):
                token_wordpiece = tokenizer.tokenize(token)
                if (idx_sent,idx_token) in mentions_start:
                    token_wordpiece = ['*'] + token_wordpiece
                if (idx_sent,idx_token) in mentions_end:
                    token_wordpiece = token_wordpiece + ['*']
                sents.extend(token_wordpiece)
                sent_map.append(len(sents))
            sents_map.append(sent_map)
        #print('sents {}'.format(sents))
        #print('sents_map {}'.format(sents_map))

        entity_pos = [] # 这里开始和结尾位置需要注意，结尾位置是*后一个位置
        for entity in entities:
            entity_pos.append([])
            for mention in entity:
                start = sents_map[mention['sent_id']][mention['pos'][0]]
                end = sents_map[mention['sent_id']][mention['pos'][1]]
                entity_pos[-1].append((start,end))
        #print('entity pos {}'.format(entity_pos))

        triple = {}
        if "labels" in sample:
            for label in sample['labels']:
                if (label['h'],label['t']) not in triple:
                    triple[(label['h'], label['t'])] = \
                        [{'r':rel2id[label['r']], 'e':label['evidence']}]
                else:
                    triple[(label['h'],label['t'])].append({'r':int(rel2id[label['r']]),'e':label['evidence']})
        #print('triple {}'.format(triple))

        labels = []
        for h in range(len(entities)):
            labels.append([])
            for t in range(len(entities)):
                tmp_relation = [0] * len(rel2id)
                if (h,t) in triple:
                    for relation in triple[h,t]:
                        tmp_relation[relation['r']] = 1
                else:
                    tmp_relation[0] = 1
                labels[-1].append(tmp_relation)
        #print('labels {}'.format(np.array(labels).shape))
        #print('2 4 {}'.format(labels[2][4]))

        input_ids = tokenizer.convert_tokens_to_ids(sents)
        #print('tmp input ids {}'.format(input_ids))
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)
        #print('input ids {}'.format(input_ids))
        sample_data = {'title':sample['title'],
            'input_ids':input_ids,
            'labels':labels,
            'entity_pos':entity_pos}
        feature.append(sample_data)
        #print('sample_data {}'.format(sample_data))
        #print('# of entity {}'.format(len(entities)))
    print('# of documents {}'.format(len(data)))
    return feature







