from tqdm import tqdm
import ujson as json

docred_rel2id = json.load(open('meta/rel2id.json', 'r'))
cdr_rel2id = {'1:NR:2': 0, '1:CID:2': 1}
gda_rel2id = {'1:NR:2': 0, '1:GDA:2': 1}


def chunks(l, n):
    res = []
    for i in range(0, len(l), n):
        assert len(l[i:i + n]) == n
        res += [l[i:i + n]]
    return res


def read_docred(file_in, tokenizer, max_seq_length=1024):
    i_line = 0
    pos_samples = 0
    neg_samples = 0
    features = []
    if file_in == "":
        return None
    with open(file_in, "r") as fh:
        data = json.load(fh)

    """对一篇文档
    为了重新编号，sents是文档全部句子拼接，sent_map是对token重新编号，让mention开始的字符编号指向第一个*
    sample['sents'] = [['Zest', 'Airways', ',', 'Inc.', 'operated', 'as', 'AirAsia', ...], []]
    'Zest Airways Inc.' 为实体的一个提及
    sents ['*', 'Z', '##est', 'Airways', ',', 'Inc', '.', '*', 'operated', 'as', '*', 'Air', '##As', '##ia', ...]
    sent_map [{0: 0, 1: 3, 2: 4, 3: 5, 4: 8, 5: 9, ...}, {第2个句子修改后的token位置}]
             [{第0个token起始位置， 第1个实体的起始位置}, ]
    """
    for sample in tqdm(data, desc="Example"):  # 对每一个文档{ 'vertexSet':[], 'labels':[],'title': ,'sents':[] }
        sents = []
        # 记录每个句子中 各个token进过分词和引入*后的的起始位置 [{0: 0(第0个token起始位置), ...}, {第2个句子修改后的token位置}]
        sent_map = []

        entities = sample['vertexSet']  # [[{提及1},{}, {}], [] ]
        entity_start, entity_end = [], []  # 标明每个提及位置
        for entity in entities:
            for mention in entity:
                sent_id = mention["sent_id"]
                pos = mention["pos"]
                entity_start.append((sent_id, pos[0],))  # [(sent_id, pos), ()]
                entity_end.append((sent_id, pos[1] - 1,))
        for i_s, sent in enumerate(sample['sents']):  # 对每个文档中每个句子 sent [[句子1]， [句子2]]
            new_map = {}
            for i_t, token in enumerate(sent):  # 对每个句子中的token - 句子i中的某个token
                # 对第i_s个文档中的第i_t个token
                # ['Z', '##est']
                tokens_wordpiece = tokenizer.tokenize(token)  # 使用 tokenize() 函数对文本进行 tokenization之后，返回的分词的 token 词
                """
                论文中给定一个文档d，我们通过在提到的开始和结束时插入一个特殊的符号“*”来标记实体被提及的位置。
                我们将提及开始的“*”作为提到嵌入。
                * mention *
                """
                if (i_s, i_t) in entity_start:
                    tokens_wordpiece = ["*"] + tokens_wordpiece
                if (i_s, i_t) in entity_end:
                    tokens_wordpiece = tokens_wordpiece + ["*"]
                new_map[i_t] = len(sents)
                sents.extend(tokens_wordpiece)  # 一个文档句子放一起
            new_map[i_t + 1] = len(sents)
            sent_map.append(new_map)

        """记录该文档中某两个实体直接的关系 {(3, 4): [{'relation':, 'evidence':}, {'relation':, 'evidence':}], ():[]}"""
        train_triple = {}  # {(h, t): [{'relation': r, 'evidence': evidence}, {}]}
        if "labels" in sample:
            for label in sample['labels']:
                # 对于该文档中的某一个label
                evidence = label['evidence']
                r = int(docred_rel2id[label['r']])
                if (label['h'], label['t']) not in train_triple:
                    train_triple[(label['h'], label['t'])] = [
                        {'relation': r, 'evidence': evidence}]
                else:
                    train_triple[(label['h'], label['t'])].append(
                        {'relation': r, 'evidence': evidence})

        """记录每个句子中 各个token进过分词和引入*后的 实体最新的位置 [[(0, 8) - 第一个实体更新后位置区间[0,8), ()...], []]"""
        entity_pos = []
        for e in entities:  # [[mention1,2,3], [mention1,2]]
            entity_pos.append([])
            for m in e:  # 对于每个实体中的提及 [{mention1},{2},{3}]
                start = sent_map[m["sent_id"]][m["pos"][0]]  # 更新现在在填充*后的位置，mention开始的字符编号指向第一个*
                end = sent_map[m["sent_id"]][m["pos"][1]]
                entity_pos[-1].append((start, end,))

        """论文中
        pos_samples 存在于实体之间的关系
        neg_samples 实体之间不存在的关系
        """
        """relations记录每个实体对的包含的关系 [[0,0,1...] - 实体对关系, []]  当实体之间没有关系时，[1,0,0,...]
           hts                             [[0, 2]     - 对应实体对, []]"""
        relations, hts = [], []
        for h, t in train_triple.keys():
            relation = [0] * len(docred_rel2id)  # 关系的个数
            for mention in train_triple[h, t]:  # [{'relation': r, 'evidence': evidence}, {}]
                relation[mention["relation"]] = 1
                evidence = mention["evidence"]
            relations.append(relation)
            hts.append([h, t])
            # 实体对之间存在关系
            pos_samples += 1

        for h in range(len(entities)):
            for t in range(len(entities)):
                if h != t and [h, t] not in hts:
                    relation = [1] + [0] * (len(docred_rel2id) - 1)
                    relations.append(relation)
                    hts.append([h, t])
                    neg_samples += 1

        assert len(relations) == len(entities) * (len(entities) - 1)  # relations中是实体与除了本身之外的实体之间 正类负类关系

        sents = sents[:max_seq_length - 2]  # 预留模型起始和结束标记
        # 将标记序列（tokens）转换为对应的标识符（IDs）  [101, 115, 163, ...]
        input_ids = tokenizer.convert_tokens_to_ids(sents)
        # 用于构建输入序列（input sequence）并添加特殊标记
        """
        输入序列的开头和结尾添加一些特殊标记(起始标记、结束标记、填充标记等)，以便模型能够正确理解和处理这些序列。
        此时加入模型的特殊标记，但是由于entity_pos并没有修改，所以需要在之后的操作中加入偏移量offset
        """
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)  # 构造模型的输入

        i_line += 1
        feature = {'input_ids': input_ids,  # 一个文档的句子放一起，tokenization，实体加入*后，转为模型对应的输入
                   'entity_pos': entity_pos,  # 各个token进过分词和引入*后的 实体最新的位置 实体的位置[(start - 实体头指向第一个*, end,)]
                   'labels': relations,  # 所有实体 之间的关系
                   'hts': hts,  # 包含了实体和除了本身之前的 所有实体
                   'title': sample['title'],
                   }
        features.append(feature)

    print("# of documents {}.".format(i_line))
    print("# of positive examples {}.".format(pos_samples))
    print("# of negative examples {}.".format(neg_samples))
    return features


def read_cdr(file_in, tokenizer, max_seq_length=1024):
    pmids = set()
    features = []
    maxlen = 0
    with open(file_in, 'r') as infile:
        lines = infile.readlines()
        for i_l, line in enumerate(tqdm(lines)):
            line = line.rstrip().split('\t')
            pmid = line[0]

            if pmid not in pmids:
                pmids.add(pmid)
                text = line[1]
                prs = chunks(line[2:], 17)

                ent2idx = {}
                train_triples = {}

                entity_pos = set()
                for p in prs:
                    es = list(map(int, p[8].split(':')))
                    ed = list(map(int, p[9].split(':')))
                    tpy = p[7]
                    for start, end in zip(es, ed):
                        entity_pos.add((start, end, tpy))

                    es = list(map(int, p[14].split(':')))
                    ed = list(map(int, p[15].split(':')))
                    tpy = p[13]
                    for start, end in zip(es, ed):
                        entity_pos.add((start, end, tpy))

                sents = [t.split(' ') for t in text.split('|')]
                new_sents = []
                sent_map = {}
                i_t = 0
                for sent in sents:
                    for token in sent:
                        tokens_wordpiece = tokenizer.tokenize(token)
                        for start, end, tpy in list(entity_pos):
                            if i_t == start:
                                tokens_wordpiece = ["*"] + tokens_wordpiece
                            if i_t + 1 == end:
                                tokens_wordpiece = tokens_wordpiece + ["*"]
                        sent_map[i_t] = len(new_sents)
                        new_sents.extend(tokens_wordpiece)
                        i_t += 1
                    sent_map[i_t] = len(new_sents)
                sents = new_sents

                entity_pos = []

                for p in prs:
                    if p[0] == "not_include":
                        continue
                    if p[1] == "L2R":
                        h_id, t_id = p[5], p[11]
                        h_start, t_start = p[8], p[14]
                        h_end, t_end = p[9], p[15]
                    else:
                        t_id, h_id = p[5], p[11]
                        t_start, h_start = p[8], p[14]
                        t_end, h_end = p[9], p[15]
                    h_start = map(int, h_start.split(':'))
                    h_end = map(int, h_end.split(':'))
                    t_start = map(int, t_start.split(':'))
                    t_end = map(int, t_end.split(':'))
                    h_start = [sent_map[idx] for idx in h_start]
                    h_end = [sent_map[idx] for idx in h_end]
                    t_start = [sent_map[idx] for idx in t_start]
                    t_end = [sent_map[idx] for idx in t_end]
                    if h_id not in ent2idx:
                        ent2idx[h_id] = len(ent2idx)
                        entity_pos.append(list(zip(h_start, h_end)))
                    if t_id not in ent2idx:
                        ent2idx[t_id] = len(ent2idx)
                        entity_pos.append(list(zip(t_start, t_end)))
                    h_id, t_id = ent2idx[h_id], ent2idx[t_id]

                    r = cdr_rel2id[p[0]]
                    if (h_id, t_id) not in train_triples:
                        train_triples[(h_id, t_id)] = [{'relation': r}]
                    else:
                        train_triples[(h_id, t_id)].append({'relation': r})

                relations, hts = [], []
                for h, t in train_triples.keys():
                    relation = [0] * len(cdr_rel2id)
                    for mention in train_triples[h, t]:
                        relation[mention["relation"]] = 1
                    relations.append(relation)
                    hts.append([h, t])

            maxlen = max(maxlen, len(sents))
            sents = sents[:max_seq_length - 2]
            input_ids = tokenizer.convert_tokens_to_ids(sents)
            input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

            if len(hts) > 0:
                feature = {'input_ids': input_ids,
                           'entity_pos': entity_pos,
                           'labels': relations,
                           'hts': hts,
                           'title': pmid,
                           }
                features.append(feature)
    print("Number of documents: {}.".format(len(features)))
    print("Max document length: {}.".format(maxlen))
    return features


def read_gda(file_in, tokenizer, max_seq_length=1024):
    pmids = set()
    features = []
    maxlen = 0
    with open(file_in, 'r') as infile:
        lines = infile.readlines()
        for i_l, line in enumerate(tqdm(lines)):
            line = line.rstrip().split('\t')
            pmid = line[0]

            if pmid not in pmids:
                pmids.add(pmid)
                text = line[1]
                prs = chunks(line[2:], 17)

                ent2idx = {}
                train_triples = {}

                entity_pos = set()
                for p in prs:
                    es = list(map(int, p[8].split(':')))
                    ed = list(map(int, p[9].split(':')))
                    tpy = p[7]
                    for start, end in zip(es, ed):
                        entity_pos.add((start, end, tpy))

                    es = list(map(int, p[14].split(':')))
                    ed = list(map(int, p[15].split(':')))
                    tpy = p[13]
                    for start, end in zip(es, ed):
                        entity_pos.add((start, end, tpy))

                sents = [t.split(' ') for t in text.split('|')]
                new_sents = []
                sent_map = {}
                i_t = 0
                for sent in sents:
                    for token in sent:
                        tokens_wordpiece = tokenizer.tokenize(token)
                        for start, end, tpy in list(entity_pos):
                            if i_t == start:
                                tokens_wordpiece = ["*"] + tokens_wordpiece
                            if i_t + 1 == end:
                                tokens_wordpiece = tokens_wordpiece + ["*"]
                        sent_map[i_t] = len(new_sents)
                        new_sents.extend(tokens_wordpiece)
                        i_t += 1
                    sent_map[i_t] = len(new_sents)
                sents = new_sents

                entity_pos = []

                for p in prs:
                    if p[0] == "not_include":
                        continue
                    if p[1] == "L2R":
                        h_id, t_id = p[5], p[11]
                        h_start, t_start = p[8], p[14]
                        h_end, t_end = p[9], p[15]
                    else:
                        t_id, h_id = p[5], p[11]
                        t_start, h_start = p[8], p[14]
                        t_end, h_end = p[9], p[15]
                    h_start = map(int, h_start.split(':'))
                    h_end = map(int, h_end.split(':'))
                    t_start = map(int, t_start.split(':'))
                    t_end = map(int, t_end.split(':'))
                    h_start = [sent_map[idx] for idx in h_start]
                    h_end = [sent_map[idx] for idx in h_end]
                    t_start = [sent_map[idx] for idx in t_start]
                    t_end = [sent_map[idx] for idx in t_end]
                    if h_id not in ent2idx:
                        ent2idx[h_id] = len(ent2idx)
                        entity_pos.append(list(zip(h_start, h_end)))
                    if t_id not in ent2idx:
                        ent2idx[t_id] = len(ent2idx)
                        entity_pos.append(list(zip(t_start, t_end)))
                    h_id, t_id = ent2idx[h_id], ent2idx[t_id]

                    r = gda_rel2id[p[0]]
                    if (h_id, t_id) not in train_triples:
                        train_triples[(h_id, t_id)] = [{'relation': r}]
                    else:
                        train_triples[(h_id, t_id)].append({'relation': r})

                relations, hts = [], []
                for h, t in train_triples.keys():
                    relation = [0] * len(gda_rel2id)
                    for mention in train_triples[h, t]:
                        relation[mention["relation"]] = 1
                    relations.append(relation)
                    hts.append([h, t])

            maxlen = max(maxlen, len(sents))
            sents = sents[:max_seq_length - 2]
            input_ids = tokenizer.convert_tokens_to_ids(sents)
            input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

            if len(hts) > 0:
                feature = {'input_ids': input_ids,
                           'entity_pos': entity_pos,
                           'labels': relations,
                           'hts': hts,
                           'title': pmid,
                           }
                features.append(feature)
    print("Number of documents: {}.".format(len(features)))
    print("Max document length: {}.".format(maxlen))
    return features
