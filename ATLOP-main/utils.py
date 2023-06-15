import torch
import random
import numpy as np


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0 and torch.cuda.is_available():
        # 设置所有可用的CUDA设备的随机种子（random seed）。
        torch.cuda.manual_seed_all(args.seed)


def collate_fn(batch):
    """ 一个数据
    [ {'input_ids': input_ids,  # 一个文档的句子放一起，转为模型对应的输入
       'entity_pos': entity_pos,  # 实体的位置[(start - 实体头指向第一个*, end,)]
       'labels': relations,  # 所有实体之间的关系
       'hts': hts,  # 包含了实体和除了本身之前的所有实体
       'title': sample['title']
       }, {} ]
    """
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]  # 填充到该batch最大长度
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]  # 标记填充部分
    labels = [f["labels"] for f in batch]
    entity_pos = [f["entity_pos"] for f in batch]
    hts = [f["hts"] for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    output = (input_ids, input_mask, labels, entity_pos, hts)
    return output
