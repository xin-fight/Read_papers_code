import argparse
import os

import numpy as np
import torch
from apex import amp
import ujson as json
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from model import DocREModel
from utils import set_seed, collate_fn
from prepro import read_docred
from evaluation import to_official, official_evaluate
import wandb


def train(args, model, train_features, dev_features, test_features):
    def finetune(features, optimizer, num_epoch, num_steps):
        """
        [ {'input_ids': input_ids,  # 一个文档的句子放一起，转为模型对应的输入
           'entity_pos': entity_pos,  # 实体的位置[(start - 实体头指向第一个*, end,)]
           'labels': relations,  # 所有实体之间的关系
           'hts': hts,  # 包含了实体和除了本身之前的所有实体
           'title': sample['title']
           }, {} ]
        """
        best_score = -1
        # 将一批数据填充到统一大小，且进行mask标记 - collate_fn: (input_ids, input_mask, labels, entity_pos, hts)
        train_dataloader = DataLoader(features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn,
                                      drop_last=True)
        train_iterator = range(int(num_epoch))
        # 如果显存不足，我们可以通过gradient_accumulation_steps梯度累计来解决
        # 梯度累积是一种训练技巧，可以在每个批次上累积一定数量的梯度，然后再进行一次参数更新。这对于在内存受限的情况下增加批次大小或减少显存使用很有用
        # total_steps表示整个训练过程中的总步数，它将在训练过程中用于控制训练的终止条件或相关计算的调度
        """将数据集的总批次数[len(train_dataloader)]乘以总轮数[num_epoch]，可以得到在完整训练过程中 总共需要处理的批次数。
        然后将这个总批次数除以梯度累积的步数(将多批的梯度累计然后再进行更新)，即可得到总步数"""
        total_steps = int(len(train_dataloader) * num_epoch // args.gradient_accumulation_steps)
        # warmup_steps存在，lr先慢慢增加，超过warmup_steps时，lr再慢慢减小
        warmup_steps = int(total_steps * args.warmup_ratio)
        # 函数会根据预定义的规则生成一个线性学习率调度器，该调度器会在预热阶段逐渐增加学习率，然后在训练的剩余步数中保持稳定的学习率。这个预热阶段可以帮助模型更好地适应训练数据。
        # 这段代码的目的是将创建的学习率调度器与优化器关联起来，以便在每个训练步骤中自动更新学习率。这样，在训练过程中，学习率将按照预设的规则进行调整
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=total_steps)
        print("Total steps: {}".format(total_steps))
        print("Warmup steps: {}".format(warmup_steps))
        for epoch in train_iterator:  # epoch
            model.zero_grad()
            for step, batch in enumerate(train_dataloader):
                model.train()
                inputs = {'input_ids': batch[0].to(args.device),
                          'attention_mask': batch[1].to(args.device),
                          'labels': batch[2],  # 所有实体对之间的关系
                          'entity_pos': batch[3],
                          'hts': batch[4],
                          }
                outputs = model(**inputs)
                loss = outputs[0] / args.gradient_accumulation_steps
                # amp.scale_loss 是混合精度训练中的一个关键步骤，它用于缩放损失值以适应低精度计算。
                # 通过使用 amp.scale_loss，可以将损失值按比例缩放，以确保在进行梯度计算时不会因为低精度计算而丢失精度。
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    # 使用 amp.scale_loss 对 loss 进行缩放后，调用 scaled_loss.backward() 进行反向传播，计算梯度。
                    # 这样，梯度会根据缩放后的损失值进行计算 - * 此时在累计梯度
                    scaled_loss.backward()
                if step % args.gradient_accumulation_steps == 0:
                    if args.max_grad_norm > 0:
                        # 这是对梯度进行剪裁的操作，用于控制梯度的大小。
                        # amp.master_params(optimizer) 返回使用自动混合精度训练的模型的参数，
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    # 执行梯度更新的步骤
                    optimizer.step()
                    # 使用了学习率调度器，会在每个梯度累积步骤后调用更新学习率。
                    scheduler.step()
                    model.zero_grad()
                    # 更新了多少次参数 - 已经执行的梯度累积步骤
                    num_steps += 1
                # wandb.log({"loss": loss.item()}, step=num_steps)

                # 如果当前步骤是训练数据加载器的倒数第二个步骤，即即将完成一个训练循环，那么就满足进行验证评估的条件
                if (step + 1) == len(train_dataloader) - 1 or (
                        args.evaluation_steps > 0 and num_steps % args.evaluation_steps == 0 and step % args.gradient_accumulation_steps == 0):
                    dev_score, dev_output = evaluate(args, model, dev_features, tag="dev")
                    # wandb.log(dev_output, step=num_steps)
                    print(dev_output)
                    if dev_score > best_score:
                        best_score = dev_score
                        pred = report(args, model, test_features)
                        with open("result.json", "w") as fh:
                            json.dump(pred, fh)
                        if args.save_path != "":
                            torch.save(model.state_dict(), args.save_path)
        return num_steps

    new_layer = ["extractor", "bilinear"]
    optimizer_grouped_parameters = [
        # 获取所有名字中不带有new_layer中字符串的层参数
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)], },
        # 名字中带有new_layer中的字符串的层参数
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)], "lr": 1e-4},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # 将模型和优化器初始化为支持混合精度训练 - 混合精度来训练模型，显存减少将近一半的情况下，训练速度也得到大幅度提升。
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
    num_steps = 0
    set_seed(args)
    model.zero_grad()
    finetune(train_features, optimizer, args.num_train_epochs, num_steps)


def evaluate(args, model, features, tag="dev"):
    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn,
                            drop_last=False)
    preds = []
    for batch in dataloader:
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  }

        with torch.no_grad():
            pred, *_ = model(**inputs)
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    # 预测的结果
    ans = to_official(preds, features)
    if len(ans) > 0:
        best_f1, _, best_f1_ign, _ = official_evaluate(ans, args.data_dir)
    output = {
        tag + "_F1": best_f1 * 100,
        tag + "_F1_ign": best_f1_ign * 100,
    }
    return best_f1, output


def report(args, model, features):
    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn,
                            drop_last=False)
    preds = []
    for batch in dataloader:
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  }

        with torch.no_grad():
            pred, *_ = model(**inputs)
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    preds = to_official(preds, features)
    return preds


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./dataset/docred", type=str)
    parser.add_argument("--transformer_type", default="bert", type=str)
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str)

    parser.add_argument("--train_file", default="train_annotated.json", type=str)
    parser.add_argument("--dev_file", default="dev.json", type=str)
    parser.add_argument("--test_file", default="test.json", type=str)
    parser.add_argument("--save_path", default="", type=str)
    parser.add_argument("--load_path", default="", type=str)

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=1024, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=8, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_labels", default=4, type=int,
                        help="Max number of labels in prediction.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.06, type=float,
                        help="Warm up ratio for Adam.")
    parser.add_argument("--num_train_epochs", default=30.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--evaluation_steps", default=-1, type=int,
                        help="Number of training steps between evaluations.")
    parser.add_argument("--seed", type=int, default=66,
                        help="random seed for initialization")
    parser.add_argument("--num_class", type=int, default=97,
                        help="Number of relation types in dataset.")

    args = parser.parse_args()
    # wandb.init(project="DocRED")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    """
    transformers的三个核心抽象类是Config, Tokenizer和Model，这些类根据模型种类的不同，派生出一系列的子类。
    transformers为这三个类都提供了自动类型，即AutoConfig, AutoTokenizer和AutoModel。
    """
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_class,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
    )

    read = read_docred

    train_file = os.path.join(args.data_dir, args.train_file)
    dev_file = os.path.join(args.data_dir, args.dev_file)
    test_file = os.path.join(args.data_dir, args.test_file)
    # max_seq_length 1024
    # of documents 3053. of positive（实体对之间存在关系） examples 35615. of negative examples 1163035.
    train_features = read(train_file, tokenizer, max_seq_length=args.max_seq_length)
    # of documents 998. of positive（实体对之间存在关系） examples 11470. of negative examples 384102.
    dev_features = read(dev_file, tokenizer, max_seq_length=args.max_seq_length)
    # of documents 1000. of positive（实体对之间存在关系） examples 0. of negative examples 392158.
    test_features = read(test_file, tokenizer, max_seq_length=args.max_seq_length)
    """
    [ {'input_ids': input_ids,  # 一个文档的句子放一起，转为模型对应的输入
       'entity_pos': entity_pos,  # 实体的位置[(start - 实体头指向第一个*, end,)]
       'labels': relations,  # 所有实体之间的关系
       'hts': hts,  # 包含了实体和除了本身之前的所有实体
       'title': sample['title']
       }, {} ]
    """

    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    # 获取分词器（tokenizer）中预定义的起始标记（[CLS]）的标识符（ID） 101
    config.cls_token_id = tokenizer.cls_token_id
    # 用于获取分词器（tokenizer）中预定义的分隔标记（[SEP]）的标识符（ID） 102
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = args.transformer_type

    set_seed(args)
    model = DocREModel(config, model, num_labels=args.num_labels)
    model.to(0)

    if args.load_path == "":  # Training
        train(args, model, train_features, dev_features, test_features)
    else:  # Testing
        model = amp.initialize(model, opt_level="O1", verbosity=0)
        model.load_state_dict(torch.load(args.load_path))
        dev_score, dev_output = evaluate(args, model, dev_features, tag="dev")
        print(dev_output)
        pred = report(args, model, test_features)
        with open("result.json", "w") as fh:
            json.dump(pred, fh)


if __name__ == "__main__":
    main()
