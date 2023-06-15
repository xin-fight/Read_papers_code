import torch
import torch.nn as nn
from opt_einsum import contract
from long_seq import process_long_input
from losses import ATLoss


class DocREModel(nn.Module):
    def __init__(self, config, model, emb_size=768, block_size=64, num_labels=-1):
        # block_size用于group bilinear，定义一个group向量的尾数
        super().__init__()
        self.config = config
        self.model = model
        self.hidden_size = config.hidden_size  # AutoConfig.from_pretrained  768
        self.loss_fnt = ATLoss()

        self.head_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.tail_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.bilinear = nn.Linear(emb_size * block_size, config.num_labels)

        self.emb_size = emb_size
        self.block_size = block_size
        self.num_labels = num_labels

    def encode(self, input_ids, attention_mask):
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)
        return sequence_output, attention

    def get_hrt(self, sequence_output, attention, entity_pos, hts):
        """
        * Localized Context Pooling 中的计算公式 - 通过乘以对es和eo的实体级注意力来定位对它们都重要的局部上下文 *
        :param sequence_output: (batch_size, sequence_length, hidden_size) 模型最后一层输出的隐藏状态序列
        :param attention: (batch_size, num_heads, sequence_length, sequence_length) 取出最后一层注意力
        :param entity_pos: list (bs, 该bs中有的实体个数， 该实体的提及出现的位置)
        :param hts: list (bs, 该bs包含了实体和除了本身之前的 所有实体对)
        :return: tss - (所有文档实体对数, hidden_size)，表明该实体对认为重要的上下文嵌入
                 hss, rss - (所有文档实体对数, hidden_size), 头实体和尾实体对应的向量集合
        """
        """输入序列的开头和结尾添加一些特殊标记(起始标记、结束标记、填充标记等)
        但entity_pos是添加特殊标记之前的，所以需要加入了偏移量"""
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        # n=batch_size; h=num_heads; c=sequence_length
        n, h, _, c = attention.size()
        hss, tss, rss = [], [], []
        # len(entity_pos) 为bs, i为其中的一篇文档
        for i in range(len(entity_pos)):
            entity_embs, entity_atts = [], []
            for e in entity_pos[i]:
                if len(e) > 1:  # 该实体有多个提及
                    """e_emb(hidden_size), e_att(num_heads, sequence_length)实体的嵌入和实体对于其他单词的注意力"""
                    e_emb, e_att = [], []
                    for start, end in e:
                        if start + offset < c:
                            # In case the entity mention is truncated due to limited max seq length.
                            # 如果由于最大seq长度有限，实体名称被截断。
                            """使用 实体前的* 来代表实体"""
                            e_emb.append(sequence_output[i, start + offset])
                            e_att.append(attention[i, :, start + offset])
                    if len(e_emb) > 0:
                        """logsumexp pooling，一个最大池的平滑版本，利用多个提及来 得到实体嵌入hei"""
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                        """计算该实体的注意力权重，将提及的注意力权重进行平均"""
                        e_att = torch.stack(e_att, dim=0).mean(0)
                    else:  # 该实体的提及不在有效长度c内或者没有提及，则默认为0向量
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    start, end = e[0]
                    if start + offset < c:
                        e_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                entity_embs.append(e_emb)
                entity_atts.append(e_att)

            # 包含该文档中的所有实体
            entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]
            entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]

            """
            获取对应头实体的在entity_embs(实体数, hidden_size)对应的向量, 
            hs,ts - (实体对数, hidden_size)
            """
            # ht_i：该bs包含了实体和除了本身之前的 所有实体对
            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])

            """
            获取对应头实体的在entity_atts(实体对数,  num_heads, sequence_length)对应的向量
            h_att, t_att - [实体对数,  num_heads, sequence_length]
            """
            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])

            """Localized Context Pooling 中的计算公式 - 通过乘以对es和eo的实体级注意力来定位对它们都重要的局部上下文
            (h_att * t_att).mean(1): q^(s,o)
            归一化ht_att / (ht_att.sum(1, keepdim=True) + 1e-5): a^(s,o)
            sequence_output[i]: contextual embeddings，对应公式中的H
            最后的结果rs: c^(s,o) - (实体对数, hidden_size)，表明该实体对认为重要的上下文嵌入
            """
            # 将多头进行平均[实体对数,  num_heads, sequence_length] => [实体对数, sequence_length]
            ht_att = (h_att * t_att).mean(1)
            # 归一化，使它们的总和等于 1
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)
            # hs,ts - 头实体和尾实体对应的向量集合(实体对数, hidden_size),
            # ht_att - 头实体和尾实体对应的注意力集合（已经将多头平均）(实体对数, sequence_length)
            # sequence_output (batch_size, sequence_length, hidden_size) 模型最后一层输出的隐藏状态序列
            # sequence_output[i] (sequence_length, hidden_size)  ht_att(实体对数, sequence_length) => rs (实体对数, hidden_size)
            rs = contract("ld,rl->rd", sequence_output[i], ht_att)

            hss.append(hs)
            tss.append(ts)
            rss.append(rs)
        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)
        rss = torch.cat(rss, dim=0)
        return hss, rss, tss

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                entity_pos=None,
                hts=None,
                instance_mask=None,
                ):

        # input_ids attention_mask (bs, bs_max_len)
        # sequence_output (batch_size, sequence_length, hidden_size) 模型最后一层输出的隐藏状态序列
        # attention (batch_size, num_heads, sequence_length, sequence_length) 取出最后一层注意力
        sequence_output, attention = self.encode(input_ids, attention_mask)
        """此处的rs就是论文中的the localized context embedding  公式中c^(s,o)"""
        # ts - (该bs中所有文档实体对数, hidden_size)，表明该实体对认为重要的上下文嵌入
        # hs, rs - (该bs中所有文档实体对数, hidden_size), 头实体和尾实体对应的向量集合
        hs, rs, ts = self.get_hrt(sequence_output, attention, entity_pos, hts)

        """公式6，7，得到加入localized context embedding的实体嵌入"""
        hs = torch.tanh(self.head_extractor(torch.cat([hs, rs], dim=1)))
        ts = torch.tanh(self.tail_extractor(torch.cat([ts, rs], dim=1)))
        """group bilinear"""
        # hs,ts  (该bs中所有文档实体对数, hidden_size) => b1,b2 (该bs中所有文档实体对数, hidden_size//block_size, block_size)
        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)
        # b1.unsqueeze(3) (所有文档实体对数, hidden_size//block_size, block_size, 1)
        # b2.unsqueeze(2) (所有文档实体对数, hidden_size//block_size, 1, block_size)
        # (b1.unsqueeze(3) * b2.unsqueeze(2)) - (所有文档实体对数, hidden_size//block_size, block_size, block_size)
        # bl - (所有文档实体对数, hidden_size*block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        # logits (所有文档实体对数, num_labels - 关系的个数97)
        logits = self.bilinear(bl)

        # self.num_labels - Max number of labels in prediction
        output = (self.loss_fnt.get_label(logits, num_labels=self.num_labels),)
        # labels 所有实体对之间的关系
        if labels is not None:
            labels = [torch.tensor(label) for label in labels]
            # (所有文档实体对数, num_labels - 关系的个数97)
            labels = torch.cat(labels, dim=0).to(logits)
            """论文中的adaptive-thresholding loss"""
            loss = self.loss_fnt(logits.float(), labels.float())
            output = (loss.to(sequence_output),) + output
        # (loss, 模型预测的多个label (所有文档实体对数, num_labels 关系的个数) )
        #        若该实体对某关系的概率大于为Na关系 且 关系的概率>=关系概率Topk中的最小值，则该关系对应位置为1，
        #        若实体对的 满足 概率大于为Na关系 且 关系的概率>=关系概率Topk中的最小值 的关系数为0时，Na为1
        return output
