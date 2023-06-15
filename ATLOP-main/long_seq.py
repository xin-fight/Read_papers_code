import torch
import torch.nn.functional as F
import numpy as np


def process_long_input(model, input_ids, attention_mask, start_tokens, end_tokens):
    """
    将输入规整为模型要求的长度，长的分为两个句子（因为之前读取数据时， 设定max_seq_length=512），并将input_ids输入到模型中得到
        sequence_output (batch_size, sequence_length, hidden_size) 模型最后一层输出的隐藏状态序列
        attention (batch_size, num_heads, sequence_length, sequence_length) 取出最后一层注意力
    :param model: AutoModel.from_pretrained
    :param input_ids: 输入模型的标记化文本的索引列表，表示文本中每个词的编号。
    :param attention_mask:
    :param start_tokens: 对于bert [config.cls_token_id]
    :param end_tokens: 对于bert [config.sep_token_id]
    :return:sequence_output, attention
    """
    # input_ids, attention_mask (bs, bs_max_len)
    # Split the input to 2 overlapping chunks. Now BERT can encode inputs of which the length are up to 1024.
    # 将输入分割为2个重叠的块。现在BERT可以对长度高达1024的输入进行编码
    n, c = input_ids.size()  # bs bs_max_len
    start_tokens = torch.tensor(start_tokens).to(input_ids)
    end_tokens = torch.tensor(end_tokens).to(input_ids)
    len_start = start_tokens.size(0)  # 获取start_tokens表格的长度
    len_end = end_tokens.size(0)  # 获取end_tokens表格的长度
    if c <= 512:  # 该bs_max_len长度小于512
        """
        last_hidden_state, pooler_output, attentions = model(input_ids, attention_mask, output_attentions=True)
            input_ids: 输入模型的标记化文本的索引列表，表示文本中每个词的编号。
            attention_mask: 注意力掩码，用于指示哪些位置需要被注意，哪些位置可以被忽略。
            output_attentions表示是否要输出注意力权重，如果输出，需要指定config.output_attentions=True,
            
        -last_hidden_state：shape是(batch_size, sequence_length, hidden_size) 模型最后一层输出的隐藏状态序列
        -pooler_output: (batch_size, hidden_size) 序列的第一个标记（分类标记）的最后一层隐藏状态
        hidden_states: (optional, output_hidden_states=True) 是一个元组（tuple），其中包含了模型每层的隐藏状态。这个元组由多个 torch.FloatTensor 组成，每个对应 一个层级 的隐藏状态。
            元组中的第一个 FloatTensor 是嵌入层的输出（如果模型有嵌入层的话），其形状为 (batch_size, sequence_length, hidden_size)。这个嵌入层的输出表示了输入序列经过嵌入处理后的表示。
            元组中的后续 FloatTensor 对应每个层级的隐藏状态。对于每个层级，隐藏状态的形状也是 (batch_size, sequence_length, hidden_size)。这些隐藏状态表示了输入序列在每个层级经过处理后的特征表示。
        -attentions: (optional, output_attentions=True) 是一个元组（tuple），其中包含了模型每层的注意力权重。这个元组由多个 torch.FloatTensor 组成，每个对应模型的 一个层级。
            每个 FloatTensor 的形状是 (batch_size, num_heads, sequence_length, sequence_length)。其中，batch_size 是输入序列的批量大小，num_heads 是注意力头的数量，sequence_length 是输入序列的长度。
        cross_attentions：是一个元组（tuple），其中包含了解码器的交叉注意力层的注意力权重。这个元组由多个 torch.FloatTensor 组成，每个 FloatTensor 对应模型的一个层级。
            每个 FloatTensor 的形状是 (batch_size, num_heads, sequence_length, sequence_length)。其中，batch_size 是输入序列的批量大小，num_heads 是注意力头的数量，sequence_length 是输入序列的长度。
        past_key_values： 是一个元组（tuple），包含了预先计算的隐藏状态，用于加速顺序解码。
        """
        # model - AutoModel.from_pretrained
        output = model(
            input_ids=input_ids,  # bs bs_max_len
            attention_mask=attention_mask,
            output_attentions=True,
        )
        # (batch_size, sequence_length, hidden_size) 模型最后一层输出的隐藏状态序列
        sequence_output = output[0]
        # print(torch.tensor([item.cpu().detach().numpy() for item in output[-1]]).size())
        # print(torch.tensor([item.cpu().detach().numpy() for item in output[-1][-2]]).size())
        # output[-1]为attentions，output[-1][-1]取出最后一层注意力, 最后一层注意力维度为(batch_size, num_heads, sequence_length, sequence_length)
        attention = output[-1][-1]
    else:  # 该bs最大长度大于512，不论大小全部统一到512长度
        new_input_ids, new_attention_mask, num_seg = [], [], []  # num_seg - 1记录小于512的，2记录大于512被要分成两个句子
        seq_len = attention_mask.sum(1).cpu().numpy().astype(np.int32).tolist()  # 按行加，得到该批数据中每个数据的有效字符长度
        # 对batch中的一个文档，l_i为有效句子长度
        for i, l_i in enumerate(seq_len):
            if l_i <= 512:  # 文档有效部分小于512，直接截取到512
                new_input_ids.append(input_ids[i, :512])
                new_attention_mask.append(attention_mask[i, :512])
                num_seg.append(1)
            else:  # 大于512，要分成两个句子，但是还要保留特殊的开始和结束符
                input_ids1 = torch.cat([input_ids[i, :512 - len_end], end_tokens], dim=-1)  # 512
                input_ids2 = torch.cat([start_tokens, input_ids[i, (l_i - 512 + len_start): l_i]], dim=-1)  # 512
                attention_mask1 = attention_mask[i, :512]
                attention_mask2 = attention_mask[i, (l_i - 512): l_i]
                new_input_ids.extend([input_ids1, input_ids2])
                new_attention_mask.extend([attention_mask1, attention_mask2])
                num_seg.append(2)
        input_ids = torch.stack(new_input_ids, dim=0)
        attention_mask = torch.stack(new_attention_mask, dim=0)

        output = model(
            input_ids=input_ids,  # bs中最后得到的512片段数量 max_len=512
            attention_mask=attention_mask,
            output_attentions=True,
        )
        sequence_output = output[0]
        attention = output[-1][-1]
        i = 0
        new_output, new_attention = [], []
        for (n_s, l_i) in zip(num_seg, seq_len):
            if n_s == 1:  # 文档有效部分小于512，直接截取到512
                # [bs, 512, hidden_size] - [bs, bs_max_len, hidden_size]
                ## sequence_output[i] - [512, hidden_size]
                output = F.pad(sequence_output[i], (0, 0, 0, c - 512))
                # [batch_size, num_heads, 512, 512] - [batch_size, num_heads, bs_max_len, bs_max_len]
                att = F.pad(attention[i], (0, c - 512, 0, c - 512))
                new_output.append(output)
                new_attention.append(att)
            elif n_s == 2:  # 文档部分大于512，被分成两个句子，但是还要保留特殊的开始和结束符
                # 去除将句子分成两个句子时第一个句子的填充的结束符
                output1 = sequence_output[i][:512 - len_end]
                mask1 = attention_mask[i][:512 - len_end]
                att1 = attention[i][:, :512 - len_end, :512 - len_end]
                # [bs, 512, hidden_size] - [bs, bs_max_len, hidden_size]
                output1 = F.pad(output1, (0, 0, 0, c - 512 + len_end))
                # [bs, 512] - [bs, bs_max_len]
                mask1 = F.pad(mask1, (0, c - 512 + len_end))
                # [batch_size, num_heads, 512, 512] - [batch_size, num_heads, bs_max_len, bs_max_len]
                att1 = F.pad(att1, (0, c - 512 + len_end, 0, c - 512 + len_end))

                # 去除将句子分成两个句子时第二个句子的填充的起始符
                output2 = sequence_output[i + 1][len_start:]
                mask2 = attention_mask[i + 1][len_start:]
                att2 = attention[i + 1][:, len_start:, len_start:]
                """
                (0, 0, l_i - 512 + len_start, c - l_i) — (左边填充数， 右边填充数， 上边填充数， 下边填充数)
                对于第二维（句子长度），在将2，3维看作一个长方形（长度维作为长，hidden_size为宽）
                    pad主要是：长方形的上边填充l_i - 512 + len_start；下边填充c - l_i
                    原始：  |_____ l_i ____|___0___|
                    mask1: |___512___|_____0______|
                    mask2: |_0__|___512___|___0___|
                    相加：  |__1_|_2__|__1_|___0___|
                    output = (output1 + output2) / mask.unsqueeze(-1): 
                        output1 + output2会有重复被相加的区域，但经过除以mask1 + mask2后 则可得到所需要的结果
                """
                # [bs, 512, hidden_size] - [bs, bs_max_len, hidden_size]  li为有效长度
                output2 = F.pad(output2, (0, 0, l_i - 512 + len_start, c - l_i))
                # [bs, 512] - [bs, bs_max_len]
                mask2 = F.pad(mask2, (l_i - 512 + len_start, c - l_i))
                # [batch_size, num_heads, 512, 512] - [batch_size, num_heads, bs_max_len, bs_max_len]
                att2 = F.pad(att2, [l_i - 512 + len_start, c - l_i, l_i - 512 + len_start, c - l_i])
                mask = mask1 + mask2 + 1e-10  # 之后要被除，因此0要变为1e-10
                output = (output1 + output2) / mask.unsqueeze(-1)
                """
                之前的output2，mask2都是要将被分成两个句子合并时，免去重复区域的累加带来的影响
                !!!但是对于att只是简单的相加，然后归一化，有一下问题：
                    设：前512个单词和后512个单词之间重复单词A; 前512个单词和后512个单词中不在A中的分别为B，C
                    1. B中的单词和C中的单词之间的attention没法计算
                    2. A中的单词之间的attention在计算att1 + att2时被重复计算了
                """
                att = (att1 + att2)
                # 将 att 张量中的每个值归一化，使它们的总和等于 1
                att = att / (att.sum(-1, keepdim=True) + 1e-10)
                new_output.append(output)
                new_attention.append(att)
            i += n_s
        sequence_output = torch.stack(new_output, dim=0)
        attention = torch.stack(new_attention, dim=0)
    return sequence_output, attention
