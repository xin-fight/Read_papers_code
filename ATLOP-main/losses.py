import torch
import torch.nn as nn
import torch.nn.functional as F


class ATLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        """
        计算adaptive-threshold loss
        :param logits: 模型预测(所有文档实体对数, num_labels - 关系的个数97)
        :param labels: 真实标签(所有文档实体对数, num_labels - 关系的个数97)
        :return:
        """
        """Adaptive Thresholding
        a threshold class TH 被认为是 第0个关系（Na）对应的概率
        这个自适应阈值是与实体相关的，实体不同，对应的阈值不同
        """
        # TH label (所有文档实体对数, num_labels - 关系的个数97)
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, 0] = 1.0
        # 真实标签, 将Na对应的位置设置为0
        labels[:, 0] = 0.0

        # 对所有实体对，Na关系 以及 该实体对存在的关系 为 1
        p_mask = labels + th_label
        # 对所有实体对，不属于该实体对的关系为 1
        n_mask = 1 - labels


        """
        第一部分：包括积极类和TH类。
        由于可能存在多个正类，总损失被计算为 所有正类上的类别交叉熵损失的和
        L1推动所有正类的对数都高于TH类。如果没有阳性标签，则不使用它
        """
        # Rank positive classes to TH
        # logit1:该实体对不存在的关系（除了Na）的概率为负无穷    (所有文档实体对数, 关系的个数97)
        logit1 = logits - (1 - p_mask) * 1e30
        # F.log_softmax(logit1, dim=-1): 该实体对不存在的关系（除了Na）的概率经过softmax后接近与0，再经过log是一个大数；对那些Na以及存在的关系经过softmax和log
        #       * 计算r属于Pt和TH类的log_softmax *
        # F.log_softmax(logit1, dim=-1) * labels: 那些Na以及存在的关系logsoftmax * 将Na对应的位置设置为0的真实标签。
        #       计算将上述结果在求和时去掉TH类的影响，这样的话就得到了论文中的L1
        loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(1)  # (所有文档实体对数, 关系的个数97)

        """
        第二部分L2涉及到负类和阈值类
        它是 一个类别交叉熵损失，TH类是真正的标签。它将负类的对数拉到低于TH类。
        """
        # Rank TH to negative classes
        # logit2:该实体对存在的关系（除了Na）的概率为负无穷   (所有文档实体对数, 关系的个数97)
        logit2 = logits - (1 - n_mask) * 1e30
        # F.log_softmax(logit2, dim=-1): 该实体对存在的关系的概率经过softmax后接近与0，再经过log是一个大数；对那些Na以及不存在的关系经过softmax和log
        #       * 计算r‘属于Nt和TH类的log_softmax *
        # F.log_softmax(logit2, dim=-1) * th_label: 只保留Na对应的log_softmax
        #       计算得到的上述结果为L2
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)

        # Sum two parts
        loss = loss1 + loss2
        loss = loss.mean()
        return loss

    def get_label(self, logits, num_labels=-1):
        """
        :param logits: (所有文档实体对数, num_labels - 关系的个数)
        :param num_labels: 关系个数
        :return: (所有文档实体对数, num_labels 关系的个数)
        """
        """Adaptive Thresholding，a threshold class TH 被认为是 第0个关系（Na）对应的概率"""
        # th_logit (所有文档实体对数, 1)  每个实体对关系为Na的概率
        th_logit = logits[:, 0].unsqueeze(1)
        # output (所有文档实体对数, num_labels 关系的个数)
        output = torch.zeros_like(logits).to(logits)
        # 对于某个实体对，对于某个关系，只有当该关系的概率大于为Na关系时，才会为True： (所有文档实体对数, num_labels - 关系的个数)
        mask = (logits > th_logit)
        if num_labels > 0:
            # num_labels: 需要获取的最大值的个数 k=4。 返回：(values, indices)
            # top_v - values (所有文档实体对数, 最多用于预测的标签个数)
            top_v, _ = torch.topk(logits, num_labels, dim=1)
            # 取出topk个中最小的那个
            top_v = top_v[:, -1]
            """对于某个实体对，对于某个关系，只有当该关系的概率大于为Na关系 且 关系的概率>=关系概率Topk中的最小时，才会为True"""
            mask = (logits >= top_v.unsqueeze(1)) & mask
        # output (所有文档实体对数, num_labels 关系的个数)
        #       只有对应实体对 中某关系的概率大于为Na关系 且 关系的概率>=关系概率Topk中的最小值 才设置为1
        output[mask] = 1.0
        # (output.sum(1) == 0.) 如果某实体对的 满足 概率大于为Na关系 且 关系的概率>=关系概率Topk中的最小值 的关系数为0时，Na为1
        output[:, 0] = (output.sum(1) == 0.).to(logits)
        return output
