import torch
import torch.nn as nn
from torch.nn import functional as F


def kd_loss(student_logits, teacher_logits, temperature=3.0):
    T = temperature
    return F.kl_div(
        F.log_softmax(student_logits / T, dim=-1),
        F.softmax(teacher_logits / T, dim=-1),
        reduction="batchmean"
    ) * (T * T)


def hidden_loss(student_h, teacher_h):
    return F.mse_loss(student_h, teacher_h)


def attention_loss(student_attn, teacher_attn):
    return F.mse_loss(student_attn, teacher_attn)


def compute_loss(student_out, teacher_out, target_ids, alpha=0.5, beta=0.3, gamma=0.2):
    s_logits, s_h, s_attn = student_out
    t_logits, t_h, t_attn = teacher_out
    ce = F.cross_entropy(s_logits.view(-1, s_logits.size(-1)), target_ids.view(-1), ignore_index=-100)
    kd = kd_loss(s_logits, t_logits)
    hd = hidden_loss(s_h, t_h)
    att = attention_loss(s_attn, t_attn)
    return ce + alpha * kd + beta * hd + gamma * att