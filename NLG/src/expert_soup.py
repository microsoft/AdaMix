#  ------------------------------------------------------------------------------------------
#  Copyright (c). All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from torch import nn

import torch
from torch import Tensor
import torch.distributed as dist
from torch.nn import Module, ModuleList
import torch.nn.functional as F
import copy
import typing

import math

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class MixtureSoup(torch.nn.Module):

    def __init__(self, expert, num_local_experts=1):
        super(MixtureSoup, self).__init__()

        self.deepspeed_experts = torch.nn.ModuleList(
            [copy.deepcopy(expert) for i in range(num_local_experts)])
        self.num_local_experts = num_local_experts
        self.expert_score_weight = torch.nn.Parameter(torch.zeros(self.num_local_experts), requires_grad=False)

    def get_expert_by_idx(self, idx):
        return self.deepspeed_experts[idx]

    def expert_soup_forward(self, input):
        output = F.linear(input,
                          self.parameter_dict["weight"],
                          self.parameter_dict["bias"])
        return output

    def expert_soup(self):
        weight = F.softmax(self.expert_score_weight, dim=-1)
        self.parameter_dict = {"weight": 0, "bias": 0}
        for idx in range(self.num_local_experts):
            single_expert = self.deepspeed_experts[idx]
            for s_name, s_param in single_expert.named_parameters():
                if "weight" in s_name:
                    p_name = "weight"
                    self.parameter_dict[p_name] = self.parameter_dict[p_name] + (weight[idx] * s_param)
                else:
                    p_name = "bias"
                    self.parameter_dict[p_name] = self.parameter_dict[p_name] + (weight[idx] * s_param)


    def forward(self, *input: Tensor):
        expert_output = None
        if self.deepspeed_experts[0].training:
            expert_idx = torch.randint(low=0, high=self.num_local_experts, size=(1,)).item()  # selected expert
            if self.expert_score_weight.requires_grad:
                self.expert_soup()
                expert_output = self.expert_soup_forward(input[0])
            else:
                expert_output = self.get_expert_by_idx(expert_idx)(input[0])
        else:
            self.expert_soup()
            expert_output = self.expert_soup_forward(input[0])

        return expert_output


class ExpertSoup(nn.Module):
    def __init__(self, dim, r, act=None, num_expert=4, sharing_down=0, sharing_up=0):
        super().__init__()

        self.act = act
        if sharing_down == 1:
            self.MoA_A = MixtureSoup(nn.Linear(dim, r), 1)
        else:
            self.MoA_A = MixtureSoup(nn.Linear(dim, r), num_expert)
        if act is not None:
            self.act = gelu

        if sharing_up == 1:
            self.MoA_B = MixtureSoup(nn.Linear(r, dim), 1)
        else:
            self.MoA_B = MixtureSoup(nn.Linear(r, dim), num_expert)

    def forward(self, x, residual):
        result = self.MoA_A(x)
        if self.act is not None:
            result = self.act(result)
        result = self.MoA_B(result)
        return result + residual
