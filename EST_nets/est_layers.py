import torch
import torch.nn as nn
from torch_scatter import scatter

from .common_block import (
    DistanceEmbed,
    EquiExpert,
    EquivariantLayerNormArraySphericalHarmonics,
    GateActivation,
    SO3_LinearV2,
)
from .est_block import EdgeEST, NodeEST, ToS2Grid_block

_AVG_DEGREE = 15.57930850982666

class EST_message(nn.Module):
    def __init__(self,
        lmax=2, 
        channels=128, 
        cutoff=5.0,
        FL_block=None,
        num_basis=128,
        num_elements=20,
        edge_com_mode='add',
        steerable_num_experts=3,
        spherical_num_experts=1,
        ):
        super().__init__()

        # init
        self.lmax = lmax
        self.channels = channels
        self.num_basis = num_basis
        self.num_elements = num_elements
        self.cutoff = cutoff
        # FFT block based on Fibonacci Lattices
        if FL_block is None:
            self.FL_block = ToS2Grid_block(l=self.lmax, res=64)
        else:
            self.FL_block = FL_block

        # encoding edge features
        self.fc_src = SO3_LinearV2(self.channels, self.channels, self.lmax)
        self.fc_tgt = SO3_LinearV2(self.channels, self.channels, self.lmax)

        # edge embedding
        self.dist_embedding = DistanceEmbed(num_basis, self.channels, self.num_elements, cutoff)
        # self.edge_embed1 = nn.Linear(self.num_basis, self.channels)
        # self.edge_embed2 = nn.Linear(self.channels, self.channels)
        # self.leakrelu = SmoothLeakyReLU(0.2)

        # compute message
        if edge_com_mode != "add":
            raise ValueError("EST_message currently supports edge_com_mode='add' only")
        self.fc_message1 = SO3_LinearV2(self.channels, self.channels, self.lmax)
        self.fc_message2 = SO3_LinearV2(self.channels, self.channels, self.lmax)
        self.fc_edge_sh1 = SO3_LinearV2(1, self.channels, self.lmax)
        self.fc_edge_sh2 = SO3_LinearV2(2 * self.channels, self.channels, self.lmax)
        self.gate_edge = GateActivation(self.lmax, self.lmax, self.channels)
        self.gate_message = GateActivation(self.lmax, self.lmax, self.channels)

        # MoE
        self.steerable_num_experts = steerable_num_experts
        self.spherical_num_experts = spherical_num_experts
        self.num_expert = self.steerable_num_experts + self.spherical_num_experts
        self.fc_gate = nn.Sequential(
            nn.LayerNorm(self.channels),
            nn.Linear(self.channels, self.channels),
            nn.SiLU(),
            nn.Linear(self.channels, self.num_expert)
            )

        # Experts
        self.edgeEST = EdgeEST(self.lmax, self.channels, to_grid=self.FL_block, spherical_num_experts=self.spherical_num_experts, edge_info=True)
        self.message_steerable_expert = EquiExpert(self.lmax, self.channels, steerable_num_experts=self.steerable_num_experts, edge_info=True)

        # MLP for scalares
        self.fc_scalars_edge = nn.Sequential(
            nn.Linear(self.channels, self.channels),
            nn.SiLU(),
            nn.Linear(self.channels, self.lmax * self.channels)
        )
        self.fc_scalars_message = nn.Sequential(
            nn.Linear(self.channels, self.channels),
            nn.SiLU(),
            nn.Linear(self.channels, self.lmax * self.channels)
        )

        self.avg_degree = _AVG_DEGREE

    def forward(self, x_node, edge_index, edge_attr, edge_scalars, node_type, batch):
        ori_edge_attr = edge_attr

        message_src = self.fc_src(x_node)
        message_dst = self.fc_tgt(x_node)

        message = message_src[edge_index[0]] + message_dst[edge_index[1]]

        # edge embedding
        edge_weight = self.dist_embedding(edge_scalars, node_type, edge_index)
        edge_weight = edge_weight.view(edge_weight.size(0), 1, -1)
        # edge_weight = self.edge_embed1(edge_weight)
        # edge_weight = self.act(edge_weight)
        # edge_weight = self.edge_embed2(edge_weight)
        # edge_weight = torch.sigmoid(edge_weight)
        # edge_weight = edge_weight.unsqueeze(1)

        edge_attr = self.fc_edge_sh1(edge_attr)
        edge_attr = torch.cat([edge_attr, message], dim=-1)
        edge_attr = self.fc_edge_sh2(edge_attr)

        gate_scalars_edge = self.fc_scalars_edge(edge_attr[:, 0, :])
        edge_attr = self.gate_edge(gate_scalars_edge, edge_attr)

        message = self.fc_message1(message)
        gate_scalars_message = self.fc_scalars_message(edge_attr[:, 0, :])
        message = self.gate_message(gate_scalars_message, message)
        message = (message + edge_attr) * edge_weight
        message = self.fc_message2(message)

        # MoE Gate
        gate_logit = self.fc_gate(message[:, 0, :])
        gate_logit_steerable, gate_logit_spherical = gate_logit[:, :self.steerable_num_experts], gate_logit[:, self.steerable_num_experts:]


        # EST Expert
        spherical_out = self.edgeEST(message, ori_edge_attr, gate_logit_spherical)
        # gate_logit_spherical = gate_logit_spherical.view(gate_logit_spherical.size(0), 1, 1, -1)
        # spherical_out = (spherical_out * gate_logit_spherical).sum(-1)

        # Strict equivariant Expert
        steerable_out = self.message_steerable_expert(message, ori_edge_attr, gate_logit_steerable)

        edge_out = spherical_out + steerable_out

        node_out = scatter(
            edge_out, index=edge_index[1], dim=0, dim_size=x_node.shape[0]
        ) / self.avg_degree

        return node_out

class EST_update(nn.Module):
    def __init__(self, 
        channels, 
        lmax,
        FL_block,
        steerable_num_experts=4,
        spherical_num_experts=4):
        super().__init__()
        self.lmax = lmax
        self.channels = channels
        self.FL_block = FL_block
        self.steerable_num_experts = steerable_num_experts
        self.spherical_num_experts = spherical_num_experts
        self.num_expert = self.steerable_num_experts + self.spherical_num_experts
        self.fc_gate = nn.Sequential(
            nn.LayerNorm(self.channels),
            nn.Linear(self.channels, self.channels),
            nn.SiLU(),
            nn.Linear(self.channels, self.num_expert)
            )

        # Experts
        self.nodeEST = NodeEST(self.lmax, self.channels, to_grid=self.FL_block, num_experts=self.spherical_num_experts)
        self.steerable_expert = EquiExpert(self.lmax, self.channels, steerable_num_experts=self.steerable_num_experts, edge_info=False)

    def forward(self, x_node):
         # MoE Gate
        gate_logit = self.fc_gate(x_node[:, 0, :])
        gate_logit_steerable, gate_logit_spherical = gate_logit[:, :self.steerable_num_experts], gate_logit[:, self.steerable_num_experts:]

        # EST Expert
        spherical_out = self.nodeEST(x_node, gate_logit_spherical)

        # Strict equivariant Expert
        steerable_out = self.steerable_expert(x_node, weights=gate_logit_steerable)

        out = spherical_out + steerable_out

        return out

class EST_layer(nn.Module):
    def __init__(
        self, 
        lmax, 
        embedding_channels, 
        hidden_state_num, 
        num_basis, 
        num_elements, 
        cutoff,
        FL_message, 
        FL_update, 
        message_steerable_nums,
        message_spherical_nums,
        update_steerable_nums,
        update_spherical_nums,
        drop_ratio=0.1
        ):
        super().__init__()

        # init
        self.lmax = lmax
        self.embedding_channels = embedding_channels

        # message block
        self.norm_message = EquivariantLayerNormArraySphericalHarmonics(
            lmax=self.lmax, 
            num_channels=self.embedding_channels
            )
        self.message = EST_message(
            lmax=self.lmax, 
            channels=self.embedding_channels,
            cutoff=cutoff,
            FL_block=FL_message, 
            num_basis=num_basis, 
            num_elements=num_elements, 
            steerable_num_experts=message_steerable_nums, 
            spherical_num_experts=message_spherical_nums,
            )

        # update block
        self.moe_update = EST_update(
            channels=self.embedding_channels, 
            lmax=self.lmax, 
            FL_block=FL_update,
            steerable_num_experts=update_steerable_nums, 
            spherical_num_experts=update_spherical_nums,
            )

        self.dropout = nn.Dropout(drop_ratio)

    
    def forward(self, x, edge_index, edge_attr, edge_scalars, node_type, batch, **kwargs):
        # Message block feedforward

        # message
        shortcut = x_node = x
        x_node = self.norm_message(x)
        x_node = self.message(x_node, edge_index, edge_attr, edge_scalars, node_type, batch)
        x_node = x_node + shortcut

        # update
        shortcut = x_node
        x_node = self.moe_update(x_node)
        x_node = self.dropout(x_node)

        x_node = x_node + shortcut

        return x_node


