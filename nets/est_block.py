import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import radius_graph
from torch_scatter import scatter

import e3nn
from e3nn import o3
from e3nn.o3 import FromS2Grid, ToS2Grid
from e3nn.util.jit import compile_mode
from e3nn.nn.models.v2106.gate_points_message_passing import tp_path_exists

import torch_geometric
import math

from .registry import register_model
from .instance_norm import EquivariantInstanceNorm
from .graph_norm import EquivariantGraphNorm
from .layer_norm import EquivariantLayerNormV2
from .fast_layer_norm import EquivariantLayerNormFast
from .radial_func import RadialProfile
from .tensor_product_rescale import (TensorProductRescale, LinearRS,
    FullyConnectedTensorProductRescale, irreps2gate, sort_irreps_even_first)
from .fast_activation import Activation, Gate
from .drop import EquivariantDropout, EquivariantScalarsDropout, GraphDropPath
from .gaussian_rbf import GaussianRadialBasisLayer
# from nets.graph_attention_transformer import SeparableFCTP

# for bessel radial basis
from ocpmodels.models.gemnet.layers.radial_basis import RadialBasis

class EdgeEST(nn.Module):
    def __init__(self, irreps_node_input, irreps_node_attr, irreps_edge_attr, irreps_node_output, fc_neurons, irreps_head, num_heads, irreps_pre_attn, rescale_degree, nonlinear_message, alpha_drop, proj_drop, to_grid, mid_irreps):

        super().__init__()
        self.irreps_node_input = irreps_node_input
        self.irreps_edge_attr = irreps_edge_attr
        self.irreps_pre_attn = irreps_pre_attn
        self.mid_irreps = mid_irreps
        # self.to_grid = to_grid
        self.to_grid = ToS2Grid_block(self.irreps_node_input.lmax, res=64)
        self.merge_src = LinearRS(self.irreps_node_input, self.mid_irreps, bias=True)
        self.merge_dst = LinearRS(self.irreps_node_input, self.mid_irreps, bias=False)
        self.fc_edge = LinearRS(self.irreps_edge_attr, self.mid_irreps, bias=True)
        self.sep_act = SeparableFCTP(self.mid_irreps, 
                self.irreps_edge_attr, self.mid_irreps, fc_neurons, 
                use_activation=True, norm_layer=None, internal_weights=False)

        # sphere
        self.reshape_module = reshape(self.mid_irreps)
        self.dim_in = mid_irreps[0].mul
        self.reduce_channel = self.dim_in // 4
        self.dim_out = self.dim_in
        self.ln1 = nn.LayerNorm(self.dim_in)
        self.ln2 = nn.LayerNorm(self.dim_in)
        self.ln3 = nn.LayerNorm(self.dim_in)
        self.W_Q = nn.Linear(self.dim_in, self.reduce_channel)
        self.W_K = nn.Linear(self.dim_in, self.reduce_channel)
        self.W_V = nn.Linear(self.dim_in, self.dim_in)

        # spherical FFN
        self.spherical_num_experts = 1
        self.steerable_num_experts = 2
        self.num_experts = self.spherical_num_experts + self.steerable_num_experts
        self.output = torch.nn.ModuleList([
            nn.Sequential(nn.Linear(self.dim_in, self.dim_in * 2),
            nn.SiLU(),
            nn.Linear(self.dim_in * 2, self.dim_in))
            for _ in range(self.spherical_num_experts)
        ])

        # steerable ffn
        irreps_mlp_mid = self.mid_irreps
        self.steerable_ffn = FeedForwardNetwork(self.mid_irreps, irreps_node_attr, self.mid_irreps, irreps_mlp_mid, proj_drop, self.steerable_num_experts)

        # gate, edge-attn
        self.fc_gate = LinearRS(self.mid_irreps, o3.Irreps(f"{self.num_experts}x0e"), bias=True)
        self.fc_edge_attn1 = LinearRS(self.mid_irreps, o3.Irreps(f"{self.dim_in}x0e"), bias=True)
        self.act_edge_attn = nn.LeakyReLU()
        self.fc_edge_attn2 = nn.Linear(self.dim_in, 1)

        # output layer
        self.out_layer = LinearRS(self.mid_irreps, self.irreps_node_input, bias=False)

    def forward(self, node_input, node_attr, edge_src, edge_dst, edge_attr, edge_scalars, batch):
        message_src = self.merge_src(node_input)
        message_dst = self.merge_dst(node_input)
        message = message_src[edge_src] + message_dst[edge_dst]

        # steerable ffn
        # steerable_message = self.sep(message, edge_attr=edge_attr, edge_scalars=edge_scalars)
        weight = self.sep_act.dtp_rad(edge_scalars)
        steerable_message = self.sep_act.dtp(message, edge_attr, weight)
        steerable_message = self.sep_act.lin(steerable_message)
        steerable_message = self.sep_act.gate(steerable_message)

        # gate
        gate_logit = self.fc_gate(steerable_message)
        gate_logit = torch.softmax(gate_logit, -1)
        gate_logit_steerable, gate_logit_spherical = gate_logit[:, :self.steerable_num_experts], gate_logit[:, self.steerable_num_experts:]
        edge_attn = self.fc_edge_attn1(steerable_message)
        edge_attn = self.fc_edge_attn2(self.act_edge_attn(edge_attn))
        edge_attn = torch_geometric.utils.softmax(edge_attn, edge_dst)

        # to grid
        edge_attr = self.fc_edge(edge_attr)
        edge_attr = self.reshape_module(edge_attr, mode='irreps2matrix')
        message = self.reshape_module(message, mode='irreps2matrix')
        message = self.to_grid.ToGrid(message)
        edge_attr = self.to_grid.ToGrid(edge_attr)
        
        # self-attention
        edge_num, s, _ = message.shape
        res = message
        message = self.ln1(message)
        edge_attr = self.ln2(edge_attr)
        q = self.W_Q(message)
        k = self.W_K(edge_attr)
        v = self.W_V(message)
        q = self.spherical_rope(q)
        k = self.spherical_rope(k)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.view(edge_num, s, -1)
        out = out + res
        
        out = self.ln3(out)
        expert_outputs = [expert(out) for expert in self.output]
        expert_outputs = torch.stack(expert_outputs, dim=1)
        gate_output = gate_logit_spherical.view(edge_num, -1, 1, 1)
        node_output = (gate_output * expert_outputs).sum(1)
        out = self.to_grid.FromGrid(node_output)
        out = self.reshape_module(out, mode='matrix2irreps')

        # steerable ffn
        steerable_message_attr = torch.ones_like(steerable_message.narrow(1, 0, 1))
        out_message = self.steerable_ffn(steerable_message, steerable_message_attr, out, gate_logit_steerable)
        out_message = out_message * edge_attn
        out_message = scatter(out_message, index=edge_dst, dim=0, dim_size=node_input.shape[0])
        out_message = self.out_layer(out_message)
        return out_message

        
    def spherical_rope(self, x):
        # # concatenate
        b, s, c = x.shape
        position_embedding = self.to_grid.grid_position_embedding.view(1, s, 3)
        position_embedding = position_embedding.expand(b, -1, -1)
        return torch.cat([x, position_embedding], dim=-1)

class NodeEST(nn.Module):
    def __init__(self, irreps_node_input, irreps_edge_attr, to_grid, mid_irreps, num_experts):
        super().__init__()
        self.dim_in = mid_channel = mid_irreps[0].mul
        self.reduce_channel = self.dim_in
        self.dim_out = mid_channel
        self.ln1 = nn.LayerNorm(self.dim_in)
        # self.ln2 = nn.LayerNorm(self.dim_in)
        self.W_Q = nn.Linear(self.dim_in, self.reduce_channel)
        self.W_K = nn.Linear(self.dim_in, self.reduce_channel)
        self.W_V = nn.Linear(self.dim_in, self.reduce_channel)

        self.output = torch.nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(self.dim_in),
                nn.Linear(self.dim_in, self.dim_in * 2),
                nn.SiLU(),
                nn.Linear(self.dim_in * 2, self.dim_in))
            for _ in range(num_experts)
        ])

        self.mid_irreps = mid_irreps
        self.reshape_module = reshape(mid_irreps)
        self.to_grid = to_grid

        self.act = nn.SiLU()
    

    def forward(self, message, gate_output):
        # message
        message = self.to_grid.ToGrid(message)
        message = self.ln1(message)
        edge_num, s, c = message.shape
        message = message.view(edge_num, s, -1)

        q = self.W_Q(message)
        k = self.W_K(message)
        v = self.W_V(message)
        q = self.spherical_rope(q)
        k = self.spherical_rope(k)
        
        # with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
        out = F.scaled_dot_product_attention(q, k, v)
    
        out = out.view(edge_num, s, -1)
        # out = self.ln2(out)

        expert_outputs = [expert(out) for expert in self.output]
        expert_outputs = torch.stack(expert_outputs, dim=1)
        
        gate_output = gate_output.view(edge_num, -1, 1, 1)
        node_output = (gate_output * expert_outputs).sum(1)

        out = self.to_grid.FromGrid(node_output)
        return out

    def spherical_rope(self, x):
        # # concatenate
        b, s, c = x.shape
        position_embedding = self.to_grid.grid_position_embedding.view(1, s, 3).to(device=x.device)
        position_embedding = position_embedding.expand(b, -1, -1)
        return torch.cat([x, position_embedding], dim=-1)

    def scaled_dot_product_attention(self, query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
        L, S = query.size(-2), key.size(-2)
        B = query.size(0)
        sphere = query.size(1)

        scale_factor = 1 / (math.sqrt(query.size(-1))) if scale is None else scale

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight = torch.softmax(attn_weight, dim=-1)
        return attn_weight @ value

def CalcSpherePoints(num_points: int, device: str = "cpu") -> torch.Tensor:
    goldenRatio = (1 + 5**0.5) / 2
    i = torch.arange(num_points, device=device).view(-1, 1)
    theta = 2 * math.pi * i / goldenRatio
    phi = torch.arccos(1 - 2 * (i + 0.5) / num_points)
    points = torch.cat(
        [
            torch.cos(theta) * torch.sin(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(phi),
        ],
        dim=1,
    )
    # weight the points by their density
    pt_cross = points.view(1, -1, 3) - points.view(-1, 1, 3)
    pt_cross = torch.sum(pt_cross**2, dim=2)
    pt_cross = torch.exp(-pt_cross / (0.5 * 0.3))
    scalar = 1.0 / torch.sum(pt_cross, dim=1)
    scalar = num_points * scalar / torch.sum(scalar)
    res = points * (scalar.view(-1, 1))
    return res, -res

class ToS2Grid_block:
    def __init__(self, l, res, device='cuda'):
        self.grid_res = res
        self.device = device
        self.sampling_num = res
 
        sphere_points, neg_sphere_points = CalcSpherePoints(self.sampling_num, 'cuda')
        sphharm_weights = o3.spherical_harmonics(
            torch.arange(0, l + 1).tolist(), sphere_points, normalize=True
        ).detach().unsqueeze(0)
        # neg_sphharm_weights = o3.spherical_harmonics(
        #     torch.arange(0, l + 1).tolist(), neg_sphere_points, normalize=True
        # ).detach().unsqueeze(0)
        # self.to_grid_mat = torch.cat([sphharm_weights, neg_sphharm_weights], dim=0).view(2, self.sampling_num, -1).contiguous()
        self.to_grid_mat = sphharm_weights.view(self.sampling_num, -1).contiguous()
        self.from_grid_mat = sphharm_weights.view(self.sampling_num, -1) / (sphharm_weights.view(self.sampling_num, -1) * sphharm_weights.view(self.sampling_num, -1)).sum(0, keepdim=True).contiguous()

        self.grid_position_embedding = sphere_points.view(1, self.sampling_num, 1, 3).contiguous()

    def ToGrid(self, x):
        to_grid_mat = self.to_grid_mat.to(device=x.device)
        return torch.einsum("si,zci->zsc", to_grid_mat, x).contiguous()
    
    def FromGrid(self, x_grid):
        from_grid_mat = self.from_grid_mat.to(device=x_grid.device)
        return torch.einsum("si,zsc->zci", from_grid_mat, x_grid).contiguous()

class reshape(torch.nn.Module):
    def __init__(self, irreps):
        super().__init__()
        self.irreps = irreps

    def forward(self, x, mode='irreps2matrix'):
        if mode == 'irreps2matrix':
            return self.irreps2matrix(x)
        elif mode == 'matrix2irreps':
            return self.matrix2irreps(x)
        else:
            raise NotImplementedError('Unseen mode')
    
    def irreps2matrix(self, x):
        ix = 0
        out = torch.tensor([], dtype=x.dtype, device=x.device)
        for mul, ir in self.irreps:
            d = ir.dim
            field = x[:, ix: ix + mul * d]
            ix = ix + mul * d

            field = field.reshape(-1, mul, d)
            out = torch.cat([out, field], dim=-1)
        return out
    
    def matrix2irreps(self, x):
        index_d = 0
        out = torch.tensor([], dtype=x.dtype, device=x.device)
        for mul, ir in self.irreps:
            d = ir.dim
            field = x[:, :, index_d : index_d + d]
            index_d = index_d + d 

            field = field.reshape(-1, mul * d)
            out = torch.cat([out, field], dim=-1)
        return out
        