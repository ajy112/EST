import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3

from .common_block import SO3_LinearV2

class EdgeEST(nn.Module):
    def __init__(self, lmax, channels, to_grid, spherical_num_experts=1, edge_info=True):
        """
            EST is used for message block. Here, q, v are both from message, k is from edge attrbutes
            Forward Input:
                message: [num_edge, number of spherical harmonic basis, channel]
                edge_attr: [num_edge, number of spherical harmonic basis, 1]
                weights: [num_edge, number of steerable experts]
        """
        super().__init__()
        self.channels = channels
        self.lmax = lmax
        self.spherical_num_experts = spherical_num_experts
        self.fc_edge = SO3_LinearV2(1, self.channels, self.lmax)
        self.ln1 = nn.LayerNorm(self.channels)
        self.ln2 = nn.LayerNorm(self.channels) if edge_info else None

        self.reduce_channel = self.channels // 2
        self.W_Q = nn.Linear(self.channels, self.reduce_channel, bias=False)
        self.W_K = nn.Linear(self.channels, self.reduce_channel, bias=False)
        self.W_V = nn.Linear(self.channels, self.channels, bias=False)

        self.ffn_list = torch.nn.ModuleList([
            nn.Sequential(
            nn.LayerNorm(self.channels),
            nn.Linear(self.channels, self.channels * 2),
            nn.SiLU(),
            nn.Linear(self.channels * 2, self.channels))
            for _ in range(self.spherical_num_experts)
        ])

        self.to_grid = to_grid
        self.edge_info = edge_info

    def forward(self, message, edge_attr=None, weights=None):
        message = self.to_grid.ToGrid(message)
        edge_num, sampling_num, _ = message.shape
        residual = message
        message = self.ln1(message)

        if self.edge_info and edge_attr is not None:
            edge_attr = self.fc_edge(edge_attr)
            edge_attr = self.to_grid.ToGrid(edge_attr)
            edge_attr = self.ln2(edge_attr)
            key_input = edge_attr
        else:
            key_input = message

        query = self.spherical_rope(self.W_Q(message))
        key = self.spherical_rope(self.W_K(key_input))
        value = self.W_V(message)
        out = F.scaled_dot_product_attention(query, key, value)
        out = out.view(edge_num, sampling_num, -1) + residual

        expert_outputs = [expert(out) for expert in self.ffn_list]
        expert_outputs = torch.stack(expert_outputs, dim=1)
        if weights is not None:
            expert_outputs = weights.view(edge_num, -1, 1, 1) * expert_outputs
        com_output = expert_outputs.sum(1)
        out = self.to_grid.FromGrid(com_output)

        return out
        
    def spherical_rope(self, x):
        batch_size, sampling_num, _ = x.shape
        position_embedding = self.to_grid.grid_position_embedding.view(
            1, sampling_num, 3
        ).to(device=x.device, dtype=x.dtype)
        position_embedding = position_embedding.expand(batch_size, -1, -1)
        return torch.cat([x, position_embedding], dim=-1)

class NodeEST(nn.Module):
    def __init__(self, lmax, channels, to_grid, num_experts):
        """
            EST is used for update block. Here, q, k, v are all from aggregated message
            Forward Input:
                message_aggr: [num_node, number of spherical harmonic basis, channel]
                weights: [num_node, number of spatial experts]
        """
        super().__init__()
        self.lmax = lmax
        self.dim_in = channels
        self.reduce_channel = channels
        self.dim_out = channels
        self.ln1 = nn.LayerNorm(self.dim_in)
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

        self.to_grid = to_grid

    def forward(self, message_aggr, weights=None):
        message_aggr = self.to_grid.ToGrid(message_aggr)
        message_aggr = self.ln1(message_aggr)
        node_num, sampling_num, _ = message_aggr.shape

        query = self.spherical_rope(self.W_Q(message_aggr))
        key = self.spherical_rope(self.W_K(message_aggr))
        value = self.W_V(message_aggr)
        out = F.scaled_dot_product_attention(query, key, value)
        out = out.view(node_num, sampling_num, -1)

        expert_outputs = [expert(out) for expert in self.output]
        expert_outputs = torch.stack(expert_outputs, dim=1)
        if weights is not None:
            weights = weights.view(node_num, -1, 1, 1)
            expert_outputs = weights * expert_outputs
        expert_outputs = expert_outputs.sum(1)

        out = self.to_grid.FromGrid(expert_outputs)
        return out

    def spherical_rope(self, x):
        batch_size, sampling_num, _ = x.shape
        position_embedding = self.to_grid.grid_position_embedding.view(
            1, sampling_num, 3
        ).to(device=x.device, dtype=x.dtype)
        position_embedding = position_embedding.expand(batch_size, -1, -1)
        return torch.cat([x, position_embedding], dim=-1)

def CalcSpherePoints(num_points: int, device: str = "cpu") -> torch.Tensor:
    """
        Calculating Fibonacci Lattices sphere
    """
    golden_ratio = (1 + 5**0.5) / 2
    i = torch.arange(num_points, device=device).view(-1, 1)
    theta = 2 * math.pi * i / golden_ratio
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
    def __init__(self, l, res, device="cpu"):
        """
            1.used for Fibonacci Lattices Sampling (to_grid_mat, from_grid_mat)
            2.provide the basis of spherical relative position embedding (grid_position_embedding)
        """
        self.grid_res = res
        self.device = device
        self.sampling_num = res
 
        sphere_points, _ = CalcSpherePoints(self.sampling_num, device)
        sphharm_weights = o3.spherical_harmonics(
            torch.arange(0, l + 1).tolist(), sphere_points, normalize=True
        ).detach().unsqueeze(0)
        self.to_grid_mat = sphharm_weights.view(self.sampling_num, -1).contiguous()
        squared_norm = self.to_grid_mat.square().sum(0, keepdim=True)
        self.from_grid_mat = (self.to_grid_mat / squared_norm).contiguous()

        self.grid_position_embedding = sphere_points.view(1, self.sampling_num, 1, 3).contiguous()

    def ToGrid(self, x):
        to_grid_mat = self.to_grid_mat.to(device=x.device, dtype=x.dtype)
        return torch.einsum("si,zic->zsc", to_grid_mat, x).contiguous()
    
    def FromGrid(self, x_grid):
        from_grid_mat = self.from_grid_mat.to(
            device=x_grid.device, dtype=x_grid.dtype
        )
        return torch.einsum("si,zsc->zic", from_grid_mat, x_grid).contiguous()
