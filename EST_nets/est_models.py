import torch
import torch.nn as nn
from torch_cluster import radius_graph
from torch_scatter import scatter

from e3nn import o3

from .basis import GaussianRadialBasisLayer
from .common_block import (
    BesselBasis,
    Edgeembedblock,
    EquivariantLayerNormArraySphericalHarmonics,
)
from .est_block import ToS2Grid_block
from .est_layers import EST_layer


class EquivariantSphericalTransformer(nn.Module):
    def __init__(self,
        embed_dim=128, 
        num_layers=6,
        lmax=2,
        number_of_basis=128,
        basis_type='gaussian',
        hidden_state_num=512,
        max_radius=5.0,
        drop_ratio=0.0,
        num_elements=20,
        message_steerable_nums=3,
        message_spherical_nums=1,
        update_steerable_nums=4,
        update_spherical_nums=4,
        grid_resolution=64,
        avg_atoms=18.03065905448718,
        avg_degree=15.57930850982666,
    ):
        super().__init__()

        # init
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.lmax = lmax
        self.sh_basis = (self.lmax + 1) ** 2
        self.number_of_basis = number_of_basis
        self.basis_type = basis_type
        self.hidden_state_num = hidden_state_num
        self.max_radius = max_radius
        self.drop_ratio = drop_ratio
        self.num_elements = num_elements
        self.grid_resolution = grid_resolution
        self.avg_atoms = avg_atoms
        self.avg_degree = avg_degree

        # MoE hyper-parameters
        self.message_steerable_nums = message_steerable_nums
        self.message_spherical_nums = message_spherical_nums
        self.update_steerable_nums = update_steerable_nums
        self.update_spherical_nums = update_spherical_nums

        # Atomic Embedding
        self.atom_embed = nn.Embedding(num_elements, self.embed_dim)

        # radius function
        if self.basis_type == "gaussian":
            self.rbf = GaussianRadialBasisLayer(self.number_of_basis, cutoff=self.max_radius)
        elif self.basis_type == "bessel":
            self.rbf = BesselBasis(
                cutoff=self.max_radius, n_rbf=self.number_of_basis
            )
        else:
            raise ValueError("basis_type must be either 'gaussian' or 'bessel'")
        
        # init edge embedding
        self.edge_embed = Edgeembedblock(self.lmax, self.embed_dim, self.number_of_basis, avg_degree)

        # K-layers EST Modules
        self.est_layers = nn.ModuleList()
        self.FL = ToS2Grid_block(self.lmax, res=self.grid_resolution)
        for _ in range(self.num_layers):
            self.est_layers.append(EST_layer(
                self.lmax, 
                self.embed_dim, 
                self.hidden_state_num, 
                self.number_of_basis, 
                self.num_elements, 
                self.max_radius,
                self.FL, 
                self.FL, 
                self.message_steerable_nums, 
                self.message_spherical_nums, 
                self.update_steerable_nums, 
                self.update_spherical_nums, 
                drop_ratio=self.drop_ratio)
                )

        # ouput layer
        self.norm = EquivariantLayerNormArraySphericalHarmonics(self.lmax, self.embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, 1)
        )

    def forward(self, f_in, pos, batch, node_atom, **kwargs) -> torch.Tensor:
        
        node_embedding = torch.zeros(
            node_atom.size(0),
            self.sh_basis,
            self.embed_dim,
            dtype=pos.dtype,
            device=pos.device,
        )
        node_embedding[:, 0, :] = self.atom_embed(node_atom)

        edge_index = radius_graph(pos, r=self.max_radius, batch=batch,
            max_num_neighbors=1000)
        edge_vec = pos.index_select(0, edge_index[0]) - pos.index_select(0, edge_index[1])
        edge_sh = o3.spherical_harmonics(l=torch.arange(self.lmax + 1).tolist(),
            x=edge_vec, normalize=True, normalization='component')
        edge_sh = edge_sh.unsqueeze(-1)

        edge_length = edge_vec.norm(dim=1)
        edge_length_embedding = self.rbf(edge_length)
        node_embedding = self.edge_embed(node_embedding, edge_sh, 
            edge_length_embedding, edge_index)

        for layer in self.est_layers:
            node_embedding = layer(node_embedding, edge_index, edge_sh, edge_length, node_atom, batch)
        
        output = self.norm(node_embedding)[:, 0, :]
        output = self.fc(output)
        molecular_out = scatter(output, index=batch, dim=0) / self.avg_atoms

        return molecular_out

def est_4layer():
    return EquivariantSphericalTransformer(
        embed_dim=64, 
        num_layers=4,
        lmax=3,
        number_of_basis=128,
        basis_type='gaussian',
        hidden_state_num=1256,
        max_radius=5.0,
        drop_ratio=0.2,
        num_elements=20,
        avg_atoms=18.03065905448718,
        avg_degree=15.57930850982666,
        )

def est_6layer():
    return EquivariantSphericalTransformer(
        embed_dim=128, 
        num_layers=6,
        lmax=2,
        number_of_basis=128,
        basis_type='gaussian',
        hidden_state_num=512,
        max_radius=5.0,
        drop_ratio=0.2,
        num_elements=20,
        avg_atoms=18.03065905448718,
        avg_degree=15.57930850982666,
        )
        
