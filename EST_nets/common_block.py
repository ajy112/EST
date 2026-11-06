import torch
import torch.nn as nn
from e3nn.o3 import FromS2Grid, ToS2Grid
from torch_scatter import scatter

import math

def gaussian_rbf(inputs: torch.Tensor, offsets: torch.Tensor, widths: torch.Tensor):
    """
    Compute Gaussian radial basis functions.

    Args:
        inputs (torch.Tensor): Input distances. Shape: [..., 1]
        offsets (torch.Tensor): Centers of the Gaussian functions. Shape: [n_rbf]
        widths (torch.Tensor): Widths of the Gaussian functions. Shape: [n_rbf]

    Returns:
        torch.Tensor: Gaussian RBF values. Shape: [..., n_rbf]
    """
    coeff = -0.5 / torch.pow(widths, 2)
    diff = inputs[..., None] - offsets
    y = torch.exp(coeff * torch.pow(diff, 2))
    return y


class GaussianRBF(nn.Module):
    """
    Gaussian radial basis functions module.

    Expands distances using Gaussian functions centered at different offsets.

    Args:
        n_rbf (int): Total number of Gaussian functions.
        cutoff (float): Center of the last Gaussian function (maximum distance).
        start (float, optional): Center of the first Gaussian function. Defaults to 0.0.
        trainable (bool, optional): If True, widths and offsets are learnable parameters. Defaults to False.
    """
    def __init__(
        self, n_rbf: int, cutoff: float, start: float = 0.0, trainable: bool = False
    ):
        super(GaussianRBF, self).__init__()
        self.n_rbf = n_rbf

        # compute offset and width of Gaussian functions
        offset = torch.linspace(start, cutoff, n_rbf)
        widths = torch.FloatTensor(
            torch.abs(offset[1] - offset[0]) * torch.ones_like(offset)
        )
        if trainable:
            self.widths = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer("widths", widths)
            self.register_buffer("offsets", offset)

    def forward(self, inputs: torch.Tensor):
        return gaussian_rbf(inputs, self.offsets, self.widths)


class BesselBasis(nn.Module):
    """
    Sine for radial basis expansion with coulomb decay (0th order Bessel functions).

    Used in DimeNet. Reference: https://arxiv.org/abs/2003.03123

    Args:
        cutoff (float, optional): Radial cutoff distance. Defaults to 5.0.
        n_rbf (int, optional): Number of basis functions. Defaults to None.
        trainable (bool, optional): Kept for compatibility, but parameters are not learnable. Defaults to False.
    """
    def __init__(self, cutoff=5.0, n_rbf=None, trainable=False):
        super(BesselBasis, self).__init__()
        if n_rbf is None:
            raise ValueError("n_rbf must be specified for BesselBasis")
        self.n_rbf = n_rbf
        # compute offset and width of Gaussian functions
        freqs = torch.arange(1, n_rbf + 1) * math.pi / cutoff
        self.register_buffer("freqs", freqs)
        self.register_buffer('norm1', torch.tensor(1.0))

    def forward(self, inputs):
        a = self.freqs[None,  :]
        inputs = inputs[..., None]
        ax = inputs * a
        sinax = torch.sin(ax)

        norm = torch.where(inputs == 0, self.norm1, inputs)
        y = sinax / norm

        return y

class EquivariantLayerNormArraySphericalHarmonics(nn.Module):
    """
    This class is Copies from EquiformerV2 (layer_norm_sh)
    1. Normalize over L = 0.
    2. Normalize across all m components from degrees L > 0.
    3. Do not normalize separately for different L (L > 0).
    """

    def __init__(
        self,
        lmax: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        normalization: str = "component",
        std_balance_degrees: bool = True,
    ):
        super().__init__()

        self.lmax = lmax
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.std_balance_degrees = std_balance_degrees

        # for L = 0
        self.norm_l0 = torch.nn.LayerNorm(
            self.num_channels, eps=self.eps, elementwise_affine=self.affine
        )

        # for L > 0
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.lmax, self.num_channels))
        else:
            self.register_parameter("affine_weight", None)

        assert normalization in ["norm", "component"]
        self.normalization = normalization

        if self.std_balance_degrees:
            balance_degree_weight = torch.zeros((self.lmax + 1) ** 2 - 1, 1)
            for lval in range(1, self.lmax + 1):
                start_idx = lval**2 - 1
                length = 2 * lval + 1
                balance_degree_weight[start_idx : (start_idx + length), :] = (
                    1.0 / length
                )
            balance_degree_weight = balance_degree_weight / self.lmax
            self.register_buffer("balance_degree_weight", balance_degree_weight)
        else:
            self.balance_degree_weight = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(lmax={self.lmax}, num_channels={self.num_channels}, eps={self.eps}, std_balance_degrees={self.std_balance_degrees})"

    @torch.autocast("cuda", enabled=False)
    def forward(self, node_input):
        """
        Assume input is of shape [N, sphere_basis, C]
        """

        out = []

        # for L = 0
        feature = node_input.narrow(1, 0, 1)
        feature = self.norm_l0(feature)
        out.append(feature)

        # for L > 0
        if self.lmax > 0:
            num_m_components = (self.lmax + 1) ** 2
            feature = node_input.narrow(1, 1, num_m_components - 1)

            # Then compute the rescaling factor (norm of each feature vector)
            # Rescaling of the norms themselves based on the option "normalization"
            if self.normalization == "norm":
                feature_norm = feature.pow(2).sum(dim=1, keepdim=True)  # [N, 1, C]
            elif self.normalization == "component":
                if self.std_balance_degrees:
                    feature_norm = feature.pow(
                        2
                    )  # [N, (L_max + 1)**2 - 1, C], without L = 0
                    feature_norm = torch.einsum(
                        "nic, ia -> nac",
                        feature_norm,
                        self.balance_degree_weight,
                    )  # [N, 1, C]
                else:
                    feature_norm = feature.pow(2).mean(dim=1, keepdim=True)  # [N, 1, C]

            feature_norm = torch.mean(feature_norm, dim=2, keepdim=True)  # [N, 1, 1]
            feature_norm = (feature_norm + self.eps).pow(-0.5)

            for lval in range(1, self.lmax + 1):
                start_idx = lval**2
                length = 2 * lval + 1
                feature = node_input.narrow(1, start_idx, length)  # [N, (2L + 1), C]
                if self.affine:
                    weight = self.affine_weight.narrow(0, (lval - 1), 1)  # [1, C]
                    weight = weight.view(1, 1, -1)  # [1, 1, C]
                    feature_scale = feature_norm * weight  # [N, 1, C]
                else:
                    feature_scale = feature_norm
                feature = feature * feature_scale
                out.append(feature)

        return torch.cat(out, dim=1)

class SO3_LinearV2(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, lmax: int, bias: bool = True
    ) -> None:
        """
        This class is Copies from EquiformerV2
        1. Use `torch.einsum` to prevent slicing and concatenation
        2. Need to specify some behaviors in `no_weight_decay` and weight initialization.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lmax = lmax

        self.weight = torch.nn.Parameter(
            torch.randn((self.lmax + 1), out_features, in_features)
        )
        bound = 1 / math.sqrt(self.in_features)
        torch.nn.init.uniform_(self.weight, -bound, bound)
        self.bias = torch.nn.Parameter(torch.zeros(out_features))

        expand_index = torch.zeros([(lmax + 1) ** 2]).long()
        for lval in range(lmax + 1):
            start_idx = lval**2
            length = 2 * lval + 1
            expand_index[start_idx : (start_idx + length)] = lval
        self.register_buffer("expand_index", expand_index)

    def forward(self, input_embedding):
        weight = torch.index_select(
            self.weight, dim=0, index=self.expand_index
        )  # [(L_max + 1) ** 2, C_out, C_in]
        out = torch.einsum(
            "bmi, moi -> bmo", input_embedding, weight
        )  # [N, (L_max + 1) ** 2, C_out]
        bias = self.bias.view(1, 1, self.out_features)
        out[:, 0:1, :] = out.narrow(1, 0, 1) + bias

        # out_embedding = SO3_Embedding(
        #     0,
        #     input_embedding.lmax_list.copy(),
        #     self.out_features,
        #     device=input_embedding.device,
        #     dtype=input_embedding.dtype,
        # )
        # out_embedding.set_embedding(out)
        # out_embedding.set_lmax_mmax(
        #     input_embedding.lmax_list.copy(), input_embedding.lmax_list.copy()
        # )

        return out

class SmoothLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()
        self.alpha = negative_slope
        
    
    def forward(self, x):
        x1 = ((1 + self.alpha) / 2) * x
        x2 = ((1 - self.alpha) / 2) * x * (2 * torch.sigmoid(x) - 1)
        return x1 + x2
    
    
    def extra_repr(self):
        return 'negative_slope={}'.format(self.alpha)

class GateActivation(torch.nn.Module):
    def __init__(self, lmax: int, mmax: int, num_channels: int) -> None:
        super().__init__()

        self.lmax = lmax
        self.mmax = mmax
        self.num_channels = num_channels

        # compute `expand_index` based on `lmax` and `mmax`
        num_components = 0
        for lval in range(1, self.lmax + 1):
            num_m_components = min((2 * lval + 1), (2 * self.mmax + 1))
            num_components = num_components + num_m_components
        expand_index = torch.zeros([num_components]).long()
        start_idx = 0
        for lval in range(1, self.lmax + 1):
            length = min((2 * lval + 1), (2 * self.mmax + 1))
            expand_index[start_idx : (start_idx + length)] = lval - 1
            start_idx = start_idx + length
        self.register_buffer("expand_index", expand_index)

        self.scalar_act = (
            torch.nn.SiLU()
        )  # SwiGLU(self.num_channels, self.num_channels)  # #
        self.gate_act = torch.nn.Sigmoid()  # torch.nn.SiLU() # #

    def forward(self, gating_scalars, input_tensors):
        """
        `gating_scalars`: shape [N, lmax * num_channels]
        `input_tensors`: shape  [N, (lmax + 1) ** 2, num_channels]
        """

        gating_scalars = self.gate_act(gating_scalars)
        gating_scalars = gating_scalars.reshape(
            gating_scalars.shape[0], self.lmax, self.num_channels
        )
        gating_scalars = torch.index_select(
            gating_scalars, dim=1, index=self.expand_index
        )

        input_tensors_scalars = input_tensors.narrow(1, 0, 1)
        input_tensors_scalars = self.scalar_act(input_tensors_scalars)

        input_tensors_vectors = input_tensors.narrow(1, 1, input_tensors.shape[1] - 1)
        input_tensors_vectors = input_tensors_vectors * gating_scalars

        return torch.cat((input_tensors_scalars, input_tensors_vectors), dim=1)

class SingleExpert(nn.Module):
    def __init__(self, lmax, channels, edge_info=True):
        super().__init__()
        if edge_info:
            self.fc_edge = SO3_LinearV2(1, 2 * channels, lmax)
        
        self.ln = EquivariantLayerNormArraySphericalHarmonics(lmax, channels)
        self.fc1 = SO3_LinearV2(channels, 2 * channels, lmax)
        self.fc_scalars = nn.Sequential(
            nn.Linear(2 * channels, 2 * channels),
            nn.SiLU(),
            nn.Linear(2 * channels, lmax * 2 * channels)
        )
        self.act = GateActivation(lmax, lmax, 2 * channels)
        self.fc2 = SO3_LinearV2(2 * channels, channels, lmax)

        self.edge_info = edge_info

    def forward(self, message, edge_sh=None):
        if self.edge_info and edge_sh is not None:
            edge_sh = self.fc_edge(edge_sh)
        
        message = self.ln(message)
        message = self.fc1(message)
        if self.edge_info and edge_sh is not None:
            message = message + edge_sh
        
        gate_scalars = self.fc_scalars(message[:, 0, :])
        message = self.act(gate_scalars, message)
        message = self.fc2(message)
        return message

class EquiExpert(nn.Module):
    def __init__(self, lmax, channels, steerable_num_experts=1, edge_info=True):
        super().__init__()
        self.ffn_list = nn.ModuleList()
        for _ in range(steerable_num_experts):
            self.ffn_list.append(SingleExpert(lmax, channels, edge_info=edge_info))

    def forward(self, message, edge_sh=None, weights=None):
        expert_outputs = [expert(message, edge_sh) for expert in self.ffn_list]
        expert_outputs = torch.stack(expert_outputs, dim=1)
        if weights is not None:
            gate_output = weights.view(message.size(0), -1, 1, 1)
            expert_outputs = gate_output * expert_outputs
        out = expert_outputs.sum(1)

        return out
    
class Edgeembedblock(nn.Module):
    def __init__(self, lmax, channels, num_basis, avg_degree, res=12, device="cpu"):
        super().__init__()
        self.lmax = lmax
        self.res = res
        self.channels = channels
        self.num_basis = num_basis
        self.device = device
        self.avg_degree = avg_degree

        to_grid = ToS2Grid(
            lmax,
            (res, res + 1),
            normalization="integral",
            device=self.device,
        )
        self.to_grid_mat = torch.einsum(
            "mbi,am->bai", to_grid.shb, to_grid.sha
        ).detach().contiguous()
        self.to_grid_mat = self.to_grid_mat.view(-1, self.to_grid_mat.size(-1))

        from_grid = FromS2Grid(
            (res, res + 1),
            lmax,
            normalization="integral",
            device=self.device,
        )
        self.from_grid_mat = torch.einsum(
            "am,mbi->bai", from_grid.sha, from_grid.shb
        ).detach().contiguous()
        self.from_grid_mat = self.from_grid_mat.view(-1, self.from_grid_mat.size(-1))

        # 2-layer MLP, multiplication after the first layer
        self.fc_edge1 = nn.Linear(1, self.channels, bias=False)
        self.act = nn.SiLU()
        self.fc_edge2 = nn.Linear(self.channels, self.channels, bias=False)

        # MLP for edge_scalars
        self.fc_scalar = nn.Sequential(
            nn.Linear(self.num_basis, self.channels),
            nn.SiLU(),
            nn.Linear(self.channels, self.channels)
        )

    def forward(self, node_embedding, edge_sh, edge_scalars, edge_index):

        # transform to sphere
        to_grid_mat = self.to_grid_mat.to(device=edge_sh.device, dtype=edge_sh.dtype)
        from_grid_mat = self.from_grid_mat.to(device=edge_sh.device, dtype=edge_sh.dtype)

        # encoding scalars
        edge_scalars = self.fc_scalar(edge_scalars)
        edge_scalars = edge_scalars.unsqueeze(1)

        # to spatial domain and point-wise Linear
        edge_sh_grid = torch.einsum('blc,sl->bsc', edge_sh, to_grid_mat)
        edge_sh_grid = self.fc_edge1(edge_sh_grid)
        edge_sh_grid = self.act(edge_sh_grid)
        edge_sh_grid = edge_sh_grid * edge_scalars
        edge_sh_grid = self.fc_edge2(edge_sh_grid)
        edge_sh = torch.einsum('bsc,sl->blc', edge_sh_grid, from_grid_mat)
        edge_sh = edge_sh.contiguous()

        node_sh = scatter(
            edge_sh,
            edge_index[1],
            dim=0,
            dim_size=node_embedding.size(0),
        ) / self.avg_degree

        node_embedding[:, 1:, :] = node_embedding[:, 1:, :] + node_sh[:, 1:, :]
        return node_embedding

class GaussianSmearing(torch.nn.Module):
    def __init__(
        self,
        start: float = -5.0,
        stop: float = 5.0,
        num_gaussians: int = 50,
        basis_width_scalar: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_output = num_gaussians
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (basis_width_scalar * (offset[1] - offset[0])).item() ** 2
        self.register_buffer("offset", offset, persistent=False)

    def forward(self, dist) -> torch.Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))

class DistanceEmbed(torch.nn.Module):
    """
    Edge Block: Compute invariant edge representation from edge diatances and atomic numbers

    Args:
        edge_channels (int):        Size of invariant edge embedding
        distance_expansion (func):  Function used to compute distance embedding
        max_num_elements (int):     Maximum number of atomic numbers
        act (function):             Non-linear activation function
    """
    def __init__(
        self,
        num_basis,
        edge_channels,
        max_num_elements,
        cutoff, 
    ) -> None:
        super().__init__()
        self.num_basis = num_basis
        self.distance_expansion = GaussianSmearing(
            0.0,
            cutoff,
            self.num_basis, 
        )
        self.in_channels = self.num_basis
        self.act = nn.SiLU()
        self.edge_channels = edge_channels
        self.max_num_elements = max_num_elements

        # Embedding function of the distance
        self.fc1_dist = nn.Linear(self.in_channels, self.edge_channels)

        # Embedding function of the atomic numbers
        self.source_embedding = nn.Embedding(self.max_num_elements, self.edge_channels)
        self.target_embedding = nn.Embedding(self.max_num_elements, self.edge_channels)
        nn.init.uniform_(self.source_embedding.weight.data, -0.001, 0.001)
        nn.init.uniform_(self.target_embedding.weight.data, -0.001, 0.001)

        # Embedding function of the edge
        self.fc1_edge_attr = nn.Linear(
            self.edge_channels,
            self.edge_channels,
        )

    def forward(self, edge_distance, element, edge_index):
        # Compute distance embedding
        x_dist = self.distance_expansion(edge_distance)
        x_dist = self.fc1_dist(x_dist)

        # Compute atomic number embeddings
        source_element = element[edge_index[0]]
        target_element = element[edge_index[1]]
        source_embedding = self.source_embedding(source_element)
        target_embedding = self.target_embedding(target_element)

        # Compute invariant edge embedding
        x_edge = self.act(source_embedding + target_embedding + x_dist)
        return self.act(self.fc1_edge_attr(x_edge))
