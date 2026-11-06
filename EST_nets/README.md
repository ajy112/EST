# Reusable EST Modules

`EST_nets` contains the backbone-independent implementation of the Equivariant Spherical Transformer. Use the core blocks directly when adding EST to an existing equivariant message-passing network, or instantiate the provided reference model.

## Public API

```python
from EST_nets import (
    EdgeEST,
    NodeEST,
    ToS2Grid_block,
    EST_layer,
    EST_message,
    EST_update,
    EquivariantSphericalTransformer,
    est_4layer,
    est_6layer,
)
```

### Core plug-in blocks

| Class | Role | Main input shape |
|---|---|---|
| `ToS2Grid_block` | Irreducible representation ↔ sampled sphere transform | `[N, (lmax + 1)^2, C]` |
| `NodeEST` | Node-update EST branch | `[N, (lmax + 1)^2, C]` |
| `EdgeEST` | Edge-message EST branch | `[E, (lmax + 1)^2, C]` |
| `EST_update` | Dense hybrid of steerable and spatial experts | node features |
| `EST_layer` | Complete EST message + update layer | node and edge features |

The simplest node-update integration is:

```python
grid = ToS2Grid_block(l=2, res=64)
est = NodeEST(lmax=2, channels=128, to_grid=grid, num_experts=1)
x = x + est(x)
```

For the hybrid architecture used in the paper, retain the backbone update and combine it with the EST output using gates computed from invariant (`l = 0`) channels.

## Reference Model Parameters

`EquivariantSphericalTransformer` accepts:

| Parameter | Meaning |
|---|---|
| `embed_dim` | Feature channels per spherical-harmonic component |
| `num_layers` | Number of EST layers |
| `lmax` | Maximum representation degree |
| `number_of_basis` | Radial-basis dimension |
| `basis_type` | `"gaussian"` or `"bessel"` |
| `max_radius` | Neighbor cutoff radius |
| `drop_ratio` | Dropout in the update block |
| `num_elements` | Maximum number of atomic types |
| `grid_resolution` | Number of Fibonacci-lattice samples |
| `message_steerable_nums` | Steerable experts in the message block |
| `message_spherical_nums` | EST experts in the message block |
| `update_steerable_nums` | Steerable experts in the update block |
| `update_spherical_nums` | EST experts in the update block |
| `avg_atoms` | Dataset normalization constant |
| `avg_degree` | Average neighborhood size for message normalization |

Two reference configurations are included:

```python
from EST_nets import est_4layer, est_6layer

small_model = est_4layer()
large_model = est_6layer()
```

`avg_atoms` and `avg_degree` are dataset-dependent and should be recomputed when the dataset or cutoff changes.
