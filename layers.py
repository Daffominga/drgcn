from typing import Tuple, Optional, Union
import torch
from torch import Tensor
from torch.nn import Parameter
import torch.nn as nn
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum, mul_
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops

from torch_geometric.utils.num_nodes import maybe_num_nodes

from torch_geometric.nn.inits import glorot, zeros

Adj = Union[Tensor, SparseTensor]
OptTensor = Optional[Tensor]
PairTensor = Tuple[Tensor, Tensor]
OptPairTensor = Tuple[Tensor, Optional[Tensor]]
PairOptTensor = Tuple[Optional[Tensor], Optional[Tensor]]
Size = Optional[Tuple[int, int]]
NoneType = Optional[Tensor]


@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    # type: (Tensor, OptTensor, Optional[int], bool, bool, Optional[int]) -> PairTensor  # noqa
    pass


@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    # type: (SparseTensor, OptTensor, Optional[int], bool, bool, Optional[int]) -> SparseTensor  # noqa
    pass


# get D^(-1/2) A D^(-1/2)
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False, add_self_loops=True, dtype=None):

    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul_(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul_(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)  # return int(index.max()) + 1 if num_nodes is None else

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                     device=edge_index.device)  # torch.Size([10556])

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        # For each value in src, it is added to an index in self which is specified by its index in src for
        # dimension != dim and by the corresponding value in index for dimension = dim.
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        # Fills elements of self tensor with value where mask is True.
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


class resGCNConv(MessagePassing):
    _cached_edge_index: Optional[Tuple[torch.Tensor, torch.Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_channels: int, out_channels: int, improved: bool = False, cached: bool = True,
                 add_self_loops: bool = True, normalize: bool = True, **kwargs):

        super(resGCNConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize
        self.add_self_loops = add_self_loops

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.weight1 = Parameter(torch.Tensor(in_channels, out_channels))
        self.weight2 = Parameter(torch.Tensor(in_channels, out_channels))
        self.epsilon = Parameter(torch.Tensor(1))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight1)
        glorot(self.weight2)
        torch.nn.init.uniform_(self.epsilon, a=0.5, b=0.51)

        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj, alpha, gamma, h0, beta, lx,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        if self.normalize:
            if isinstance(edge_index, Tensor):  # true
                cache = self._cached_edge_index
                if cache is None:  # true and edge_weight is None
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        support = alpha * x + beta * torch.matmul(x, self.weight1)

        reshp = torch.matmul(lx, self.weight2)

        initial = (1 - beta) * h0  # H0 is more important when layers are deep

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=support, edge_weight=edge_weight, size=None) + initial + (beta) * reshp

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        assert edge_weight is not None
        return edge_weight.view(-1, 1) * x_j  # x_j contains the source node features of each edge

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

