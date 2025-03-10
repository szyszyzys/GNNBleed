from math import sqrt

import graphviz
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import torch
from torch import Tensor

from torch_geometric.explain import Explainer, GNNExplainer, PGExplainer
from torch_geometric.explain.explanation import Explanation
import matplotlib.pyplot as plt
from torch_geometric.explain.config import (ExplanationType,
                                            MaskType,
                                            ModelConfig,
                                            ModelMode,
                                            ModelReturnType,
                                            ThresholdConfig,
                                            )


def get_explainer(model, explain_algo="GNNExplainer"):
    model_config = ModelConfig(
        mode='multiclass_classification',
        task_level='node',
        return_type='probs',
    )
    algo = GNNExplainer()
    if explain_algo == "GNNExplainer":
        algo = GNNExplainer()
    elif explain_algo == "PGExplainer":
        algo = PGExplainer()
    explainer = Explainer(
        model=model,
        algorithm=algo,
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=model_config,
        threshold_config=None,
    )
    return explainer


def _visualize_graph_via_graphviz(
        edge_index: Tensor,
        edge_weight: Tensor,
        path=None,
):
    suffix = path.split('.')[-1] if path is not None else None
    g = graphviz.Digraph('graph', format=suffix)
    g.attr('node', shape='circle', fontsize='11pt')

    for node in edge_index.view(-1).unique().tolist():
        g.node(str(node))

    for (src, dst), w in zip(edge_index.t().tolist(), edge_weight.tolist()):
        hex_color = hex(255 - round(255 * w))[2:]
        hex_color = f'{hex_color}0' if len(hex_color) == 1 else hex_color
        g.edge(str(src), str(dst), color=f'#{hex_color}{hex_color}{hex_color}')

    if path is not None:
        path = '.'.join(path.split('.')[:-1])
        g.render(path, cleanup=True)
        g.view()
    else:
        g.view()

    return g


def _visualize_graph_via_networkx(
        edge_index: Tensor,
        edge_weight: Tensor,
        path=None,
):
    g = nx.DiGraph()
    node_size = 800

    for node in edge_index.view(-1).unique().tolist():
        g.add_node(node)

    for (src, dst), w in zip(edge_index.t().tolist(), edge_weight.tolist()):
        g.add_edge(src, dst, alpha=w)

    ax = plt.gca()
    pos = nx.spring_layout(g)
    for src, dst, data in g.edges(data=True):
        ax.annotate(
            '',
            xy=pos[src],
            xytext=pos[dst],
            arrowprops=dict(
                arrowstyle="->",
                alpha=data['alpha'],
                shrinkA=sqrt(node_size) / 2.0,
                shrinkB=sqrt(node_size) / 2.0,
                connectionstyle="arc3,rad=0.1",
            ),
        )

    nodes = nx.draw_networkx_nodes(g, pos, node_size=node_size,
                                   node_color='white', margins=0.1)
    nodes.set_edgecolor('black')
    nx.draw_networkx_labels(g, pos, font_size=10)

    if path is not None:
        plt.savefig(path)
        plt.show()

    else:
        plt.show()

    plt.close()


def has_graphviz() -> bool:
    try:
        import graphviz
    except ImportError:
        return False

    try:
        graphviz.Digraph().pipe()
    except graphviz.backend.ExecutableNotFound:
        return False

    return True


def visualize_graph_process(
        edge_index: Tensor,
        edge_weight=None,
        path=None,
        backend='networkx',
):
    r"""Visualizes the graph given via :obj:`edge_index` and (optional)
    :obj:`edge_weight`.
    Args:
        edge_index (torch.Tensor): The edge indices.
        edge_weight (torch.Tensor, optional): The edge weights.
        path (str, optional): The path to where the plot is saved.
            If set to :obj:`None`, will visualize the plot on-the-fly.
            (default: :obj:`None`)
        backend (str, optional): The graph drawing backend to use for
            visualization (:obj:`"graphviz"`, :obj:`"networkx"`).
            If set to :obj:`None`, will use the most appropriate
            visualization backend based on available system packages.
            (default: :obj:`None`)
    """
    BACKENDS = {'graphviz', 'networkx'}
    if edge_weight is not None:  # Normalize edge weights.
        edge_weight = edge_weight - edge_weight.min()
        edge_weight = edge_weight / edge_weight.max()

    if edge_weight is not None:  # Discard any edges with zero edge weight:
        mask = edge_weight > 0.5
        edge_index = edge_index[:, mask]
        edge_weight = edge_weight[mask]

    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1))

    if backend is None:
        backend = 'graphviz' if has_graphviz() else 'networkx'

    if backend.lower() == 'networkx':
        return _visualize_graph_via_networkx(edge_index, edge_weight, path)
    elif backend.lower() == 'graphviz':
        return _visualize_graph_via_graphviz(edge_index, edge_weight, path)

    raise ValueError(f"Expected graph drawing backend to be in "
                     f"{BACKENDS} (got '{backend}')")


def visualize_feature_importance(
        explanation,
        path=None,
        feat_labels=None,
        top_k=None,
):
    r"""Creates a bar plot of the node features importance by summing up
    :attr:`self.node_mask` across all nodes.
    Args:
        path (str, optional): The path to where the plot is saved.
            If set to :obj:`None`, will visualize the plot on-the-fly.
            (default: :obj:`None`)
        feat_labels (List[str], optional): Optional labels for features.
            (default :obj:`None`)
        top_k (int, optional): Top k features to plot. If :obj:`None`
            plots all features. (default: :obj:`None`)
    """

    node_mask = explanation.get('node_mask')
    if node_mask is None:
        raise ValueError(f"The attribute 'node_mask' is not available "
                         f"in '{explanation.__class__.__name__}' "
                         f"(got {explanation.available_explanations})")
    if node_mask.dim() != 2 or node_mask.size(1) <= 1:
        raise ValueError(f"Cannot compute feature importance for "
                         f"object-level 'node_mask' "
                         f"(got shape {node_mask.size()})")

    feat_importance = node_mask.sum(dim=0).cpu().numpy()

    if feat_labels is None:
        feat_labels = range(feat_importance.shape[0])

    if len(feat_labels) != feat_importance.shape[0]:
        raise ValueError(f"The '{explanation.__class__.__name__}' object holds "
                         f"{feat_importance.numel()} features, but "
                         f"only {len(feat_labels)} were passed")

    df = pd.DataFrame({'feat_importance': feat_importance},
                      index=feat_labels)
    df = df.sort_values("feat_importance", ascending=False)
    df = df.round(decimals=3)

    if top_k is not None:
        df = df.head(top_k)
        title = f"Feature importance for top {len(df)} features"
    else:
        title = f"Feature importance for {len(df)} features"

    ax = df.plot(
        kind='barh',
        figsize=(10, 7),
        title=title,
        xlabel='Feature label',
        xlim=[0, float(feat_importance.max()) + 0.3],
        legend=False,
    )
    plt.gca().invert_yaxis()
    ax.bar_label(container=ax.containers[0], label_type='edge')

    if path is not None:
        plt.savefig(path)
        plt.show()
    else:
        plt.show()

    plt.close()


def visualize_graph(explanation, path: Optional[str] = None,
                    backend: Optional[str] = None):
    r"""Visualizes the explanation graph with edge opacity corresponding to
    edge importance.
    Args:
        path (str, optional): The path to where the plot is saved.
            If set to :obj:`None`, will visualize the plot on-the-fly.
            (default: :obj:`None`)
        backend (str, optional): The graph drawing backend to use for
            visualization (:obj:`"graphviz"`, :obj:`"networkx"`).
            If set to :obj:`None`, will use the most appropriate
            visualization backend based on available system packages.
            (default: :obj:`None`)
    """
    edge_mask = explanation.get('edge_mask')
    if edge_mask is None:
        raise ValueError(f"The attribute 'edge_mask' is not available "
                         f"in '{explanation.__class__.__name__}' "
                         f"(got {explanation.available_explanations})")
    visualize_graph_process(explanation.edge_index, edge_mask, path, backend)
