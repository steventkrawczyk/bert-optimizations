import torch
import torch.nn.functional as F


def fuse(graph_module: torch.nn.Module) -> torch.nn.Module:
    """
    Identifies all sequences of nodes where a RELU follows a BatchNorm, and replaces them
    with the new BatchNorm+RELU node defined below.
    """
    graph = graph_module.graph
    for node in list(graph.nodes):
        if node.op == "call_function" and node.target == torch.nn.functional.batch_norm:
            next_node = next(node.users.keys().__iter__(), None)
            if (
                next_node
                and next_node.op == "call_function"
                and next_node.target == torch.nn.functional.relu
            ):
                with graph.inserting_before(next_node):
                    fused_node = graph.call_function(
                        fused_batchnorm_relu, args=node.args
                    )
                next_node.replace_all_uses_with(fused_node)
                graph.erase_node(node)
                graph.erase_node(next_node)
    graph.lint()
    return graph_module


def fused_batchnorm_relu(
    input,
    running_mean,
    running_var,
    weight=None,
    bias=None,
    training=False,
    momentum=0.1,
    eps=1e-5,
):
    """
    A node representing the batchnorm and relu opertions performed in sequence.
    """
    normalized = F.batch_norm(
        input, running_mean, running_var, weight, bias, training, momentum, eps
    )
    return F.relu(normalized)
