import torch
import torch.nn.functional as F


def fuse(graph_module: torch.nn.Module) -> torch.nn.Module:
    """
    Identifies all sequences of nodes where a GELU follows a Linear operation, and
    replaces them with the new Linear+GELU node defined below.
    """
    graph = graph_module.graph
    for node in list(graph.nodes):
        if node.op == "call_function" and node.target == torch.nn.functional.linear:
            next_node = next(node.users.keys().__iter__(), None)
            if (
                next_node
                and next_node.op == "call_function"
                and next_node.target == torch.nn.functional.gelu
            ):
                with graph.inserting_before(next_node):
                    fused_node = graph.call_function(
                        fused_linear_gelu,
                        args=(node.args[0], node.args[1], node.args[2]),
                    )
                next_node.replace_all_uses_with(fused_node)
                graph.erase_node(node)
                graph.erase_node(next_node)
    graph.lint()
    return graph_module


def fused_linear_gelu(input, weight, bias=None):
    """
    A node representing the linear and gelu opertions performed in sequence.
    """
    linear_output = F.linear(input, weight, bias)
    return F.gelu(linear_output)
