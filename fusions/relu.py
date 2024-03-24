import torch


def fuse(graph_module: torch.nn.Module) -> torch.nn.Module:
    """
    Identifies all places where two sequential RELU opeartions are used, and replaces
    them with the new Linear+GELU node defined below.
    """
    graph = graph_module.graph
    for node in list(graph.nodes):
        if node.op == "call_function" and node.target == torch.nn.functional.relu:
            next_node = next(node.users.keys().__iter__(), None)
            if (
                next_node
                and next_node.op == "call_function"
                and next_node.target == torch.nn.functional.relu
            ):
                next_node.replace_all_uses_with(node)
                graph.erase_node(next_node)
    graph.lint()
    return graph_module
