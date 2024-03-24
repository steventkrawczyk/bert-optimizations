import argparse
from transformers import BertModel
from transformers.utils.fx import symbolic_trace
import torch

from benchmark import benchmark_model
from fusions.apply import apply


def main(model_name: str, fusion_types: list[str], mode: str):
    # Load and trace the model
    model = BertModel.from_pretrained(model_name)
    traced = symbolic_trace(
        model,
        input_names=["input_ids", "attention_mask"],
    )

    # Apply the optimizations specified by the fusion_types
    optimized_model = apply(traced, fusion_types)

    # Create sample inputes to use, then either verify or benchmark the model
    input_ids = torch.randint(0, 30522, (1, 512))
    attention_mask = torch.ones(1, 512)

    if mode == "verify":
        original_output = model(input_ids=input_ids, attention_mask=attention_mask)
        optimized_output = optimized_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        assert torch.allclose(
            original_output["last_hidden_state"],
            optimized_output["last_hidden_state"],
            atol=1e-6,
        )
        print(
            "Verification Successful: Optimized output is close to the original output."
        )
    elif mode == "benchmark":
        original_time = benchmark_model(model, input_ids, attention_mask)
        optimized_time = benchmark_model(optimized_model, input_ids, attention_mask)
        print(f"Original Model Time: {original_time:.6f} sec/iteration")
        print(f"Optimized Model Time: {optimized_time:.6f} sec/iteration")
        speedup = (original_time - optimized_time) / original_time
        if speedup > 0:
            print(f"That's a {speedup:.2f}% speedup!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Select the fusion method for the BERT model optimization and choose between verifying or benchmarking the optimization."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-uncased",
        help="Name of the model to pull from huggingface to perform optimizations, verification and benchmarking",
    )
    parser.add_argument(
        "--fusion",
        type=str,
        nargs="+",
        default=["relu"],
        choices=["relu", "linear_gelu", "batchnorm_relu"],
        help="Types of fusion to apply. This can be specified multiple times with different values. Examples: --fusion relu linear_gelu.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="verify",
        choices=["verify", "benchmark"],
        help="Operation mode: 'verify' to check the correctness of the optimization or 'benchmark' to measure the performance. Default is 'verify'.",
    )
    args = parser.parse_args()
    main(args.model_name, args.fusion, args.mode)
