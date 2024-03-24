import torch
from time import time


def benchmark_model(model, input_ids, attention_mask, iterations=100):
    start_time = time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
    end_time = time()
    return (end_time - start_time) / iterations
