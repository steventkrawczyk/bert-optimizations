# BERT Optimization with `torch.fx`

This repo provides a simple tool for applying `torch.fx` optimizations to BERT models from the huggingface `transformers` library.

## Example Usage

```
python transform.py --model_name=bert-base-uncased --fusion linear_gelu relu --mode=benchmark
Original Model Time: 1.142008 sec/iteration
Optimized Model Time: 1.073801 sec/iteration
That's a 0.06% speedup!
```

## Supported Optimizations 

* `relu`: Combine two sequential RELU nodes into a single node.
* `batchnorm_relu`: Combine a BatchNorm and RELU node into a single node.
* `linear_gelu`: Combine a Linear and GELU node into a single node.

By providing optimizations as a space separated list, you can apply them in sequences (e.g. `--fusion linear_gelu relu`)

## Supported Run Modes

* `verify`: Check that the outcomes before and after optimizations are sufficiently close
* `benchmark`: Run 100 iterations and compare the average runtime for the base and optimized models

## Supported Models

This tool uses the [BertModel](https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertModel) from huggingface and supports all models implementing this class. 