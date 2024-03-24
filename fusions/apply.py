def apply(traced_model, fusion_types):
    for fusion_type in fusion_types:
        if fusion_type == "relu":
            from fusions import relu_fuse

            traced_model = relu_fuse(traced_model)
        elif fusion_type == "linear_gelu":
            from fusions import linear_gelu_fuse

            traced_model = linear_gelu_fuse(traced_model)
        elif fusion_type == "batchnorm_relu":
            from fusions import batchnorm_relu_fuse

            traced_model = batchnorm_relu_fuse(traced_model)
        else:
            raise ValueError(f"Unsupported fusion type: {fusion_type}")
    return traced_model
