MODEL_CONFIGS = {
    "micro": {
        "d_model": 32,
        "num_layers": 2,
        "num_heads": 4,
        "ff_hidden_size": 128,
        "block_size": 64,
        "dropout": 0.1,
    },
    "small": {
        "d_model": 64,
        "num_layers": 4,
        "num_heads": 8,
        "ff_hidden_size": 256,
        "block_size": 128,
        "dropout": 0.1,
    },
    "base": {
        "d_model": 128,
        "num_layers": 6,
        "num_heads": 8,
        "ff_hidden_size": 512,
        "block_size": 256,
        "dropout": 0.1,
    },
}
