"""Seeds do grupo e set_seed()."""
import random

import numpy as np

SEEDS_CMNIST = list(range(10))      # Colored MNIST: 10 seeds (como o repo oficial)
SEEDS_WATERBIRDS = list(range(5))   # Waterbirds: 5 seeds


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
