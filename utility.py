import random, torch, os, numpy as np

def set_random_seed(seed: int = 42):
    """
    固定全局随机数种子，确保实验可复现。

    Args:
        seed (int): 随机数种子，默认42。
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False