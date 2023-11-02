from envs import (
    BasicMinecraft,
    IntermediateMinecraft,
    AdvancedMinecraft,
    MaskedMinecraft,
)

import numpy as np
import random
import torch

SEED = 63
np.random.seed(SEED)  # random seed for reproducibility
random.seed(SEED)
torch.manual_seed(SEED)


def main():
    env_index = 2  # 0: BasicMinecraft, 1: IntermediateMinecraft, 2: AdvancedMinecraft, 3: MaskedMinecraft
    minecraft = [
        BasicMinecraft,
        IntermediateMinecraft,
        AdvancedMinecraft,
        MaskedMinecraft,
    ][env_index]

    # start&close pal
    env = minecraft(visually=False, start_pal=True, keep_alive=False, debug_pal=True)
    env.reset()
    env.close()


if __name__ == "__main__":
    main()
