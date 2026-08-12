from __future__ import annotations

import argparse

import pandapower as pp
import torch
from pandapower.networks import create_cigre_network_mv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classical", action="store_true")
    args = parser.parse_args()

    print(f"Python/Torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA-enabled PyTorch installation or NVIDIA access was not found"
        )
    properties = torch.cuda.get_device_properties(0)
    vram_gib = properties.total_memory / 2**30
    print(f"GPU: {properties.name}")
    print(f"VRAM GiB: {vram_gib:.2f}")
    if "5060 Ti" not in properties.name or vram_gib < 15:
        raise RuntimeError("Expected RTX 5060 Ti 16 GiB hardware was not found")

    net = create_cigre_network_mv(with_der="all")
    pp.runpp(net, numba=False)
    if not net.converged:
        raise RuntimeError("pandapower CIGRE MV smoke did not converge")
    print("pandapower CIGRE MV smoke: OK")

    if args.classical:
        from aeon.classification.convolution_based import (
            MultiRocketHydraClassifier,
        )

        model = MultiRocketHydraClassifier(random_state=1)
        if model.get_params()["random_state"] != 1:
            raise RuntimeError("MultiRocket-Hydra construct smoke failed")
        print("MultiRocket-Hydra import/construct smoke: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
