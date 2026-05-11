"""Check PyTorch CUDA visibility for the active Python environment."""

from __future__ import annotations

import torch


def main() -> None:
    print("torch:", torch.__version__)
    print("torch cuda version:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    print("device count:", torch.cuda.device_count())

    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
        x = torch.randn(4, device="cuda")
        print("cuda tensor ok:", x.device, float(x.sum()))


if __name__ == "__main__":
    main()
