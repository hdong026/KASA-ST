import os
import sys

import torch

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from basicts.archs import TFSTGN
from examples.TFSTGN.TFSTGN_PEMS04_full import build_model_param


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TFSTGN(**build_model_param()).to(device)

    x = torch.randn(2, 12, 307, 4, device=device)
    y = model(x, None, 0, 0, False)
    print("prediction:", y.shape)
    assert y.shape == (2, 12, 307, 1)

    out = model(x, None, 0, 0, False, return_intermediates=True)
    assert isinstance(out, dict)
    assert out["prediction"].shape == (2, 12, 307, 1)
    assert out["y_temporal"].shape == (2, 12, 307, 1)
    assert out["delta_spatial"].shape == (2, 12, 307, 1)
    print("params:", count_params(model))
    print("spatial_alpha:", out["spatial_alpha"].item())

    model_no_tf = TFSTGN(num_nodes=307, input_len=12, output_len=12, input_dim=4, use_tf_spatial=False).to(device)
    y2 = model_no_tf(x, None, 0, 0, False)
    assert y2.shape == (2, 12, 307, 1)
    print("TFSTGN smoke test passed.")


if __name__ == "__main__":
    main()
