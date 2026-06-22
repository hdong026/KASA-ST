"""Resolve D2STGNN input/output sequence lengths for decoupled H vs F."""


def normalize_d2stgnn_seq_lens(model_args: dict) -> tuple[int, int]:
    """Return (input_seq_len, output_seq_len) and write normalized keys into model_args."""
    legacy = int(model_args.get("seq_length", 12))
    input_len = int(model_args.get("input_seq_len", legacy))
    output_len = int(model_args.get("output_seq_len", legacy))
    model_args["input_seq_len"] = input_len
    model_args["output_seq_len"] = output_len
    # Legacy key used by forecast heads; keep aligned with output horizon.
    model_args["seq_length"] = output_len
    return input_len, output_len
