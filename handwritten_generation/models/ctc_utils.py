import numpy as np
import torch


def _merge_labels(labels: list) -> list:
    result = []
    prev_label = None

    for label in labels:
        if label == prev_label:
            continue

        result.append(label)
        prev_label = label

    return result


def _remove_blank(labels: list, blank: int = 0) -> list:
    return [label for label in labels if label != blank]


def ctc_decode(log_probas_sequences: torch.Tensor, blank: int = 0) -> list:
    labels_sequences = (
        log_probas_sequences.detach().cpu().transpose(0, 1).argmax(dim=-1).numpy()
    )
    decoded_labels_sequences = []

    for labels_sequence in labels_sequences:
        decoded_labels_sequence = _merge_labels(labels_sequence)
        decoded_labels_sequence = _remove_blank(decoded_labels_sequence)
        decoded_labels_sequences.append(decoded_labels_sequence)

    return decoded_labels_sequences
