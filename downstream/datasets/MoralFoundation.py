import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


class MoralFoundation(Dataset):
    """
    huggingface dataset wrapper for moral foundations classification.
    """

    def __init__(
            self,
            encoder: SentenceTransformer,
            dataset_path: str = 'USC-MOLA-Lab/MFRC',
            split: str = 'train'
    ):
        full_data = load_dataset(dataset_path, split=split)

        self.data = full_data.shuffle(seed=42).select(
            range(int(len(full_data) * 0.1)))

        texts = [str(text) for text in self.data['text']]

        # map string labels to integers
        raw_annotations = [str(a) for a in self.data['annotation']]
        self.unique_labels = sorted(list(set(raw_annotations)))
        self.num_classes = len(self.unique_labels)

        label_to_idx = {label: i for i, label in enumerate(self.unique_labels)}
        indices = [label_to_idx[a] for a in raw_annotations]

        self.embeddings = encoder.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=True,
            batch_size=256
        )
        self.labels = torch.tensor(indices, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param idx: sample index.
        """
        return self.embeddings[idx], self.labels[idx]