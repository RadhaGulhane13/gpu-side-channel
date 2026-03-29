import os
import torch
from tqdm import tqdm
from torch.utils.data import IterableDataset, DataLoader

DATA_DIR = "./router_dataset"
BATCH_SIZE = 4096



import os
import torch
from torch.utils.data import IterableDataset, DataLoader

class ExpertDatasetSeq(IterableDataset):
    def __init__(self, data_dir, seq_len=32):
        """
        data_dir: directory containing shard_*.pt files
        seq_len: length of sequences to produce
        """
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.shards = sorted(
            f for f in os.listdir(data_dir)
            if f.startswith("shard_") and f.endswith(".pt")
        )

        # Precompute total number of tokens (optional)
        self._len = 0
        for shard_file in self.shards:
            path = os.path.join(data_dir, shard_file)
            shard = torch.load(path, map_location="cpu")
            self._len += shard["input_tokens"].size(0)

    def __len__(self):
        return self._len

    def __iter__(self):
        for shard_file in self.shards:
            path = os.path.join(self.data_dir, shard_file)
            shard = torch.load(path, map_location="cpu")

            labels = shard["pred_tokens"]           # [N]
            expert_logits = shard["expert_logits"]  # [N, L, E]  full logits over all experts

            N = labels.size(0)
            seq_len = self.seq_len

            for i in range(0, N, seq_len):
                if i + seq_len > N:
                    break

                yield {
                    "labels":         labels[i:i+seq_len],                      # [S]
                    "expert_logits":  expert_logits[i:i+seq_len].float(),       # [S, L, E]
                    "attention_mask": torch.ones(seq_len, dtype=torch.bool),    # [S]
                }


def get_dataloader_seq(data_dir, seq_len=32, batch_size=32, num_workers=4):
    dataset = ExpertDatasetSeq(data_dir, seq_len=seq_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True,
    )


class ExpertDatasetSeqShards(IterableDataset):
    """Like ExpertDatasetSeq but operates on an explicit list of shard filenames."""

    def __init__(self, data_dir, shard_files, seq_len=32):
        self.data_dir = data_dir
        self.shards = shard_files
        self.seq_len = seq_len

    def __iter__(self):
        for shard_file in self.shards:
            path = os.path.join(self.data_dir, shard_file)
            shard = torch.load(path, map_location="cpu")
            labels = shard["pred_tokens"]
            expert_logits = shard["expert_logits"]
            N = labels.size(0)
            for i in range(0, N, self.seq_len):
                if i + self.seq_len > N:
                    break
                yield {
                    "labels": labels[i : i + self.seq_len],
                    "expert_logits": expert_logits[i : i + self.seq_len].float(),
                    "attention_mask": torch.ones(self.seq_len, dtype=torch.bool),
                }


def get_train_val_dataloaders(
    data_dir, seq_len=32, batch_size=32, num_workers=4, val_shards=2
):
    all_shards = sorted(
        f for f in os.listdir(data_dir) if f.startswith("shard_") and f.endswith(".pt")
    )
    if len(all_shards) <= val_shards:
        raise ValueError(
            f"Only {len(all_shards)} shards found; cannot hold out {val_shards} for validation."
        )
    train_shard_files = all_shards[:-val_shards]
    val_shard_files = all_shards[-val_shards:]

    train_loader = DataLoader(
        ExpertDatasetSeqShards(data_dir, train_shard_files, seq_len=seq_len),
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        ExpertDatasetSeqShards(data_dir, val_shard_files, seq_len=seq_len),
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader

class ExpertDataset(IterableDataset):

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.shards = sorted(
            f for f in os.listdir(data_dir)
            if f.startswith("shard_") and f.endswith(".pt")
        )

        # Precompute total number of samples
        self._len = 0
        for shard_file in self.shards:
            path = os.path.join(data_dir, shard_file)
            shard = torch.load(path, map_location="cpu")
            self._len += shard["input_tokens"].size(0)  # number of samples in shard

    def __len__(self):
        return self._len

    def __iter__(self):

        shards = sorted(
            f for f in os.listdir(self.data_dir)
            if f.startswith("shard_") and f.endswith(".pt")
        )

        for shard_file in shards:

            path = os.path.join(self.data_dir, shard_file)
            shard = torch.load(path, map_location="cpu")

            input_tokens = shard["input_tokens"]
            pred_tokens = shard["pred_tokens"]
            expert_idx = shard["expert_idx"]
            expert_scores = shard["expert_scores"]

            for i in range(len(input_tokens)):
                yield {
                    "token": input_tokens[i],
                    "pred": pred_tokens[i],
                    "expert_idx": expert_idx[i],
                    "expert_scores": expert_scores[i],
                }


def get_dataloader(data_dir, batch_size = None):

    dataset = ExpertDataset(data_dir)
    if batch_size is None:
        batch_size = BATCH_SIZE

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        persistent_workers=True
    )


def main():

    loader = get_dataloader(DATA_DIR)

    for batch in tqdm(loader):
        print(batch.keys())

        tokens = batch["token"]
        preds = batch["pred"]
        idxs = batch["expert_idx"]
        scores = batch["expert_scores"]

        # Example processing
        print(f"preds.shape: {preds.shape}, idxs.shape: {idxs.shape}, scores.shape: {scores.shape}")
        print(preds[0], idxs[0], scores[0])

        break


if __name__ == "__main__":
    main()