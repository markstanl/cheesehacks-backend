from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import torch
import random
ethics = ["commonsense", "deontology", "justice", "utilitarianism", "virtue"]

device = "cuda" if torch.cuda.is_available() else "cpu"
print("CUDA available:", torch.cuda.is_available())
print("Using device:", device)

model = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)

SCENARIO_PREFIX = "Scenario:"

def build_utilitarian_examples(baseline,less_pleasant):
    # keep labels exactly as-is (just cast to int if needed)
    labels = []
    scenarios = []
    for b, lp in zip(baseline, less_pleasant):

        x = random.randint(0, 1)

        labels.append(x)
        if x == 0:
            s = f"{SCENARIO_PREFIX} Option A: {b} [SEP] Option B: {lp}"
        else:
            s = f"{SCENARIO_PREFIX} Option A: {lp} [SEP] Option B: {b}"

        scenarios.append(s) 
        
    return scenarios, labels

def get_label_list(batch):
    # utilitarianism sometimes uses different label field names depending on dataset config
    candidates = ["label", "labels", "gold", "answer", "target", "y", "choice", "preferred"]
    for k in candidates:
        if k in batch:
            return batch[k]
    raise KeyError(f"No label-like key found in batch. Keys: {list(batch.keys())}")    


def process_batch(batch, ethic_name):
    if ethic_name == "commonsense":
        scenarios = batch["input"]
        labels = batch["label"]

    elif ethic_name == "deontology":
        scenarios = [f"{s} {e}".strip() for s, e in zip(batch["scenario"], batch["excuse"])]
        labels = batch["label"]

    elif ethic_name in ["justice", "virtue"]:
        scenarios = batch["scenario"]
        labels = batch["label"]
    elif ethic_name == "utilitarianism":
        #print(batch.keys())
        
        # assumes utilitarianism split has these columns
        scenarios, labels = build_utilitarian_examples(
            batch["baseline"],       # <--- include scenario text
            batch["less_pleasant"],                  # labels unchanged
        )
    else:
        raise ValueError(f"Unknown ethic: {ethic_name}")

    vecs = model.encode(
        scenarios,
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    return {
        "scenario": scenarios,
        "label": labels,
        "embedding": vecs.tolist(),
    }



for ethic in ethics:
    print(f"\nProcessing: {ethic}")
    ds = load_dataset("./ethics", ethic, trust_remote_code=True)
    print("COLUMNS:", ethic, {k: ds[k].column_names for k in ds.keys()})
    for split in ds.keys():

        
        old_cols = ds[split].column_names
        ds[split] = ds[split].map(
            lambda batch: process_batch(batch, ethic),
            batched=True,
            batch_size=256,
            remove_columns=old_cols,
        )
        print(f"{ethic}/{split} => columns: {ds[split].column_names} | n={len(ds[split])}")

ds.save_to_disk(f"data/ethics_{ethic}_with_embeddings")
print(f"Saved: data/ethics_{ethic}_with_embeddings")