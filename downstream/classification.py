import torch
from downstream.model import CoolProjectionHead
from huggingface_hub import hf_hub_download

from model import SharedEncoderBinaryHeads

def load_mlp_model(checkpoint_path: str = None) -> SharedEncoderBinaryHeads:
    """
    Load the MLP model from a checkpoint path.
    """
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model = SharedEncoderBinaryHeads(input_dim=checkpoint['input_dim'], latent_dim=checkpoint['latent_dim'], tasks=checkpoint['tasks'])
        model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model

model_path = hf_hub_download(
    repo_id="Praneet-P/ethics-multihead-model",
    filename="checkpoints/shared_encoder_heads.pt"
)
embedding_model = load_mlp_model(model_path)

def load_classification_model(dataset_name) -> CoolProjectionHead:
    model_path = hf_hub_download(
        repo_id="Praneet-P/ethics-multihead-model",
        filename=f"checkpoints/downstream/{dataset_name}.pt"
    )
    ckpt = torch.load(model_path, map_location="cpu")
    model = CoolProjectionHead(encoder=embedding_model, in_features=ckpt['in_features'], out_classes=ckpt['out_classes'])
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model


@torch.no_grad()
def _classify_pv(model, personality_vector, labels):
    logits = bigfive(personality_vector)
    label_idx = torch.argmax(logits, dim=-1).item()
    return starsign_labels[label_idx]

bigfive = load_classification_model('bigfive')
briggs = load_classification_model('briggs')
moralfoundation = load_classification_model('moralfoundation')
politicalleaning = load_classification_model('politicalleaning')
starsign = load_classification_model('starsign')


def _classify_pv(personality_vector):
    big_five = _classify_pv(bigfive, personality_vector, )
