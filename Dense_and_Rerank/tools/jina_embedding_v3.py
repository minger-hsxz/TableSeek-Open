import torch
from transformers import AutoModel
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

def get_embedding(text: str, tokenizer=None, model=None) -> torch.Tensor:
    """
        Input:
            text: a single string
            tokenizer: a loaded tokenizer
            model: a loaded model
            device: torch.device, optional
        Output:
            embedding: a torch.Tensor with shape [d]
    """
    if model is None:
        raise ValueError("Please provide the loaded model")

    embedding = model.encode([text], task="text-matching")  # shape=[1, d]
    return embedding[0]  # shape=[d]


if __name__ == '__main__':
    model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True, local_files_only=True)
    text = "A test text"
    embedding = get_embedding(text, tokenizer=None, model=model)
    print(embedding.shape)
