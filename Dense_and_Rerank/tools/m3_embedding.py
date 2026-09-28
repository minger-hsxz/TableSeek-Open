import torch
from typing import Optional
from FlagEmbedding import BGEM3FlagModel

def get_embedding(text: str, tokenizer=None, model=None, device: Optional[str] = None) -> torch.Tensor:
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
        raise ValueError("The model parameter cannot be empty")

    # Encode a single text using the model
    # The BGEM3FlagModel.encode method can accept a single string or a list of strings
    result = model.encode([text], batch_size=1, max_length=8192)

    # Obtain dense vector representations and convert them to Tensor
    # Note: encode returns a numpy array, which needs to be converted to a torch Tensor
    embedding_tensor = torch.from_numpy(result['dense_vecs'][0])

    return embedding_tensor


# Example usage
if __name__ == "__main__":
    model = BGEM3FlagModel('BAAI/bge-m3', device_map="cuda")
    text = "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction."
    embedding = get_embedding(text=text, model=model)
    print(embedding)