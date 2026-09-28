import torch
from transformers import AutoModel, AutoTokenizer

def get_embedding(
    text: str,
    tokenizer,   # Reserved for interface compatibility, not actually used
    model,
    device: str = "cuda"
) -> torch.Tensor:
    """
        Input:
            text: a single string
            tokenizer: a loaded tokenizer
            model: a loaded model
            device: torch.device, optional
        Output:
            embedding: a torch.Tensor with shape [d]
    """
    with torch.no_grad():
        embeddings = model.encode_text(
            texts=[text],
            task="retrieval",
            prompt_name="query",
            max_length=8192
        )

    # encode_text returns List[Tensor], take the first one
    embedding = embeddings[0].to(device)

    return embedding

if __name__ == '__main__':
    text='test'
    model_path='jinaai/jina-embeddings-v4'
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
    model = AutoModel.from_pretrained(model_path, device_map="cuda", trust_remote_code=True)
    result=get_embedding(text,tokenizer,model)
    print(result)