from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

def get_embedding(text: str, tokenizer, model, device=None):
    """
        Input:
            text: a single string
            tokenizer: a loaded tokenizer
            model: a loaded model
            device: torch.device, optional
        Output:
            embedding: a torch.Tensor with shape [d]
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    encoded_input = tokenizer(
        text,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=8192
    ).to(device)

    with torch.no_grad():
        model_output = model(**encoded_input)
        # CLS pooling
        embedding = model_output.last_hidden_state[:, 0]  # shape: [1, d]
        # L2 normalize
        embedding = F.normalize(embedding, p=2, dim=1)

    return embedding.squeeze(0)  # shape: [d]

if __name__ == '__main__':
    text='test'
    model_path = 'BAAI/bge-large-en-v1.5'
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left', trust_remote_code=True, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, device_map="cuda", trust_remote_code=True, local_files_only=True)
    result=get_embedding(text, tokenizer, model)
    print(result)