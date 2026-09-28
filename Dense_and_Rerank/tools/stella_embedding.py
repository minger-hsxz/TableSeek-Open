import os
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import normalize

def get_embedding(text, tokenizer, model, vector_linear=None, max_length=32768, normalize_output=True, device="cuda"):
    """
    Get embedding vector for a single text.

    Args:
        text: Must be a single text string (not a list)
        tokenizer: Tokenizer
        model: Pretrained model
        vector_linear: Optional linear layer for dimensionality reduction (not used by default)
        max_length: Maximum sequence length
        normalize_output: Whether to normalize the output
        device: Device type

    Returns:
        Embedding vector of the text, 1D tensor of shape (d,)
    """
    # Validate that input is a single text
    if not isinstance(text, str):
        raise TypeError(f"text parameter must be a single string, but received {type(text)}")

    # Move model to specified device
    model = model.to(device)
    if vector_linear is not None:
        vector_linear = vector_linear.to(device)

    with torch.no_grad():
        # Tokenize
        input_data = tokenizer(
            text,
            padding="longest",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        input_data = {k: v.to(device) for k, v in input_data.items()}

        # Get model output
        attention_mask = input_data["attention_mask"]
        last_hidden_state = model(**input_data)[0]

        # Mean pooling (mask out padding tokens)
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        embedding = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

        # Apply linear layer for dimensionality reduction if provided
        if vector_linear is not None:
            embedding = vector_linear(embedding)

        # Optional: normalize
        if normalize_output:
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

        # Squeeze dimensions to ensure shape [d]
        embedding = embedding.squeeze(0)

    return embedding

if __name__ == "__main__":
    text = 'test'
    model_path = 'QwenCollection/stella_en_1.5B_v5'
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left', trust_remote_code=True,
                                              local_files_only=True)
    model = AutoModel.from_pretrained(model_path, device_map="cuda", trust_remote_code=True, local_files_only=True)
    result = get_embedding(text, tokenizer, model)
    print(result)