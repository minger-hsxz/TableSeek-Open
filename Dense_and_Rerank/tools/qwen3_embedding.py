import torch
from torch import Tensor
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_embedding(text: str, tokenizer, model, max_length: int = 32768) -> torch.Tensor:
    batch_dict = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**batch_dict)
    emb = last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
    emb = F.normalize(emb, p=2, dim=1)
    return emb.squeeze(0)   # shape [d]

if __name__ == '__main__':
    text="This passage summarizes the core technical contributions of the AR1 model, as well as future research directions, including enhancing consistency in reasoning and actions, improving the model's adaptability and reasoning efficiency, and introducing auxiliary tasks and world models, in order to further enhance the performance and robustness of autonomous driving systems."
    model_path='Qwen/Qwen3-Embedding-0.6B'
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
    model = AutoModel.from_pretrained(model_path, device_map="cuda")
    result=get_embedding(text,tokenizer,model)
    print(result)

    text = 'Overall, this passage emphasizes the efficiency, accuracy, and cost advantages of Falcon-H1R in inference tasks, and looks forward to the possibility of further improving inference performance through optimizing small models and architectures in the future.'
    result = get_embedding(text, tokenizer, model)
    print(result)

