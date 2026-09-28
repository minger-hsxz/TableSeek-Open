import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

def format_instruction(instruction, query, doc):
    """Format the input text."""
    if instruction is None:
        instruction = 'Given a TableQA query, retrieve a relevant table that can answer the query. Note that the retrieved table should contain sufficient information to provide an answer, rather than resulting in an empty answer. '
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

def process_inputs(model, tokenizer, pairs, prefix_tokens, suffix_tokens, max_length):
    """Convert text pairs to model inputs."""
    inputs = tokenizer(
        pairs, padding=False, truncation='longest_first',
        return_attention_mask=False, max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
    )
    for i, ele in enumerate(inputs['input_ids']):
        inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
    inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
    for key in inputs:
        inputs[key] = inputs[key].to(model.device)
    return inputs

@torch.no_grad()
def compute_logits(model, inputs, token_true_id, token_false_id):
    """Compute yes/no logits scores."""
    batch_scores = model(**inputs).logits[:, -1, :]
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    batch_scores = torch.stack([false_vector, true_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    scores = batch_scores[:, 1].exp().tolist()  # probability of "yes"
    return scores

def rerank(model, tokenizer, task, query, document, max_length=32768):
    """Main function: compute query-doc relevance score."""
    prefix = (
        "<|im_start|>system\n"
        "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
        'Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

    token_false_id = tokenizer.convert_tokens_to_ids("no")
    token_true_id = tokenizer.convert_tokens_to_ids("yes")

    # Build input text
    pair = format_instruction(task, query, document)

    # Preprocess inputs
    inputs = process_inputs(model, tokenizer, [pair], prefix_tokens, suffix_tokens, max_length)

    # Get score
    score = compute_logits(model, inputs, token_true_id, token_false_id)[0]
    return score

if __name__ == '__main__':
    query="This passage summarizes the core technical contributions of the AR1 model, as well as future research directions, including enhancing consistency in reasoning and actions, improving the model's adaptability and reasoning efficiency, and introducing auxiliary tasks and world models, in order to further enhance the performance and robustness of autonomous driving systems."
    task='Given a TableQA query, retrieve a relevant table that can answer the query. Note that the retrieved table should contain sufficient information to provide an answer, rather than resulting in an empty answer. '
    document='AR1'
    model_path='Qwen/Qwen3-Reranker-0.6B'
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda")
    result=rerank(model, tokenizer, task, query, document)
    print(result)