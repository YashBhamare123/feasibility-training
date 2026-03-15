import numpy as np
import torch
import torch.nn.functional as F
import tiktoken
from dataclasses import dataclass

from model import GPT


class GPT2Inference:
    """ To generate text sequences using a trained GPT2 model """

    def __init__(self, model, token_encoder, device):
        self.model = model
        self.token_encoder = token_encoder
        self.device = device
        self.device_type = 'cuda' if device.startswith('cuda') else 'cpu'

    def generate_sequences(self, prompt, num_seq=5, max_tokens=50, temperature=1.0, top_k=50):
        self.model.eval()
        tokens = self.token_encoder.encode(prompt)
        tokens = torch.tensor(tokens, dtype=torch.long)    # (n,)   n : current sequence length
        tokens = tokens.unsqueeze(0).repeat(num_seq, 1)    # (1,n) --> (num_seq, n)
        gen_tokens = tokens.to(self.device)
        # create a different rng generator so as not to impact the global rng state used for training
        sample_rng = torch.Generator(device=self.device).manual_seed(42)

        # generate new tokens one token at a time until the sequence length becomes 'max_tokens'
        while gen_tokens.shape[-1] <= max_tokens:
            with torch.no_grad():
                with torch.autocast(device_type=self.device_type, dtype=torch.float16):
                    logits, loss = self.model(gen_tokens)    # (num_seq, n, vocab_size)
                logits = logits[:, -1, :]    # (num_seq, vocab_size)
                
                # apply temperature scaling
                logits = logits / max(temperature, 1e-5)
                
                probs = F.softmax(logits, dim=-1)    # (num_seq, vocab_size)
                
                # take top-k probs
                topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)    # (num_seq, top_k), (num_seq, top_k)
                # sample a token from top-k probabilities
                ix = torch.multinomial(topk_probs, num_samples=1, generator=sample_rng)    # (num_seq, 1)
                next_tok = torch.gather(topk_indices, -1, ix)    # (num_seq, 1)
                gen_tokens = torch.cat([gen_tokens, next_tok], dim=1)
        # decode generated tokens and print generated text
        for i in range(num_seq):
            tokens = gen_tokens[i, :max_tokens].tolist()
            gen_text = self.token_encoder.decode(tokens)
            print(f"> sample {i}: {gen_text}")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default="Hello, I am a language model,")
    parser.add_argument('--num_seq', type=int, default=5)
    parser.add_argument('--max_tokens', type=int, default=50)
    parser.add_argument('--temperature', type=float, default=1.0, help="Higher values = more random, lower values = more deterministic")
    parser.add_argument('--top_k', type=int, default=50, help="Number of highest probability vocabulary tokens to keep for sampling")
    args = parser.parse_args()
    return args


@dataclass
class GPTConfig:
    context_length: int = 1024    # max context / sequence length
    vocab_size: int = 50257    # number of tokens: 50000 BPE merges + 256 bytes tokens + 1 <endoftext> token
    num_layers: int = 12
    embd_size: int = 768    # embedding dim
    num_heads: int = 12


def inference(args=None):
    if args is None:
        args = parse_args()

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'    # for apple macbook GPUs
    print(f'using device: {device}')

    model_path = './logs/model_95364.pt'
    checkpoint = torch.load(model_path, weights_only=False)
    print(f"loaded model from: {model_path}")
    # print(checkpoint['model'].keys())

    # fix the state_dict keys if the model was compiled using torch.compile()
    unwanted_prefix = '_orig_mod.'
    state_dict = checkpoint['model']
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model = GPT(config=checkpoint['config'])
    model.load_state_dict(state_dict)
    model = model.to(device)
    token_encoder = tiktoken.get_encoding('gpt2')
    generator = GPT2Inference(model, token_encoder, device)

    generator.generate_sequences(
        args.prompt, 
        args.num_seq, 
        args.max_tokens,
        args.temperature,
        args.top_k
    )

if __name__ == '__main__':
    inference()
