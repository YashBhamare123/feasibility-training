import os
import math
import numpy as np
import time
from dataclasses import dataclass
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import GPT

# ==========================================
# CONFIGURATION
# ==========================================
@dataclass
class TrainConfig:
    # Batch sizing
    total_batch_size: int = 65536  # Number of tokens processed for each weight update
    mini_batch_size: int = 4       # Mini batch size per forward pass
    context_length: int = 1024     # Max sequence length

    # Model configuration
    num_layers: int = 12
    embd_size: int = 768
    num_heads: int = 12
    vocab_size: int = 50257        # HF standard vocab size

    # Optimization
    max_lr: float = 2e-5           # Max learning rate for fine-tuning
    min_lr: float = 0.0            # Min learning rate
    warmup_steps: int = 100
    weight_decay: float = 0.1
    num_epochs: int = 3
    steps_per_epoch: int = 1000
    eval_freq: int = 200

    # Paths and Misc
    seed: int = 1337
    logdir: str = "./logs_alpaca/"
    use_torch_compile: bool = False

config = TrainConfig()
# ==========================================

# Assuming an AlpacaDataLoader is implemented in dataloader.py or elsewhere
# that handles instruction-tuning format (packing or masking prompt loss)
# from .dataloader import AlpacaDataLoader

# TF32 is an Ampere-specific optimization, Turing T4 doesn't support it natively.
# torch.set_float32_matmul_precision('high')    # enable TF32 precision

class Trainer:
    def __init__(
            self, 
            model, 
            optimizer, 
            train_loader, 
            val_loader, 
            token_encoder, 
            eval_freq, 
            grad_accum_steps, 
            device, 
            logpath
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.token_encoder = token_encoder

        self.eval_freq = eval_freq
        self.grad_accum_steps = grad_accum_steps
        self.device = device
        self.device_type = 'cuda' if device.startswith('cuda') else 'cpu'
        self.logpath = logpath

    def train(
        self, 
        max_steps, 
        warmup_steps, 
        max_lr, 
        min_lr
    ):
        # T4 relies on FP16 which is more prone to underflow/overflow than BF16, so we use a GradScaler
        scaler = torch.amp.GradScaler(device='cuda' if self.device_type == 'cuda' else 'cpu')
        
        for step in range(max_steps):
            t0 = time.time()
            self.is_last_step = (step == max_steps - 1)

            # evaluate validation loss
            if step % self.eval_freq == 0 or self.is_last_step:
                if self.val_loader is not None:
                    self.evaluate_validation(step)

            # generate sequences from the model every once in a while
            if ((step > 0 and step % self.eval_freq == 0) or self.is_last_step) and (not config.use_torch_compile):
                self.generate_sequences(num_seq=2, max_tokens=32)

            # training loop starts here
            self.model.train()    # sets model to train mode
            self.optimizer.zero_grad()    # resets all gradients
            batch_loss = 0.0
            
            for mini_step in range(self.grad_accum_steps):
                inp, tar = self.train_loader.next_batch()
                inp, tar = inp.to(self.device), tar.to(self.device)
                
                # FORWARD PASS
                # autocast to float16 (T4 optimized) for faster compute and memory efficiency
                with torch.autocast(device_type=self.device_type, dtype=torch.float16):
                    logits, loss = self.model(inp, tar)

                loss /= self.grad_accum_steps
                batch_loss += loss.detach()

                # Scale the loss and call backward to create scaled gradients
                scaler.scale(loss).backward()

            # Unscales the gradients of optimizer's assigned params in-place
            scaler.unscale_(self.optimizer)

            # once gradients are computed and unscaled, clip the global l2-norm of the gradient at 1.0
            norm = nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # determine learning rate with decay
            lr = self.estimate_lr(step, warmup_steps, max_steps, max_lr, min_lr)
            # set learning rate for this iteration
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            # optimizer's gradients are already unscaled, so scaler.step does not unscale them,
            # it just calls optimizer.step() if no inf/NaN gradients are found
            scaler.step(self.optimizer)
            # Updates the scale for next iteration.
            scaler.update()
            if self.device_type == 'cuda':
                torch.cuda.synchronize()    # wait for the GPU to finish work
            
            dt = (time.time() - t0) * 1000.0    # in ms
            # For alpaca, tokens mapped to batch_size * context_length
            tokens_processed = self.train_loader.B * self.train_loader.T * self.grad_accum_steps
            tokens_per_sec = tokens_processed / dt

            print(f'step {step:4d} | loss: {batch_loss.item():.6f} | lr: {lr:.2e} | norm: {norm:.4f} | dt: {dt:.4f}ms | tok/sec: {tokens_per_sec:.4f}')
            with open(self.logpath, 'a') as f:
                f.write(f'{step} train {batch_loss.item():.6f}\n')

    def evaluate_validation(self, step):
        self.model.eval()    # sets model to eval mode
        self.val_loader.reset()
        # evaluate the model on validation set
        with torch.no_grad():
            val_loss_accum = 0.0
            val_steps = 20
            for _ in range(val_steps):
                inp, tar = self.val_loader.next_batch()
                inp, tar = inp.to(self.device), tar.to(self.device)
                with torch.autocast(device_type=self.device_type, dtype=torch.float16):
                    logits, loss = self.model(inp, tar)
                loss /= val_steps
                val_loss_accum += loss.detach()

        print(f'Val loss: {val_loss_accum.item():.4f}')
        with open(self.logpath, 'a') as f:
            f.write(f'{step} val {val_loss_accum.item():.4f}\n')

        if step > 0 and (step % 10000 == 0 or self.is_last_step):
            logdir = os.path.dirname(self.logpath)
            ckpt_path = os.path.join(logdir, f'model_{step:05d}.pt')
            checkpoint = {
                'model': self.model.state_dict(),
                'config': self.model.config,
                'step': step,
                'val_loss': val_loss_accum.item()
            }
            torch.save(checkpoint, ckpt_path)

    def generate_sequences(self, num_seq=4, max_tokens=32):
        self.model.eval()
        # Adjusted prompt for Alpaca testing
        prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nWhat is the capital of France?\n\n### Response:\n"
        tokens = self.token_encoder.encode(prompt)
        tokens = torch.tensor(tokens, dtype=torch.long)    # (n,)
        tokens = tokens.unsqueeze(0).repeat(num_seq, 1)    # (1,n) --> (num_seq, n)
        gen_tokens = tokens.to(self.device)
        sample_rng = torch.Generator(device=self.device)
        sample_rng.manual_seed(42)
        
        while gen_tokens.shape[-1] <= max_tokens + len(prompt):
            with torch.no_grad():
                with torch.autocast(device_type=self.device_type, dtype=torch.float16):
                    logits, loss = self.model(gen_tokens)    # (num_seq, n, vocab_size)
                logits = logits[:, -1, :]    # (num_seq, vocab_size)
                probs = F.softmax(logits, dim=-1)    # (num_seq, vocab_size)
                # take top-k 50 probs
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)    # (num_seq, 50), (num_seq, 50)
                # sample a token from top-50 probabilities
                ix = torch.multinomial(topk_probs, num_samples=1, generator=sample_rng)    # (num_seq, 1)
                next_tok = torch.gather(topk_indices, -1, ix)    # (num_seq, 1)
                gen_tokens = torch.cat([gen_tokens, next_tok], dim=1)
        
        # decode generated tokens and print generated text
        for i in range(num_seq):
            tokens = gen_tokens[i, :].tolist()
            gen_text = self.token_encoder.decode(tokens)
            print(f"> sample {i}:\n{gen_text}\n")


    def estimate_lr(self, step, warmup_steps, max_steps, max_lr, min_lr):
        """
        Learning rate scheduler: Cosine-decay learning schedule with warmup
        """
        if step < warmup_steps:
            return max_lr * (step+1) / warmup_steps
        if step > max_steps:
            return min_lr
        decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)


@dataclass
class GPTConfig:
    context_length: int = 1024
    vocab_size: int = 50257
    num_layers: int = 12
    embd_size: int = 768
    num_heads: int = 12


def main():
    # Print the hyperparameters
    print("Hyperparameter Configuration:")
    for key, value in vars(config).items():
        print(f"{key}: {value}")

    # create the logs directory if it doesn't exist
    os.makedirs(config.logdir, exist_ok=True)
    logpath = os.path.join(config.logdir, 'log.txt')
    with open(logpath, 'w') as f:
        pass

    # No DDP - single process only
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    print(f'using device: {device}')
    device_type = 'cuda' if device.startswith('cuda') else 'cpu'

    # setting seed for reproducibility
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)

    assert config.total_batch_size % (config.mini_batch_size * config.context_length) == 0, f'ensure total_batch_size divisible by B*T'
    grad_accum_steps = config.total_batch_size // (config.mini_batch_size * config.context_length)
    
    print(f'desired batch size (number of tokens): {config.total_batch_size}')
    print(f'gradient accumulation steps: {grad_accum_steps}')

    # Point DataLoaderLite to the Alpaca directory
    from .dataloader import DataLoaderLite
    data_root = os.path.join(os.path.dirname(__file__), "../data/alpaca")
    train_loader = DataLoaderLite(B=config.mini_batch_size, T=config.context_length, process_rank=0, num_processes=1, split='train', data_root=data_root)
    val_loader = DataLoaderLite(B=config.mini_batch_size, T=config.context_length, process_rank=0, num_processes=1, split='val', data_root=data_root)

    # create GPT model from pretrained 125M huggingface weights
    print("Loading pretrained GPT-2 model weights...")
    model = GPT.from_pretrained('gpt2')

    
    model.to(device)
    if config.use_torch_compile:
        model = torch.compile(model)

    # Note: master_process is always True for single-GPU setup
    optimizer = model.configure_optimizers(weight_decay=config.weight_decay, lr=config.max_lr, device_type=device_type, master_process=True)
    token_encoder = tiktoken.get_encoding('gpt2')

    start_time = time.time()
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        token_encoder=token_encoder,
        eval_freq=config.eval_freq,
        grad_accum_steps=grad_accum_steps,
        device=device,
        logpath=logpath
    )

    max_steps = config.steps_per_epoch * config.num_epochs
    trainer.train(max_steps, config.warmup_steps, config.max_lr, config.min_lr)

    dt = (time.time() - start_time) / (60*60)
    print(f"Total training time: {dt:.4f}hr")

if __name__ == "__main__":
    main()

