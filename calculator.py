import torch 
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

##### Data generation hyperparams
n_digit = 2   # how many digit at most for the summands in the addition problems
max_summand = 10**n_digit
len_prob = n_digit * 2 + 2
len_answer = n_digit + 1

##### Scaled-up hyperparams
# batch_size = 64
# block_size = 256 #k-gram
# max_iters = 5000
# eval_interval = 500
# learning_rate = 3e-4
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# eval_iters = 200
# n_embd = 384
# n_head = 6
# n_layer = 6
# dropout = 0.2
##### Baby model hyperparams
batch_size = 32
block_size = len_prob + n_digit + 1 #k-gram
max_iters = 2500
eval_interval = 500
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32
n_head = 4
n_layer = 4
dropout = 0.2

torch.manual_seed(42)
np.random.seed(42)

chars = [str(i) for i in range(10)] + ['+', '=', ' ']
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
itos[-1] = '*'
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])


def mask(encoded):
    encoded[:len_prob] = -1
    return encoded

def get_batch():
    problems = []
    for _ in range(batch_size):
        a = np.random.randint(max_summand)
        b = np.random.randint(max_summand)
        prob = f'{a:{n_digit}}+{b:{n_digit}}={str(a+b)[::-1]:{n_digit+2}}'
        problems.append(torch.tensor(encode(prob)))
    x = torch.stack([prob[:block_size] for prob in problems])
    y = torch.stack([mask(prob)[1:block_size+1] for prob in problems])
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch()
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd) # projection for residual layer, not sure if necessary
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        o = torch.cat([h(x) for h in self.heads], dim=-1)
        o = self.dropout(o)
        return self.proj(o)

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5
        wei = wei.masked_fill(self.tril[:T, :T]==0, float('-inf')) 
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd), # per token level
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd), # projection for residual layer
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        xn = self.ln1(x) # pre-norm
        x = x + self.sa(xn) # residual connection
        xn = self.ln2(x) # pre-norm
        x = x + self.ffwd(xn) # residual connection
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size) # lm for language modeling
        self.ffwd = FeedForward(n_embd)

    def forward(self, idx, targets=None): # both idx and targets of shape [B, T, C]
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # idx: [B, T] -> [B, T, C] because each element is mapped to a length C embedding
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        x = self.ffwd(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            # cross ent in pytorch takes inputs with channel at the second dimension
            # ignore -1 which masks the problem statement
            loss = F.cross_entropy(logits, targets, ignore_index=-1) 

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) # sample rather than take max proba, interesting!
            idx = torch.cat((idx, idx_next), dim=1) # extent in the time dimension (dim 1)
        return idx


model = GPTLanguageModel()
m = model.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}", flush=True)
    xb, yb = get_batch()
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

x = get_batch()[0].to(device).split(1, dim=0)
for example in x:
    context = example[:, :len_prob]
    decoded = decode(m.generate(context, max_new_tokens=len_answer)[0].tolist())
    decoded = decoded[:len_prob] + decoded[len_prob:][::-1] # reverse the inverted representation
    print(decoded)
