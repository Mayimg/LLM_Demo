import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Tokenizer :

    @staticmethod
    def create_vocab(dataset) :
        vocab = {
            token:index for index, token in enumerate(sorted(list(set(dataset))))
        }
        vocab['<UNK>'] = len(vocab)
        return vocab
    
    def __init__(self, vocab) :
        self.vocab_encode = {str(k) : int(v) for k, v in vocab.items()}
        self.vocab_decode = {v : k for k, v in self.vocab_encode.items()}

    def encode(self, text) :
        return [self.vocab_encode.get(char, self.vocab_encode['<UNK>']) for char in text]
    
    def decode(self, indices) :
        return ''.join([self.vocab_decode.get(index, '<UNK>') for index in indices])


class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size, dropout):  # Added block_size parameter
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Create tril matrix with block_size instead of head_size
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        v = self.value(x) # (B, T, head_size)

        # Compute attention scores
        wei = q @ k.transpose(-2, -1) # (B, T, T)
        wei = wei * k.shape[-1]**-0.5 # Scale by sqrt(head_size)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # Apply attention to values
        out = wei @ v # (B, T, head_size)
        return out
    
class MultiHeadAttention(nn.Module) :
    def __init__(self, n_embd, n_head, head_size, block_size, dropout):  # Added block_size
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size, dropout) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x) :
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class FeedFoward(nn.Module) :
    def __init__(self, n_embd, dropout) :
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout, )
        )

    def forward(self, x) :
        return self.net(x)

class Block(nn.Module) :

    def __init__(self, n_embd, n_head, block_size, dropout):  # Added block_size
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_embd, n_head, head_size, block_size, dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x) :
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, x, targets=None):
        B, T = x.shape
        tok_emb = self.token_embedding_table(x)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        B, T, C = logits.shape  # この行を条件分岐の外に移動
        
        if targets is None:
            loss = None
        else:
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss

    def generate(self, idx, block_size, max_new_tokens=200):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self.forward(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=-1)
        return idx


batch_size = 16
block_size = 500
max_iters = 100000
eval_interval = 1000
learning_rate = 1e-3

eval_iters = 444
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0

torch.manual_seed(1337)

def read_data(file_path):
    texts = []
    summaries = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file :
            json_data = json.loads(line)
            text = json_data.get('text')
            summary = json_data.get('summary')

            if text and summary:
                texts.append(text)
                summaries.append(summary)

    return texts, summaries

file_path_train = 'japanese_train.jsonl'
file_path_val = 'japanese_val.jsonl'
texts, summaries = read_data(file_path_train)

dataset = ''.join(texts + summaries)
vocab = Tokenizer.create_vocab(dataset)
tokenizer = Tokenizer(vocab)

train_data = ""
for text, summary in zip(texts, summaries):
    train_data += "<BOS>" + text + "<SUMMARY>" + summary + "<EOS>"

texts, summaries = read_data(file_path_val)
val_data = ""
for text, summary in zip(texts, summaries):
    val_data += "<BOS>" + text + "<SUMMARY>" + summary + "<EOS>"

train_data = torch.tensor(tokenizer.encode(train_data), dtype=torch.long).to(device)
val_data = torch.tensor(tokenizer.encode(val_data), dtype=torch.long).to(device)

def get_batch(split='train') :
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

vocab_size = len(vocab)


model = BigramLanguageModel(vocab_size, n_embd, n_head, n_layer, block_size, dropout).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


for iter in tqdm(range(max_iters)) :
    xb, yb = get_batch()

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if (iter > 0 and iter % eval_interval == 0) or iter == max_iters - 1 :
        with torch.no_grad() :
            val_loss = 0
            for _ in tqdm(range(eval_iters)) :
                xb, yb = get_batch('val')
                logits, loss = model(xb, yb)
                val_loss += loss.item()
            print(f'Iter: {iter}, Val Loss: {val_loss / eval_iters}')


            texts, _ = read_data(file_path_val)
            vali_input_text = "<BOS>" + texts[0] + "<SUMMARY>"
            context = torch.tensor(tokenizer.encode(vali_input_text), dtype=torch.long).to(device)
            idx = model.generate(context.unsqueeze(0), block_size)
            out_text = tokenizer.decode(idx[0].tolist())

            print("Context: ", out_text.split("<SUMMARY>")[0])
            print("Generated: ", out_text.split("<SUMMARY>")[1].split("<EOS>")[0])

        








