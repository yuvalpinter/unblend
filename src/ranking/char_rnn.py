import torch
import random
from torch import nn, optim
import argparse
from tqdm import tqdm
from datetime import datetime

NUM_EPOCHS = 5
LEARNING_RATE = 0.01
CHUNK_LEN = 200
HIDDEN_DIM = 128
EMB_DIM = 200
BATCH_SIZE = 32
NUM_LAYERS = 1
DROPOUT = 0.5

class CharRNN(nn.Module):

    def __init__(self, vocab_size, emb_size, hidden_size, n_layers=NUM_LAYERS, dropout=DROPOUT):
        super(CharRNN, self).__init__()
        
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        # TODO trained hidden?
        self.encoder = nn.Embedding(vocab_size, emb_size)
        self.rnn = nn.GRU(emb_size, hidden_size, n_layers, batch_first=True)
        self.decoder = nn.Linear(hidden_size, vocab_size)
        
        self.drop = nn.Dropout(dropout)

    def forward(self, input, h_0=None):
        encoded = self.drop(self.encoder(input))
        output, hidden = self.rnn(encoded, h_0)
        output = self.decoder(self.drop(output))
        return output, hidden


def enc_c(c, dct):
    if c in dct:
        return dct[c]
    else:
        return random.choice(range(len(dct)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='train file location')
    parser.add_argument('--dev', help='dev file location')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--layers', type=int, default=NUM_LAYERS)
    parser.add_argument('--hid-dim', type=int, default=HIDDEN_DIM)
    parser.add_argument('--emb-dim', type=int, default=EMB_DIM)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    
    torch.manual_seed(496351)
    random.seed(496351)
    
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f'run timestamp: {run_timestamp}')
    
    with open(args.dev) as fdev:
        ddocs = fdev.readlines()
    
    with open(args.train) as ftrain:
        docs = ftrain.readlines()
        charset = list(set(''.join(docs)))
        with open(f'models/{run_timestamp}_charset', 'w') as charf:
            charf.write(''.join(charset))
        
        char_to_id = {c:i for i,c in enumerate(charset)}
        num_chars = len(charset)
        print(f'found {num_chars} characters.')
        
        docs_idcs = [[char_to_id[c] for c in d] for d in docs]
        ddocs_idcs = [[enc_c(c, char_to_id) for c in d] for d in ddocs]
        
        fwd_model = CharRNN(num_chars, emb_size=args.emb_dim, hidden_size=args.hid_dim, n_layers=args.layers)
        fopt = optim.Adam(fwd_model.parameters(), lr=LEARNING_RATE)  # 0.01
        
        bwd_model = CharRNN(num_chars, emb_size=args.emb_dim, hidden_size=args.hid_dim, n_layers=args.layers)
        bopt = optim.Adam(bwd_model.parameters(), lr=LEARNING_RATE)
        
        criterion = nn.CrossEntropyLoss()
        min_loss = float('inf')
        
        for ep in tqdm(range(args.epochs)):
            random.shuffle(docs_idcs)
            fwd_loss = 0.0
            bwd_loss = 0.0
            batch_finp = []
            batch_ftrg = []
            batch_binp = []
            batch_btrg = []
            for d in tqdm(docs_idcs):
                loc = 0  # chunk
                while loc < len(d):
                    chunk = d[loc:loc+CHUNK_LEN]
                    if len(chunk) < CHUNK_LEN:
                        # TODO pad instead?
                        loc += CHUNK_LEN
                        continue
                    batch_finp.append(chunk[:-1])
                    batch_ftrg.append(chunk[1:])
                    batch_binp.append(list(reversed(chunk))[:-1])
                    batch_btrg.append(list(reversed(chunk))[1:])
                    if len(batch_finp) >= args.batch_size:
                        # train step
                        
                        # forward
                        fwd_model.zero_grad()
                        fopt.zero_grad()

                        output, h_n = fwd_model(torch.tensor(batch_finp))
                        target = torch.tensor(batch_ftrg).view(args.batch_size, -1)
                        floss = criterion(output.transpose(1,2), target)

                        floss.backward()
                        fopt.step()
                        fwd_loss += (float(floss) * args.batch_size)
                        
                        # backward
                        bwd_model.zero_grad()
                        bopt.zero_grad()

                        output, h_n = bwd_model(torch.tensor(batch_binp))
                        target = torch.tensor(batch_btrg).view(args.batch_size, -1)
                        bloss = criterion(output.transpose(1,2), target)

                        bloss.backward()
                        bopt.step()

                        bwd_loss += (float(bloss) * args.batch_size)
                        batch_finp = []
                        batch_ftrg = []
                        
                        batch_binp = []
                        batch_btrg = []
                        
                    loc += CHUNK_LEN
            
            print(f'training loss at epoch {ep}: fwd {fwd_loss}, bwd {bwd_loss}')
            
            # dev
            with torch.no_grad():
                dev_floss = 0.0
                for d in tqdm(ddocs_idcs):
                    loc = 0  # chunk
                    while loc < len(d):
                        chunk = d[loc:loc+CHUNK_LEN]
                        if len(chunk) < CHUNK_LEN:
                            loc += CHUNK_LEN
                            continue
                        finp = torch.tensor(chunk[:-1])
                        ftrg = torch.tensor(chunk[1:])
                        
                        output, h_n = fwd_model(finp.view(-1,1))
                        floss = criterion(output.view(output.shape[0], output.shape[2], -1), ftrg.view(-1,1))
                        dev_floss += float(floss)
                        
                        loc += CHUNK_LEN
                
                print(f'dev loss at epoch {ep}: {dev_floss}')
                if dev_floss < min_loss:
                    print('saving model')
                    torch.save(fwd_model.state_dict(), f'models/{run_timestamp}_char_gru_f_ep{ep:02d}.pt')
                    torch.save(bwd_model.state_dict(), f'models/{run_timestamp}_char_gru_b_ep{ep:02d}.pt')
                    min_loss = dev_floss


if __name__=='__main__':
    main()
    
