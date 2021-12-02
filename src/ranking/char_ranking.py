import torch
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from char_rnn import CharRNN, enc_c, NUM_LAYERS, HIDDEN_DIM, EMB_DIM

SENT_CONTEXT_DELIM = '\t'
NEGSAMP_DELIM = '\t'

MASK_TEXT = '[MASK]'


def rev(text):
    return ''.join(reversed(text))


def nltk_clean(s):
    return s.replace('""','"').replace('""','"')
    
    
def conditioned_loss(trg, model, loss_crit, hid_last, out_last):
    inp = trg[:-1]
    out_w, _ = model(torch.tensor(inp).view(1, -1), hid_last)
    preds = torch.cat([out_last, out_w], dim=1)
    out_w = preds[0]
    return float(loss_crit(out_w, torch.tensor(trg))) # takes mean by default
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--char-dict', help='character list file location')
    parser.add_argument('--fwd-in', help='forward model file location')
    parser.add_argument('--bwd-in', help='backward model file location')
    parser.add_argument('--layers', type=int, default=NUM_LAYERS)
    parser.add_argument('--hid-dim', type=int, default=HIDDEN_DIM)
    parser.add_argument('--emb-dim', type=int, default=EMB_DIM)
    
    parser.add_argument('--anns', help='annotations')
    parser.add_argument('--contexts', help='contexts file location')
    parser.add_argument('--neg-samps', help='negative samples file location')
    parser.add_argument('--not-nltk', action='store_true', help='input was not tokenized using nltk')
    
    parser.add_argument('--output', help='output file location')
    
    args = parser.parse_args()

    with open(args.char_dict) as chars_f:
        char_dict = {c:i for i,c in enumerate(chars_f.read())}
    num_chars = len(char_dict)
    print(f'loaded {num_chars} characters')
    
    # load backup bases
    bases = {}
    with open(args.anns) as in_f:
        in_f.readline()  # header
        for l in in_f:
            # Annotation, Form, Bases, Semantic Affixes, Triv Affixes, Additional Segment Count, PAXOBS, Blend Type
            _, fullform, basess, _, _, _, _, _ = l.strip().split('\t')
            bases[fullform] = basess.split(' ')
    
    fmodel = CharRNN(num_chars, args.emb_dim, args.hid_dim, n_layers=args.layers)
    fmodel.load_state_dict(torch.load(args.fwd_in))
    fmodel.eval()
    
    bmodel = CharRNN(num_chars, args.emb_dim, args.hid_dim, n_layers=args.layers)
    bmodel.load_state_dict(torch.load(args.bwd_in))
    bmodel.eval()
    
    loss_crit = torch.nn.CrossEntropyLoss()

    sent_contexts_df = pd.read_csv(args.contexts, sep=SENT_CONTEXT_DELIM)
    neg_samps = pd.read_csv(args.neg_samps, sep=NEGSAMP_DELIM).to_dict(orient='record')
    
    results = defaultdict(list)
    resultlog = []
    exceptions = 0
    for ix, row in tqdm(sent_contexts_df.iterrows()):
        neo = row['neologism']
        if neo not in bases:
            continue
        
        f_tru = bases[neo][0]
        b_tru = bases[neo][-1]
        instkey = (neo, f_tru, b_tru)
        
        negs = [x for x in neg_samps if x['FORM'] == neo]
        if len(negs) == 0:
            instres = (10000, 10000, 0, 0, 'CHECK', 'CHECK')
            results[instkey].append(instres)
            continue
            
        sent = '\\n'+row['sentence_context']+'\\n'
        if not args.not_nltk:
            sent = nltk_clean(sent)
        tnes = rev(sent)
        
        # find the location for each bases's start
        start_loc = sent.find(MASK_TEXT)
        end_loc = tnes.find(rev(MASK_TEXT))
        
        sent_chars = [enc_c(c, char_dict) for c in sent[:start_loc]]
        tnes_chars = [enc_c(c, char_dict) for c in tnes[:end_loc]]
        
        # run each model on the input
        fwd_outs, f_hids = fmodel(torch.tensor(sent_chars).view(1, -1))
        f_out_last = fwd_outs[:,-1,:].view(1, 1, num_chars)  # needed for predicting first candidate char
        f_hid_last = f_hids[:,-1,:].view(args.layers, 1, args.hid_dim)
        
        bwd_outs, b_hids = bmodel(torch.tensor(tnes_chars).view(1, -1))
        b_out_last = bwd_outs[:,-1,:].view(1, 1, num_chars)
        b_hid_last = b_hids[:,-1,:].view(args.layers, 1, args.hid_dim)
        
        # evaluate loss on each candidate
        fcand_losses = {}
        bcand_losses = {}
        
        ftrg = [enc_c(c, char_dict) for c in f_tru]
        fcand_losses[f_tru] = conditioned_loss(ftrg, fmodel, loss_crit, f_hid_last, f_out_last)
        btrg = [enc_c(c, char_dict) for c in rev(b_tru)]
        bcand_losses[b_tru] = conditioned_loss(btrg, bmodel, loss_crit, b_hid_last, b_out_last)
        
        for n in negs:
            w = n['NEGATIVE']
            if n['PLACE'] == "PRE":
                if w not in fcand_losses:  # should always be the case
                    trg = [enc_c(c, char_dict) for c in w]
                    fcand_losses[w] = conditioned_loss(trg, fmodel, loss_crit, f_hid_last, f_out_last)
            elif n['PLACE'] == "SUF":
                if w not in bcand_losses:  # should always be the case
                    trg = [enc_c(c, char_dict) for c in rev(w)]
                    bcand_losses[w] = conditioned_loss(trg, bmodel, loss_crit, b_hid_last, b_out_last)
            else:
                raise Exception(f'unknown location value: {n["PLACE"]}')
        
        # complete from bases
        if f_tru not in fcand_losses:
            fcand_losses[f_tru] = 0.0
        if b_tru not in bcand_losses:
            bcand_losses[b_tru] = 0.0
            
        # rank
        ftnll = fcand_losses[f_tru]
        btnll = bcand_losses[b_tru]
        fnlls = sorted(fcand_losses.values())
        bnlls = sorted(bcand_losses.values())
        frank = fnlls.index(ftnll) + 1
        brank = bnlls.index(btnll) + 1
        instres = (frank, brank, fnlls[0], bnlls[0], len(fnlls), len(bnlls))
        instlog = (f'{ftnll:.3f}', f'{btnll:.3f}', str(frank), str(brank), f'{fnlls[0]:.3f}', f'{bnlls[0]:.3f}', str(len(fnlls)), str(len(bnlls)))
        results[instkey].append(instres)
        resultlog.append(instkey + instlog)
        
    for b, bs in bases.items():
        k = (b, bs[0], bs[-1])
        if k not in results:
            instres = (10000, 10000, 0, 0, 'CHECK', 'CHECK')
            results[k].append(instres)
            
    with open(args.output+'.log', 'w') as outf:
        outf.write('form\tpref\tsuf\tpref nll\tsuf nll\tpref rank\tsuf rank\tpref min\tsuf min\t#prefs\t#sufs\n')
        for res in resultlog:
            outf.write('\t'.join(res) + '\n')
            
    with open(args.output, 'w') as outf:
        outf.write('Form\tPref\tSuf\tNULL\tBoth rank\tPref rank\tSuf rank\tpref max\tsuf max\t#prefs\t#sufs\n')
        for k, resl in sorted(results.items()):
            mean_frank = np.average([r[0] for r in resl])
            mean_brank = np.average([r[1] for r in resl])
            both_rank = mean_frank * mean_brank
            mean_fmax = np.average([r[2] for r in resl])
            mean_bmax = np.average([r[3] for r in resl])
            assert len(set([r[4] for r in resl])) == 1, f'uneven prefix candidates in {k}'
            assert len(set([r[5] for r in resl])) == 1, f'uneven suffix candidates in {k}'
            outf.write('\t'.join(k) + f'\t\t{both_rank:.1f}\t{mean_frank:.1f}\t{mean_brank:.1f}\t{mean_fmax:.3f}\t{mean_bmax:.3f}\t{resl[0][-2]}\t{resl[0][-1]}\n')
            
    print(f'finished with {exceptions} unfound true values. reporting {len(resultlog)} results from {len(results)} blends.')


if __name__ == '__main__':
    main()
