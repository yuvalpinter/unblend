import argparse
import numpy as np
import pandas as pd
import torch
from collections import defaultdict, Counter
from itertools import product
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig


def _softmax_score(vec, idx):
    return float(vec[idx].exp() / vec.exp().sum())


def mask_blend(sentence, blend, replacement='[MASK]'):
    sentence = sentence.lower() # this is necessary because blends are lowercase
    sentence = sentence.replace(blend, replacement)
    return sentence


def augment_sentence(tokenized_sentence, twosided_flag=False):
    tokenized_sentence = ['[CLS]'] + tokenized_sentence + ['[SEP]']
    tokenized_sentence_tmp = []
    for i, w in enumerate(tokenized_sentence):
        if w=='[MASK]':
            if twosided_flag:
                tokenized_sentence_tmp += ['[MASK]', '[MASK]']
            else:
                tokenized_sentence_tmp.append(w)
        else:
            tokenized_sentence_tmp.append(w)
    return tokenized_sentence_tmp


def find_mask(tokenized_sentence, twosided_flag=False):
    target_indices = [i for i, x in enumerate(tokenized_sentence) if x=='[MASK]']
    tokenized_sent_tmp = []
    for i, x in enumerate(target_indices):
        if twosided_flag: # then double i's
            tokenized_sent_tmp.extend([x + i, x + i + 1])
        else:
            tokenized_sent_tmp.append(x)
    return tokenized_sent_tmp
    
    
def embed_sentence(model, tokenizer, augmented_sentence):
    # check sentence shape
    if (augmented_sentence[0]!='[CLS]') or (augmented_sentence[-1]!='[SEP]'):
        print("Please check your tokenization.")
    indexed_tokens = tokenizer.convert_tokens_to_ids(augmented_sentence)
    segments_ids = [0] * len(augmented_sentence)
    tokens_tensor, segments_tensor = torch.tensor([indexed_tokens]), torch.tensor([segments_ids])
    with torch.no_grad():
        return model(tokens_tensor, segments_tensor)
    
    
def parse_single_sentence(model, tokenizer, sentence, blend, twosided_flag=True):
    masked_sentence = mask_blend(sentence, blend)
    tokenized_sentence = tokenizer.tokenize(masked_sentence)
    augmented_sentence = augment_sentence(tokenized_sentence, twosided_flag=twosided_flag)
    target_indices = find_mask(tokenized_sentence, twosided_flag=twosided_flag)
    embedded_sentence = embed_sentence(model, tokenizer, augmented_sentence)
    return embedded_sentence, target_indices
    
    
def get_word_embedding(embedded_sentence, target_index):
    return embedded_sentence[0, target_index + 1, :].detach()
    
    
def get_piece_score(embedding, piece_index, raw=False):
    if raw:
        score = embedding[piece_index].item()
    else:
        score = _softmax_score(embedding, piece_index)
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--anns', help='annotations file in tsv format')
    parser.add_argument('--neg-samps', help='negative samples file in tsv format')
    parser.add_argument('--sentences', help='contexts for each word in tsv format, leave empty for no context')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--outf', help='output file in tsv format')
    args = parser.parse_args()
    
    # load from files
    neg_samps = pd.read_csv(args.neg_samps, sep="\t", encoding='utf-8')
    
    bases = {}
    with open(args.anns) as in_f:
        in_f.readline()  # header
        for l in in_f:
            # Annotation, Form, Bases, Semantic Affixes, Triv Affixes, Additional Segment Count, PAXOBS, Blend Type
            _, fullform, basess, _, _, _, _, _ = l.strip().split('\t')
            bases[fullform] = basess.split(' ')
    
    if args.sentences is not None:
        sentences = pd.read_csv(args.sentences, sep="\t")
        blend_sentences_only = sentences[sentences['Category']=='blend']
        blend_sentences_only = blend_sentences_only.reset_index().iterrows()
    else:
        blend_sentences_only = []
        for b in bases.keys():
            blend_sentences_only.append((None, {'sentence':f'{b}', 'Word':b}))
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True)
    model = BertForMaskedLM.from_pretrained('bert-base-uncased', config=config)
    model.eval()
    
    # main loop
    ranks = defaultdict(list)
    lcands = {}
    rcands = {}
    bcands = {}
    token_counts = {}
    token_neg_counts = []
    for _, row in tqdm(blend_sentences_only):
        sentence = row['sentence']
        blend = row['Word']

        # run bert
        embedded_sentence, target_indices = parse_single_sentence(model, tokenizer, sentence, blend)

        # get candidate lists
        true_bases = bases[blend]
        if len(true_bases) != 2 and args.verbose:
            print(f'{len(true_bases)} bases in {blend}')
        t_l = true_bases[0]
        t_r = true_bases[-1]
        
        negsamp_df = neg_samps[neg_samps['FORM']==blend]
        negsamp_prefs = negsamp_df[negsamp_df['PLACE']=='PRE']['NEGATIVE'].tolist()
        negsamp_sufs = negsamp_df[negsamp_df['PLACE']=='SUF']['NEGATIVE'].tolist()
       
        all_ls = [t_l] + negsamp_prefs
        all_rs = [t_r] + negsamp_sufs
        lcands[blend] = len(all_ls)
        rcands[blend] = len(all_rs)
        
        all_l_toks = [tokenizer.encode(b) for b in all_ls]  # list of list of ints
        all_r_toks = [tokenizer.encode(b) for b in all_rs]  # list of list of ints
        true_l_len = len(all_l_toks[0])
        true_r_len = len(all_r_toks[0])
        
        # token count stats
        if t_l not in token_counts:
            token_counts[t_l] = true_l_len
            for toks in all_l_toks[1:]:
                token_neg_counts.append(len(toks))
                
        if t_r not in token_counts:
            token_counts[t_r] = true_r_len
            for toks in all_r_toks[1:]:
                token_neg_counts.append(len(toks))
        
        # scoring
        l_scores = np.zeros(len(tokenizer))
        r_scores = np.zeros(len(tokenizer))
        trg_i = 0
        while trg_i < len(target_indices):
            # try summing raw scores
            l_scores += get_word_embedding(embedded_sentence[0], target_indices[trg_i]).numpy()
            trg_i += 1
            r_scores += get_word_embedding(embedded_sentence[0], target_indices[trg_i]).numpy()
            trg_i += 1
        l_exp = np.exp(l_scores)
        r_exp = np.exp(r_scores)
        l_softmax = l_exp / (l_exp.sum())
        r_softmax = r_exp / (r_exp.sum())
        
        pair_token_ids = product(all_l_toks, all_r_toks)
       
        pair_scores = []
        
        pairs = []
        for l, r in pair_token_ids:
            pair_scores.append(-l_softmax[l[0]] - r_softmax[r[0]])  # so sort is good to bad
            pairs.append((l, r))  # for tie breaking
        bcands[blend] = len(pairs)
        assert bcands[blend] == lcands[blend] * rcands[blend]
        
        np_pscores = np.array(pair_scores)
        true_score = pair_scores[0]
        tied = np.where(np_pscores==true_score)[0]
        sorted_scores = np.argsort(np_pscores)
        true_bases_rank = np.where(sorted_scores==0)[0][0] + 1
        
        toks_compared = 1
        # check if tie and true bases have more than one piece
        while len(tied) > 1 and (true_l_len > toks_compared or true_r_len > toks_compared):
            # anchor rank to start tiebreak from
            start_rank = true_bases_rank
            
            new_cands = np.array(pairs)[tied].tolist()
            assert len(new_cands) == len(tied)
            leftmask = ' '.join([tokenizer.ids_to_tokens[all_l_toks[0][j]]
                                 for j in range(min(toks_compared, len(all_l_toks[0])))])
            rightmask = ' '.join([tokenizer.ids_to_tokens[all_r_toks[0][j]]
                                  for j in range(min(toks_compared, len(all_r_toks[0])))])
            
            # re-rank finished token pairs above unfinished ones and save true's sublist
            # filtering missing tokens from left first, this is a heuristic.
            if true_l_len <= toks_compared:
                new_cands = [(i, c) for i, c in zip(tied, new_cands) if len(c[0]) <= toks_compared]
            else:
                leftmask += ' [MASK]'
                new_cands = [(i, c) for i, c in zip(tied, new_cands) if len(c[0]) > toks_compared]
                start_rank += (len(tied) - len(new_cands))
            if true_r_len <= toks_compared:
                new_cands = [(i, c) for i, c in new_cands if len(c[1]) <= toks_compared]
            else:
                rightmask += ' [MASK]'
                old_len  = len(new_cands)
                new_cands = [(i, c) for i, c in new_cands if len(c[1]) > toks_compared]
                start_rank += (old_len - len(new_cands))
            assert new_cands[0][0] == 0
            replacement = leftmask + ' ' + rightmask

            # tie-break
            trg_id = 0
            l_scores = np.zeros(len(l_scores))
            r_scores = np.zeros(len(r_scores))
            while trg_id < len(target_indices):
                if true_l_len > toks_compared:
                    l_scores += get_word_embedding(embedded_sentence[0], target_indices[trg_id]).numpy()
                    trg_id += 1
                if true_r_len > toks_compared:
                    r_scores += get_word_embedding(embedded_sentence[0], target_indices[trg_id]).numpy()
                    trg_id += 1
            l_exp = np.exp(l_scores)
            r_exp = np.exp(r_scores)
            l_softmax = l_exp / (l_exp.sum())
            r_softmax = r_exp / (r_exp.sum())
            
            cand_pair_scores = []
            for i, (l, r) in new_cands:
                score = 0.0
                if true_l_len > toks_compared: score -= l_softmax[l[toks_compared]]
                if true_r_len > toks_compared: score -= r_softmax[r[toks_compared]]
                cand_pair_scores.append(score)
                
            np_pscores = np.array(cand_pair_scores)
            true_score = cand_pair_scores[0]
            tied = np.where(np_pscores==true_score)[0]
            sorted_scores = np.argsort(np_pscores)
            new_true_rank = start_rank + np.where(sorted_scores==0)[0][0]        

            true_bases_rank = new_true_rank
            toks_compared += 1
        
        ranks[blend].append(true_bases_rank)
    
    # aggregate
    with open(args.outf, 'w') as out_f:
        rrs = []
        at1_hits = 0
        out_f.write('blend\tleft cands\tright cands\tpair cands\trank\treciprocal\n')
        
        for b in sorted(bases):
            if b not in ranks:
                negsamp_df = neg_samps[neg_samps['FORM']==b]
                negsamp_prefs = negsamp_df[negsamp_df['PLACE']=='PRE']['NEGATIVE'].tolist()
                negsamp_sufs = negsamp_df[negsamp_df['PLACE']=='SUF']['NEGATIVE'].tolist()
                lcands[b] = (1+len(negsamp_prefs))
                rcands[b] = (1 + len(negsamp_sufs))
                bcands[b] = (1+len(negsamp_prefs)) * (1 + len(negsamp_sufs))
                rank = bcands[b]
            else:
                rlist = ranks[b]
                rank = np.average(rlist)
            if rank == 1:
                at1_hits += 1
            rr = 1.0/rank
            rrs.append(rr)
            outstr = f'{b}\t{lcands[b]}\t{rcands[b]}\t{bcands[b]}\t{rank:.1f}\t{rr:.3f}\n'
            out_f.write(outstr)
        pat1 = at1_hits/len(ranks)
        mrr = np.average(rrs)
        out_f.write(f'\nH@1\t{at1_hits}\tP@1\t{pat1:.4f}\n\t\tMRR\t{mrr:.4f}\n')
        print(at1_hits, mrr, pat1)
    
    print(f'average gold token counts: {np.average(list(token_counts.values()))}')
    print(f'average false candidate token counts: {np.average(token_neg_counts)}')
    
    one_tok_bases = Counter(token_counts.values())[1]
    print(f'gold bases with only one token: {one_tok_bases}, prop {one_tok_bases/len(token_counts)}')


if __name__ == '__main__':
    main()

