'''
Testing similarity between bases composing blends relative to negative examples based on static word embeddings.
'''
import argparse
from tqdm import tqdm
from gensim.models import KeyedVectors
from collections import defaultdict
import numpy as np


def main():
    # handle input
    parser = argparse.ArgumentParser()
    parser.add_argument('--anns', help='annotations')
    parser.add_argument('--vectors', help='word vectors in w2v format')
    parser.add_argument('--two-sides', action='store_true', help='compute two-sides MRR')
    parser.add_argument('--neg-samps', help='table (tsv) of negative sample pairs')
    parser.add_argument('--output', help='file for data output')
    args = parser.parse_args()
    
    # load vectors
    vecs = KeyedVectors.load_word2vec_format(args.vectors)
    print(f'read {len(vecs.vocab)} word vectors')
    
    # load backup bases
    bases = {}
    with open(args.anns) as in_f:
        in_f.readline()  # header
        for l in in_f:
            # Annotation, Form, Bases, Semantic Affixes, Triv Affixes, Additional Segment Count, PAXOBS, Blend Type
            _, fullform, basess, _, _, _, _, _ = l.strip().split('\t')
            bases[fullform] = basess.split(' ')
    
    # load data
    prefs = {}
    pref2s = {}
    sufs = {}
    suf2s = {}
    pref_negs = defaultdict(list)
    suf_negs = defaultdict(list)
    duds = []
    with open(args.neg_samps) as negs_inf:
        _ = negs_inf.readline()  # header: FORM PLACE NEGATIVE THROW
        for l in negs_inf.readlines():
            form, pl, neg, _ = l.strip().split('\t')
            tr = bases[form][0] if pl == 'PRE' else bases[form][-1]
            if tr not in vecs and form not in duds:
                print(f"{form}'s base {tr} not found in vector file")
                duds.append(form)
                continue
            if pl == 'SUF':
                if form not in sufs:
                    sufs[form] = tr
                elif sufs[form] != tr and form not in suf2s:
                    suf2s[form] = tr
                elif form in suf2s and suf2s[form] != tr:
                    assert sufs[form] == tr, f'suffix mismatch: {form} {sufs[form]} {suf2s[form]} {tr}'
                if neg in vecs:
                    if sufs[form] == tr:
                        suf_negs[form].append(neg)
                    else:  # suf2s
                        pref_negs[form].append(neg)
            elif pl == 'PRE':
                if form not in prefs:
                    prefs[form] = tr
                elif prefs[form] != tr and form not in pref2s:
                    pref2s[form] = tr
                elif form in pref2s and pref2s[form] != tr:
                    assert prefs[form] == tr, f'prefix mismatch: {form} {prefs[form]} {pref2s[form]} {tr}'
                if neg in vecs:
                    if prefs[form] == tr:
                        pref_negs[form].append(neg)
                    else:  # pref2s
                        suf_negs[form].append(neg)
            else:
                print(f'invalid place marker: {pl}')
    
    # report any form that doesn't have both pref and suf
    for f in prefs:
        if f not in sufs and f not in pref2s:
            print(f'no suffix or second prefix encountered for {f}')
            sufs[f] = bases[f][-1]
    for f in sufs:
        if f not in prefs and f not in suf2s:
            print(f'no prefix or second suffix encountered for {f}')
            prefs[f] = bases[f][0]
    
    results = []
    brrs = []
    lrrs = []
    rrrs = []
    for f in tqdm(bases):
        if f not in prefs:
            results.append((f, bases[f][0], bases[f][-1],
                            'N/A', 'LOOK UP', 'LOOK UP', 'LOOK UP', 'N/A', 'N/A', 'LOOK UP', 'LOOK UP'))
            brrs.append(0)
            lrrs.append(0)
            rrrs.append(0)
            continue
            
        p = suf2s[f] if f in suf2s else prefs[f]
        s = pref2s[f] if f in pref2s else sufs[f]
        if p not in vecs or s not in vecs:
            results.append((f, bases[f][0], bases[f][-1],
                            'N/A', 'LOOK UP', 'LOOK UP', 'LOOK UP', 'N/A', 'N/A', 'LOOK UP', 'LOOK UP'))
            continue
        base_sim = vecs.similarity(p, s)
        
        if args.two_sides:
            neg_sims = [vecs.similarity(allp, alls) for allp in [p]+pref_negs[f] for alls in [s]+suf_negs[f]]
            brank = 1 + len([n for n in neg_sims if n > base_sim])
            brstr = (str(brank),)
            brrs.append(1/brank)
        
        if len(pref_negs[f]) > 0:
            neg_psims = [vecs.similarity(negp, s) for negp in pref_negs[f]]
            maxpsim = max(neg_psims)
            prank = 1 + len([ps for ps in neg_psims if ps > base_sim])
        else:
            neg_psims = []
            maxpsim = 0.0
            prank = 1
        lrrs.append(1/prank)
        
        if len(suf_negs[f]) > 0:
            neg_ssims = [vecs.similarity(p, negs) for negs in suf_negs[f]]
            maxssim = max(neg_ssims)
            srank = 1 + len([ss for ss in neg_ssims if ss > base_sim])
        else:
            neg_ssims = []
            maxssim = 0.0
            srank = 1
        rrrs.append(1/srank)
        
        results.append((f, p, s, f'{base_sim:.3f}') + brstr +
                       (str(prank), str(srank), f'{maxpsim:.3f}', f'{maxssim:.3f}',
                        str(len(neg_psims)+1), str(len(neg_ssims)+1)))
    
    with open(args.output, 'w') as outf:
        # header
        if args.two_sides:
            both_head = 'Both rank\t'
        else:
            both_head = ''
        outf.write(f'Form\tPref\tSuf\tBase sim\t{both_head}'
                   f'Pref rank\tSuf rank\tpref max sim\tsuf max sim\t#prefs\t#sufs\n')
        for res in sorted(results):
            outf.write('\t'.join(res)+'\n')
            
    print(len(lrrs), len(rrrs), len(brrs))
    print(np.average(lrrs), np.average(rrrs), np.average(brrs))

if __name__ == '__main__':
    main()
