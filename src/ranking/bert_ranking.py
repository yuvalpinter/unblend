import torch
import pandas as pd
import click

@click.command()
@click.option('--sent_context_filename')
@click.option('--neg_samp_filename')
@click.option('--save_to_file', type=bool, default=False)
@click.option('--segmentation_filename', default=None)
@click.option('--parse_colname', default='segmentinfo')
@click.option('--word_colname', default='word', help="neologism column")
@click.option('--segmentation_delim', default='\t')
@click.option('--bert_type', default='bert-base-uncased')
@click.option('--sent_context_delim', default='\t')
@click.option('--negsamp_delim', default='\t')
def main(sent_context_filename, neg_samp_filename, save_to_file,
         bert_type, sent_context_delim, negsamp_delim):
    from transformers import BertForMaskedLM, BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(bert_type)
    bmodel = BertForMaskedLM.from_pretrained(bert_type)
    print(f"Tokenizer and Bert model {bert_type} loaded")
    sent_contexts_df = pd.read_csv(
        sent_context_filename, sep=sent_context_delim)
    neg_samps = pd.read_csv(
        neg_samp_filename, sep=negsamp_delim)
    rankings_for_pos_negative_portmanteaus = get_scores(
        sent_contexts_df, neg_samps, tokenizer, bmodel)
    print("Done")
    if save_to_file:
        from time import time
        from datetime import date
        import json
        time_ = time()
        todays_date = date.today().strftime("%Y%m%d")
        outfilename = f"./data/{todays_date}_{time_}_scores_bert_portmanteaus.json"
        with open(outfilename, 'w') as f:
            for line in rankings_for_pos_negative_portmanteaus:
                f.write(json.dumps(line) + '\n')


def get_scores(s, n, tokenizer, model):
    rankings_for_pos_negative_portmanteaus = []
    n_records = n.to_dict(orient='record')
    for ix, row in s.iterrows():
        neo = row['neologism']
        negs = [x for x in n_records if x['FORM'] == neo]
        vecs, target_locations = _munge_sentence_mask(
            row, tokenizer, model)
        these_vecs = vecs[0]
        for record in negs:
            pos, neg, position = (record['TRUE'], record['NEGATIVE'], record['PLACE'])
            pos_tokenized, neg_tokenized = (tokenizer.tokenize(pos), tokenizer.tokenize(neg))
            pos_index = [tokenizer.vocab[x] for x in pos_tokenized]
            neg_index = [tokenizer.vocab[x] for x in neg_tokenized]
            for t in target_locations:
                loc_preds = these_vecs[0, t, :]  # scores over word embeddings
                # get prediction scores of subwords
                pos_predictions = [_softmax_score(loc_preds, i) for i in pos_index]
                # get activations of negative examples
                neg_predictions = [_softmax_score(loc_preds, i) for i in neg_index]
                rankings_for_pos_negative_portmanteaus.append(
                    [neo, pos, neg, record['PLACE'],
                     pos_tokenized, pos_index, pos_predictions,
                     neg_tokenized, neg_index, neg_predictions])
    return rankings_for_pos_negative_portmanteaus


def _softmax_score(vec, idx):
    return float(vec[idx].exp() / vec.exp().sum())
    

def _munge_sentence_mask(row, tokenizer, model):
    # from stackoverflow
    sent = row['sentence_context']
    sent_ = '[CLS]' + sent + '[SEP]'
    tokenized_sent = tokenizer.tokenize(sent_)
    target_indices = [i for i, x in enumerate(tokenized_sent) if x=='[MASK]']
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_sent)
    segments_ids = [0] * len(tokenized_sent)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    with torch.no_grad():
        predictions = model(tokens_tensor, segments_tensors)
    return predictions, target_indices


def _munge_sentence(row, tokenizer, model):
    sent = row['sentence_context']
    sent_ = '[CLS]' + sent + '[SEP]'
    tokenized_sent = tokenizer.tokenize(sent_)
    input_ids = torch.tensor([tokenizer.encode(
        sent_)]).squeeze(-1)  # batch size = 1
    target_locations = [
        i for i, x in enumerate(tokenized_sent)
        if x == '[MASK]']
    vecs = model(input_ids)
    return vecs, target_locations


def _get_parses(segmentation_filename, delim,
                parse_colname, word_colname):
    words_w_parses_df = pd.read_csv(
        segmentation_filename, sep=delim)
    word_parses_dict = dict(zip(words_w_parses_df[word_colname],
                                words_w_parses_df[parse_colname]))
    parsed_parses_realization = _parse_to_list(word_parses_dict)
    return parsed_parses_realization


def _parse_to_list(word_parses_dict: dict):
    import re
    parsed_parses = {}
    for term in word_parses_dict:
        split_str = []
        for x in word_parses_dict[term].split('; '):
            if x[0] == '-':
                split_str.append(x.split("-")[1])
            else:
                split_str.append(x.split("-")[0])
            parsed_parses[term] = split_str
    parsed_parses_realization = {}
    for word in parsed_parses:
        parts = parsed_parses[word]
        for_comparison_to_bert = []
        for i, m in enumerate(parts):
            ow = re.sub(r'\[(.)\]', r'\1', m)
            ow = re.sub(r'\{.\}', '', ow)
            if i == 0:
                for_comparison_to_bert.append(ow)
            else:
                for_comparison_to_bert.append("##" + ow)
        parsed_parses_realization[word] = for_comparison_to_bert
    return parsed_parses_realization

if __name__=="__main__":
    main()