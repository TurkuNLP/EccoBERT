import pytorch_lightning as pl
import transformers
import torch
import argparse
import random
import collections
import itertools
import gzip
import math
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error, explained_variance_score
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from captum.attr import LayerIntegratedGradients
from captum.attr import visualization as viz
from pathlib import Path

#    # one-hot encoding of words
#    train_batches = list(itertools.islice(data.train_dataloader(), 100))
#    eval_batches = [batch for batch in data.val_dataloader()]
#    train_encoded = [' '.join(tokenizer.convert_ids_to_tokens(e)) for batch in train_batches for e in batch['input_ids']]
#    train_labels = [e for batch in train_batches for e in batch['year']]
#    eval_encoded = [' '.join(tokenizer.convert_ids_to_tokens(e)) for batch in eval_batches for e in batch['input_ids']]
#    eval_labels = [e for batch in eval_batches for e in batch['year']]
#   
#    train_avg = torch.mean(torch.tensor(train_labels))
#    print(f"Train average: {train_avg:.2f}")
#    avg_baseline_preds = len(eval_labels)*[train_avg]
#    print(f"Average baseline mean squared error: {mean_squared_error(eval_labels, avg_baseline_preds)}")
#    print(f"Average baseline explained variance: {explained_variance_score(eval_labels, avg_baseline_preds)}")
# 
#    vectorizer = CountVectorizer()
#    vectorizer.fit(train_encoded)
#    train_data = vectorizer.transform(train_encoded)
#    eval_data = vectorizer.transform(eval_encoded)
#    regr = svm.SVR()
#    regr.fit(train_data, train_labels)
#    svr_pred = regr.predict(eval_data)
#    # print(f"Year predictions: {[f'{i:.2f}' for i in year_preds_display]}")
#    # print(f"Year target: {year_target_display}")
#    print(f"SVR mean squared error: {mean_squared_error(eval_labels, svr_pred)}")
#    print(f"SVR explained variance: {explained_variance_score(eval_labels, svr_pred)}")
#
    model.eval()
    model.cuda()
    print("Evaluating.")
    with torch.no_grad():
        batches = list(itertools.islice(data.val_dataloader(), 5000))
        # preds = []
        # labels = []
        year_preds = []
        year_target = []
        attention_masks = []
        collection_ids = []
        for batch in batches:
            output = model({k: v if type(v) == list else v.cuda() for k, v in batch.items()})
            # preds.append(output.logits.argmax(-1))
            # labels.append(batch['module'])
            year_preds.append(output.regression)
            year_target.append(batch['year'])
            attention_masks.append(batch['attention_mask'])
            collection_ids.append(batch['collection_id'])

    # preds = [e.item() for t in preds for e in t]
    # target = [e.item() for t in labels for e in t]
    year_preds = [e.item() for t in year_preds for e in t]
    year_target = [e.item() for t in year_target for e in t]
    year_difference = [p - t for p, t in zip(year_preds, year_target)]
    seq_lengths = [torch.sum(e).item() for t in attention_masks for e in t]
    collection_ids = [e for t in collection_ids for e in t]
    # print(f"Predictions: {preds[:10]}")
    # print(f"Target: {target[:10]}")
    year_preds_display, year_target_display = list(zip(*random.sample(list(zip(year_preds, year_target)), 50)))
    print(f"Year predictions: {[f'{i:.2f}' for i in  year_preds_display]}")
    print(f"Year target: {year_target_display}")
    def print_results(year_target, year_preds, collection_ids):
        collection_id_dict = {}
        for (t, p, m) in zip(year_target, year_preds, collection_ids):
            collection_id_dict.setdefault(m, []).append((t, p))

        collection_id_dict = {k: list(zip(*v)) for k, v in collection_id_dict.items()}
        collection_id_list = [(v[0], v[1], k) for k, v in collection_id_dict.items()]
        for target, preds, title in collection_id_list + [(year_target, year_preds, "Total")]:
            print(f"Results for {title} (size {len(target)}):")
            print(f"Mean squared error: {mean_squared_error(target, preds)}")
            print(f"Mean absolute error: {mean_absolute_error(target, preds)}")
            print(f"Explained variance: {explained_variance_score(target, preds)}")
            # print(classification_report(target, preds, digits=4, labels=list(label_dict.values()), target_names=list(label_dict.keys())))
            yd = torch.tensor([float(p - t) for p, t in zip(preds, target)])
            mean = torch.mean(yd).item()
            stdev = torch.std(yd).item()
            print(f"Year error mean: {mean}, stdev: {stdev}")

    print_results(year_target, year_preds, collection_ids)

    # sns.set_theme(style='darkgrid')
    path = Path('plots')
    path.mkdir(parents=True, exist_ok=True)
    g = sns.jointplot(x=year_target, y=year_preds, kind='hist', xlim=(1700, 1800), ylim=(1675, 1825), joint_kws=dict(binwidth=4), binwidth=4)
    g.plot_joint(sns.regplot, scatter=False, color='m')
    g.ax_joint.plot([1700, 1800], [1700, 1800], '--b', linewidth=2)
    g.fig.suptitle("Text segment year prediction - EccoBERT vs gold", fontsize=16)
    g.set_axis_labels('Correct year', 'Predicted year', fontsize=14)
    g.fig.tight_layout()
    plt.show()
    plt.savefig(path / 'year_heatmap.pdf')
    plt.close()

    g = sns.jointplot(x=year_target, y=year_difference, kind='hist', xlim=(1700, 1800), ylim=(-80, 100), joint_kws=dict(binwidth=4))
    g.fig.suptitle("EccoBERT prediction error by year", fontsize=16)
    g.set_axis_labels('Year', 'Error (years)', fontsize=14)
    g.fig.tight_layout()
    plt.show()
    plt.savefig(path / 'year_error_heatmap.pdf')
    plt.close()
    # g.plot_joint(sns.scatterplot, s=5, zorder=0)

    g = sns.jointplot(x=seq_lengths, y=[abs(e) for e in year_difference], kind='hist')
    g.plot_joint(sns.lineplot, color='m')
    g.fig.suptitle("EccoBERT prediction error by sequence length", fontsize=16)
    g.set_axis_labels('Sequence length (tokens)', 'Error (years, non-negative)', fontsize=14)
    g.fig.tight_layout()
    plt.show()
    plt.savefig(path / 'sequence_length_heatmap.pdf')
    plt.close()

    # Document level prediction
    doc_dataset = EccoDocumentLevelDataset(segments_per_document=None, input_size=input_size, min_segment_size=min_segment_size, tokenizer=tokenizer, label_dict=label_dict_full, file_names=[args.eval], shuffle=False, keep_structure=args.keep_structure)
    doc_dataloader = torch.utils.data.DataLoader(doc_dataset, collate_fn=lambda x: collate(x, tokenizer.pad_token_id), batch_size=batch_size, num_workers=1, pin_memory=True)

    # model.cuda()   
    print("Evaluating on document level.")
    with torch.no_grad():
        year_preds = []
        year_target = []
        document_ids = []
        collection_ids = []
        word_count = []
        subword_count = []
        words = []
        avg_last_4_layers = []
        for batch in doc_dataloader:
            subword_count.append(sum((e != tokenizer.pad_token) for e in tokenizer.convert_ids_to_tokens(t)) for t in batch['input_ids'])
            output = model({k: v if type(v) == list else v.cuda() for k, v in batch.items()})
            year_preds.append(output.regression)
            # TODO: Are the last 4 layers the first or last four items of the list?
            # TODO: Also average within segment.
            # Test the functionality in a Python command line prompt.
            for l, attention_mask in zip(output.hidden_states, batch['attention_mask']):
                token_embeddings = torch.stack(l[-4:], dim=0)
                token_embeddings = token_embeddings.permute(1, 2, 0, 3)
                token_embeddings = torch.mean(token_embeddings, axis=2)
                # choose only those tokens where attention mask is not zero
                token_embeddings = (token_embeddings * attention_mask.unsqueeze(-1))
                # average token embeddings to get text embeddings
                avg_last_4_layers.append(token_embeddings.sum(axis=1) / attention_mask.sum(axis=-1).unsqueeze(-1))

            year_target.append(batch['year'])
            document_ids.append(batch['document_id'])
            collection_ids.append(batch['collection_id'])
            # word_count.append(sum((not e.startswith('##')) and (e != tokenizer.cls_token) and (e != tokenizer.sep_token) for e in tokenizer.convert_ids_to_tokens(t)) for t in batch['input_ids'])
            word_count.append(sum((not e.startswith('##')) and (e != tokenizer.pad_token) for e in tokenizer.convert_ids_to_tokens(t)) for t in batch['input_ids'])
            # subword_count.append(sum((e != tokenizer.cls_token) and (e != tokenizer.sep_token) for e in tokenizer.convert_ids_to_tokens(t)) for t in batch['input_ids'])
            words.append(batch['input_ids'])

    year_preds_all = [e.item() for t in year_preds for e in t]
    year_target = [e.item() for t in year_target for e in t]
    document_ids = [e for t in document_ids for e in t]
    collection_ids = [e for t in collection_ids for e in t]
    word_count = [e for t in word_count for e in t]
    subword_count = [e for t in subword_count for e in t]
    words = [e for t in words for e in t]
    document_dict = {}
    for doc_id, col_id, pred, target, wc, sc, w, avg in zip(document_ids, collection_ids, year_preds_all, year_target, word_count, subword_count, words, averages):
        document_dict.setdefault(doc_id, []).append((pred, target, col_id, wc, sc, w, avg))

    # Filter out segments shorter than minimum segment size, unless only one segment in document
    # document_dict = {k: [t for t in v if t[4] >= min_segment_size] for k, v in document_dict.items()}
    # document_dict = {k: v for k, v in document_dict.items() if len(v) > 0}
    # print(f"Number of documents with 1 segment: {sum(len(v) == 1 for v in document_dict.values())}")

    year_preds = [[t[0] for t in l] for l in document_dict.values()]
    year_target = [[t[1] for t in l] for l in document_dict.values()]
    collection_ids = [[t[2] for t in l][0] for l in document_dict.values()]
    word_count = [sum(t[3] for t in l) for l in document_dict.values()]
    subword_count = [sum(t[4] for t in l) for l in document_dict.values()]
    words = [[t[5] for t in l] for l in document_dict.values()]
    word_sets = []
    for doc in words:
        doc = tokenizer.convert_ids_to_tokens(torch.cat(doc))
        word_set = set()
        cur_word = doc[0]
        for tok in doc[1:]:
            if tok.startswith('##'):
                cur_word += tok[2:]
            else:
                word_set.add(cur_word)
                cur_word = tok
        word_set.add(cur_word)
        word_sets.append(word_set)
    df = collections.Counter()
    for ws in word_sets:
        df.update(ws)
    idf = {k: math.log(len(word_sets)/v) for k, v in df.items()}
    print([(k, v) for k, v in list(idf.items())[:10]])
    document_lengths = [len(l) for l in year_preds]
    year_preds = [sum(l)/len(l) for l in year_preds]
    # Arithmetic mean seems to perform slightly better than median.
    # year_preds = [torch.quantile(torch.tensor(l), q=0.5).item() for l in year_preds]
    year_target = [l[0] for l in year_target]
    year_difference = [p - t for p, t in zip(year_preds, year_target)]
    word_subword_ratios = [sc / wc for wc, sc in zip(word_count, subword_count)]

    print(collection_ids[:10])
    collection_dict = {}
    for col_id, doc_length in zip(collection_ids, document_lengths):
        collection_dict.setdefault(col_id, []).append(doc_length)

    for col_id, doc_lengths in collection_dict.items():
        doc_lengths = torch.tensor([float(e) for e in doc_lengths])
        mean = torch.mean(doc_lengths).item()
        median = torch.quantile(doc_lengths, q=0.5).item()
        stdev = torch.std(doc_lengths).item()
        print(f"Collection ID: {col_id}")
        print(f"Document segment count mean: {mean}, median: {median}, stdev: {stdev}")

    print_results(year_preds, year_target, collection_ids)
    print(len(list(document_dict.keys())), len(year_preds))
    with open('model_predictions.csv', 'w') as f:
        f.write("id\tprediction\n")
        for doc_id, yp in zip(document_dict.keys(), year_preds):
            f.write(f"{doc_id}\t{yp}\n")

    g = sns.jointplot(x=year_target, y=year_preds, kind='hist', xlim=(1700, 1800), ylim=(1675, 1825), joint_kws=dict(binwidth=4), binwidth=4)
    g.plot_joint(sns.regplot, scatter=False, color='m')
    g.ax_joint.plot([1700, 1800], [1700, 1800], '--b', linewidth=2)
    # g.fig.suptitle("Document year prediction - EccoBERT vs gold", fontsize=16)
    g.set_axis_labels('Correct year', 'Predicted year', fontsize=14)
    g.fig.tight_layout()
    plt.show()
    plt.savefig(path / 'document_year_heatmap.pdf')
    plt.close()

    g = sns.jointplot(x=year_target, y=year_difference, kind='hist', xlim=(1700, 1800), ylim=(-80, 100), joint_kws=dict(binwidth=4))
    # g.fig.suptitle("EccoBERT document prediction error by year", fontsize=16)
    g.set_axis_labels('Year', 'Error (years)', fontsize=14)
    g.fig.tight_layout()
    plt.show()
    plt.savefig(path / 'document_year_error_heatmap.pdf')
    plt.close()
    # g.plot_joint(sns.scatterplot, s=5, zorder=0)

    g = sns.jointplot(x=document_lengths, y=[abs(e) for e in year_difference], kind='hist')
    g.plot_joint(sns.lineplot, color='m')
    g.ax_joint.set_xscale('log')
    g.fig.suptitle("EccoBERT prediction error by document length", fontsize=16)
    g.set_axis_labels('Document length (segments)', 'Error (years, non-negative)', fontsize=14)
    g.fig.tight_layout()
    plt.show()
    plt.savefig(path / 'document_length_heatmap.pdf')
    plt.close()

    print(f"Year target: {year_target[:5]} (length {len(year_target)}), document lengths: {document_lengths[:5]} (length {len(document_lengths)})")
    g = sns.JointGrid()
    sns.kdeplot(x=year_target, y=document_lengths, log_scale=(False, True), hue=collection_ids, ax=g.ax_joint)
    sns.histplot(x=year_target, hue=collection_ids, element='step', fill=False, legend=False, ax=g.ax_marg_x)
    sns.histplot(y=document_lengths, log_scale=True, hue=collection_ids, element='step', fill=False, legend=False, ax=g.ax_marg_y)
    # g.ax_joint.set_yscale('log')
    # g.ax_marg_y.set_yscale('log')
    g.set_axis_labels("Year", "Segment count")
    g.fig.tight_layout()
    # ax = sns.kdeplot(x=year_target, y=document_lengths, hue=collection_ids, clip=(1700, 1800), ylim=(0, None), log_scale=(False, True))
    # plt.xlabel("Year")
    # plt.ylabel("Segment count")
    plt.show()
    plt.savefig(path / 'collection_id_scatterplot.pdf')
    plt.close()

    print(f"Word counts: {word_count[:5]}, subword counts: {subword_count[:5]}, word-subword ratios: {word_subword_ratios[:5]}")
    g = sns.jointplot(x=word_subword_ratios, y=[abs(e) for e in year_difference], kind='hist')
    # g.plot_joint(sns.lineplot, color='m')
    g.plot_joint(sns.regplot, scatter=False, color='m')
    # g.ax_joint.set_xscale('log')
    g.fig.suptitle("EccoBERT prediction error by subwords-to-words ratio", fontsize=16)
    g.set_axis_labels('Subwords per word', 'Error (years, non-negative)', fontsize=14)
    g.fig.tight_layout()
    plt.show()
    plt.savefig(path / 'document_subword_word_error.pdf')
    plt.close()

    raise SystemExit

    # Prediction explanation
    def aggregate_subwords(attrs, subwords):
        result=[]
        current_subw=[]
        current_attrs=[]
        for a,s in zip(attrs,subwords):
            if s.startswith("##"):
                current_subw.append(s[2:])
                current_attrs.append(a)
            else:
                if current_subw:
                    maxval=sorted(current_attrs,key=lambda a:abs(a))[-1]
                    result.append((maxval,"".join(current_subw)))
                current_subw=[s]
                current_attrs=[a]

        if current_subw:
            maxval=sorted(current_attrs,key=lambda a:abs(a))[-1]
            result.append((maxval,"".join(current_subw)))

        return result

    def summarize_attributions(attributions):
        attributions = attributions.sum(dim=-1).squeeze(0)
        return attributions / torch.norm(attributions)

    def attribute_batch(batch):
        batch['position_ids'] = torch.tensor(range(batch['input_ids'].shape[1])).repeat(len(batch['input_ids']), 1)
        # ref_input_ids = (batch['input_ids']*((batch['input_ids'] == tokenizer.cls_token_id) | (batch['input_ids'] == tokenizer.sep_token_id))).cuda() # All nonspecial tokens replaced with [PAD]
        ref_input_ids = torch.zeros_like(batch['input_ids']).cuda() # All tokens except first and last replaced with [PAD]
        ref_input_ids[:, 0] = tokenizer.cls_token_id
        ref_input_ids[:, -1] = tokenizer.sep_token_id
        # print(batch['input_ids'][:5])
        # print(ref_input_ids[:5])
        ref_token_type_ids = batch['token_type_ids'].cuda() # Identical to input
        ref_position_ids = torch.zeros_like(batch['input_ids']).cuda() # Zeros
        ref_attention_mask = batch['attention_mask'].cuda() # Identical to input
        input_ids = batch['input_ids'].detach().clone().cuda()
        token_type_ids = batch['token_type_ids'].detach().clone().cuda()
        position_ids = batch['position_ids'].detach().clone().cuda()
        attention_mask = batch['attention_mask'].detach().clone().cuda()
        year = batch['year'].detach().clone().cuda()
        attrs = lig.attribute(inputs=(input_ids),
                              baselines=(ref_input_ids),
                              additional_forward_args=(token_type_ids, position_ids, attention_mask, year),
                              return_convergence_delta=False,
                              internal_batch_size=batch_size)
        # torch.cuda.empty_cache()
        print("Completed attribution for a batch.", flush=True)
        del ref_input_ids
        del ref_token_type_ids
        del ref_position_ids
        del ref_attention_mask
        del input_ids
        del token_type_ids
        del position_ids
        del attention_mask
        del year
        return attrs

    model.zero_grad()
    lig = LayerIntegratedGradients(lambda input_ids, token_type_ids, position_ids, attention_mask, year: model({'input_ids': input_ids, 'token_type_ids': token_type_ids, 'position_ids': position_ids, 'attention_mask': attention_mask, 'year': year}).regression, model.bert.embeddings)
    attr_list = []
    input_ids = []
    year_target = []
    # for batch in itertools.islice(doc_dataloader, 200):
    print("Starting batch attribution.")
    for batch in doc_dataloader:
        attr_list.append(attribute_batch(batch))
        input_ids.append(batch['input_ids'])
        year_target.append(batch['year'])

    print("Batch attribution complete.")
    attr_summaries = [a for b in attr_list for a in summarize_attributions(b)]
    input_ids = [tokenizer.convert_ids_to_tokens(e) for b in input_ids for e in b]
    year_target = [e.item() for t in year_target for e in t]


    document_dict = {}
    for yt, yp, doc_id, tokens, importances in zip(year_target, year_preds_all, document_ids, input_ids, attr_summaries):
        importances, words = list(zip(*aggregate_subwords(importances, tokens)))
        document_dict.setdefault(doc_id, []).append((yt, yp, words, importances))

    for doc_id, ls in document_dict.items():
        with open(path / f"{doc_id}_viz.html", 'wt') as f:
            for yt, yp, words, importances in ls:
                f.write(f"(Document ID {doc_id}) Segment prediction: {yp:.2f}, target: {yt}<br>\n")
                f.write(viz.format_word_importances(words, importances))
                f.write("<br>\n")

    print(f"Prediction: {year_preds_all[0]}, target: {year_target[0]}")

# Top features by document
    for doc_id, ls in document_dict.items():
        words = [e for l in ls for e in l[2]]
        importances = [e.item() for l in ls for e in l[3]]
        ts = sorted(list(zip(words, importances)), key=lambda x: -x[1])[:50]
        print(f"Document ID ({doc_id}), target {ls[0][0]}, prediction {sum(t[1] for t in ls)/len(ls):.2f}")
        print(' '.join(t for t, i in ts) + '\n')
        ws = [e for _, _, w, i in ls for e in sorted(list(zip(w, i)), key=lambda x: -x[1])[:10]]
        wcounts = sorted(list(collections.Counter((w for w, _ in ws)).items()), key=lambda x: -x[1]*idf[x[0]])[:50]
        print(f"{' '.join('(' + w + ', ' + str(count) + ', ' + f'{idf[w]:.4f}' + ')' for w, count in wcounts)}\n")
        groups = [(k, list(e[1] for e in g)) for k, g in itertools.groupby(sorted(list(zip(words, importances)), key=lambda x: x[0]), key=lambda x: x[0])]
        averages = sorted([(k, sum(g)/len(g), len(g)) for k, g in groups if len(g) > 1], key=lambda x: -x[1])[:50]
        print(f"{' '.join('(' + w + ', ' + f'{a:.4f}' + ', ' + str(c) + ')' for w, a, c in averages)}\n\n")

# Top features by decade
#    decade_dict = {}
#    for yp, tokens, importances in zip(year_preds_all, input_ids, attr_summaries):
#        importances, words = list(zip(*aggregate_subwords(importances, tokens)))
#        decade_dict.setdefault((max(min(yp, 1800), 1701) - 1) // 10 * 10 + 1, []).append((yp, words, importances))
#
#    for decade, ls in decade_dict.items():
#        words = [e for l in ls for e in l[1]]
#        importances = [e.item() for l in ls for e in l[2]]
#        ts = sorted(list(zip(words, importances)), key=lambda x: -x[1])[:50]
#        print(f"Decade {decade}-{decade+9}")
#        print(' '.join(t for t, i in ts) + '\n')
#        ws = [e for _, w, i in ls for e in sorted(list(zip(w, i)), key=lambda x: -x[1])[:10]]
#        wcounts = sorted(list(collections.Counter((w for w, _ in ws)).items()), key=lambda x: -x[1]*idf[x[0]])[:50]
#        print(f"{' '.join('(' + w + ', ' + str(count) + ', ' + f'{idf[w]:.4f}' + ')' for w, count in wcounts)}\n")
#        groups = [(k, list(e[1] for e in g)) for k, g in itertools.groupby(sorted(list(zip(words, importances)), key=lambda x: x[0]), key=lambda x: x[0])]
#        averages = sorted([(k, sum(g)/len(g), len(g)) for k, g in groups if len(g) > 1], key=lambda x: -x[1])[:50]
#        print(f"{' '.join('(' + w + ', ' + f'{a:.4f}' + ', ' + str(c) + ')' for w, a, c in averages)}\n\n")

#    prediction_words = {"mens", "countrey", "origin"}
#    attrs = []
#    input_ids = []
#    year_target = []
#    # Only calculate attributions for segments that contain given words
#    for batch in doc_dataloader:
#        words = [[w for _, w in aggregate_subwords(len(t)*[0], tokenizer.convert_ids_to_tokens(t))] for t in batch['input_ids']]
#        filter_list = [any(w in prediction_words for w in ws) for ws in words]
#        # filtered_batch = {k: v[filter_list] for k, v in batch.items()}
#        if any(filter_list):
#            filtered_batch = {}
#            for k, v in batch.items():
#                if type(v) == list:
#                    filtered_batch[k] = [e for e, b in zip(v, filter_list) if b]
#                else:
#                    filtered_batch[k] = v[filter_list]
#            attrs.append(attribute_batch(filtered_batch))
#            input_ids.append(filtered_batch['input_ids'])
#            year_target.append(filtered_batch['year'])
#
#    attrs = [summarize_attributions(t) for t in torch.cat(attrs)]
#    input_ids = [tokenizer.convert_ids_to_tokens(t) for t in torch.cat(input_ids)]
#    year_target = [e.item() for t in year_target for e in t]
#
#    word_dict = {k: [] for k in prediction_words}
#    for yt, tokens, importances in zip(year_target, input_ids, attrs):
#        for i, w in aggregate_subwords(importances, tokens):
#            if w in prediction_words:
#                word_dict[w].append((yt, i.item()))
#    # points = [(yt, i.item()) for yt, tokens, importances in zip(year_target, input_ids, attrs) for i, w in aggregate_subwords(importances, tokens) if w == 'revolution']
#    # points = [(yt, yp, i.item()) for doc_id, ls in document_dict.items() for yt, yp, words, importances in ls for w, i in zip(words, importances) if w == 'revolution']
#    for word, points in word_dict.items():
#        print(points[:10])
#        g = sns.jointplot(x=[yt for yt, _ in points], y=[i for _, i in points], kind='scatter', joint_kws=dict(marker='+'))
#        # g = sns.jointplot(x=[yp for _, yp, _ in points], y=[i for _, _, i in points], kind='scatter')
#        # g.plot_joint(sns.lineplot, color='m')
#        g.plot_joint(sns.regplot, scatter=False, color='m')
#        g.fig.suptitle(f"EccoBERT attribution of word '{word}' by predicted year", fontsize=16)
#        g.set_axis_labels('Predicted year', 'Attribution', fontsize=14)
#        g.fig.tight_layout()
#        plt.show()
#        plt.savefig(path / f'year_word_attribution_{word}.pdf')
#        plt.close()
