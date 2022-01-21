import pytorch_lightning as pl
import transformers
import torch
import argparse
import random
import collections
import itertools
import gzip
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error, explained_variance_score
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from pathlib import Path

# TODO: Implement a custom model that predicts both module and year

def load_fields(fn):
    with gzip.open(fn, 'rt') as f:
        for l in f:
            yield l.rstrip('\n').split('\t')

# Model class
class EccoBERTPredict(pl.LightningModule):
    def __init__(self, bert_model, num_labels, steps_train, lr, mean, stdev, vocab_size):
        super().__init__()
        self.save_hyperparameters()
        self.bert = transformers.BertModel.from_pretrained(bert_model)
        self.bert.resize_token_embeddings(vocab_size) # If additional special tokens are added, the token embedding matrix must be resized.
        self.cls_layer = torch.nn.Linear(self.bert.config.hidden_size, num_labels)
        self.reg_layer = torch.nn.Linear(self.bert.config.hidden_size, 1)
        # self.relu1 = torch.nn.ReLU()
        # self.ff1 = torch.nn.Linear(128, 128)
        # self.tanh1 = torch.nn.Tanh()
        # self.ff2 = torch.nn.Linear(128, 1)
        self.steps_train = steps_train
        self.lr = lr
        self.mean = mean
        self.stdev = stdev
        self.accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()

    def forward(self, batch):
        enc = self.bert(input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        token_type_ids=batch['token_type_ids'])
        # module_logits = self.cls_layer(enc['last_hidden_state'][:, 0])
        # TODO: Find a better solution to slow convergence
        # Try exponential output activation, or sigmoid or tanh with scaling
        # year_regression = torch.exp(self.reg_layer(enc['last_hidden_state'][:, 0])[:, 0])
        year_regression = self.reg_layer(enc['last_hidden_state'][:, 0])[:, 0] * self.stdev + self.mean
        # year_regression = self.reg_layer(enc['last_hidden_state'][:, 0])
        # year_regression = self.relu1(year_regression)
        # year_regression = self.ff1(year_regression)
        # year_regression = self.tanh1(year_regression)
        # year_regression = self.ff2(year_regression)[:, 0]
        # print(enc['last_hidden_state'].size(), module_logits.size(), batch['module'].size(), year_regression.size())
        # module_loss = torch.nn.functional.cross_entropy(module_logits, batch['module'])
        year_loss = torch.nn.functional.mse_loss(year_regression, batch['year'])
        # loss = module_loss + year_loss
        # Out = collections.namedtuple('Out', 'logits regression loss')
        Out = collections.namedtuple('Out', 'regression loss')
        # return Out(module_logits, year_regression, loss)
        return Out(year_regression, year_loss)

    def training_step(self, batch, batch_idx):
        output = self(batch)
        # self.accuracy(output.logits.softmax(-1), batch['module'])
        self.log('loss', output.loss)
        # self.log('train_acc', self.accuracy, prog_bar=True, on_step=True, on_epoch=True)
        return output.loss

    def validation_step(self, batch, batch_idx):
        output = self(batch)
        self.log('val_loss', output.loss, prog_bar=True)
        # self.val_accuracy(output.logits.softmax(-1), batch['module'])
        # self.log('val_acc', self.val_accuracy, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = transformers.optimization.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        scheduler = transformers.optimization.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(self.steps_train*0.1), num_training_steps=self.steps_train)
        scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
        return [optimizer], [scheduler]

# Dataset
def group_text_gen(fnames, keep_structure):
    for fname in fnames:
#        if keep_structure:
#            lines = load_fields(fname)
#            header = lines[0]
#            next_header = False
#            group_text = [header] # Include first header of every document in output
#            for line in lines:
#                if next_header:
#                    if line != header:
#                        yield group_text
#                        header = line
#                        group_text = [header]
#                    next_header = False
#                elif line == ['']:
#                    group_text.append('[ECCO_SB]')
#                    next_header = True
#                else:
#                    group_text.append(line + '[ECCO_LB]')
#        else:
            group_text = []
            for text in load_fields(fname):
                if text == ['']:
                    # print(group_text, flush=True)
                    yield group_text
                    group_text = []
                else:
                    group_text.append(text)

# From https://docs.python.org/3/library/itertools.html#itertools-recipes
def sliding_window(iterable, n):
    # sliding_window('ABCDEFG', 4) -> ABCD BCDE CDEF DEFG
    it = iter(iterable)
    window = collections.deque(itertools.islice(it, n), maxlen=n)
    if len(window) == n:
        yield tuple(window)
    for x in it:
        window.append(x)
        yield tuple(window)

def segment(tokenizer, input_size, min_segment_size, group_texts):
    # min_len = input_size // 4
    # max_len = input_size // 2
    min_len = min_segment_size - 2
    max_len = input_size - 2
    for text in group_texts:
        header = text[0]
        body = ' '.join([t[0] for t in text[1:]])
        t_ids = tokenizer(body, add_special_tokens=False)['input_ids']
        t_tokens = tokenizer.convert_ids_to_tokens(t_ids)
        i = 0
        while(i < len(t_ids)):
            # Find all period-space-capital formations within the target range
            matches = [j for j, (p, u) in enumerate(sliding_window(t_tokens[i+min_len-1:i+max_len], 2)) if p[-1] == '.' and u[0].isupper()]
            if matches:
                split = random.choice(matches) + i + min_len
            else:
                split = random.randrange(i+min_len, i+max_len+1)
            yield header, t_ids[i:split]
            i = split

def to_example(header, segment, input_size, label_dict, cls, sep):
    document_id, collection_id, module, year, uncertain = header
    if year:
        year = int(year)
    else:
        year = None
    return {'input_ids': [cls] + segment + [sep], 'token_type_ids': [0]*(len(segment)+2), 'attention_mask': [1]*(len(segment)+2), 'document_id': document_id, 'collection_id': collection_id, 'module': label_dict[module], 'year': year, 'uncertain': uncertain == 'TRUE'}

def chunk_shuffle(iterable, chunk_size):
    args = [iter(iterable)] * chunk_size
    chunks = (list(chunk) for chunk in zip(*args))
    for chunk in chunks:
        random.shuffle(chunk)
        yield from chunk

# Cycle a list, with each cycle being a random permutation of the list.
def random_permutation_cycle(ls):
    while True:
        shuffled = random.sample(ls, len(ls))
        for e in shuffled:
            yield e

# class EccoDataset(torch.utils.data.IterableDataset):
#     def __init__(self, tokenizer, label_dict, input_size, min_segment_size, file_names, shuffle):
#         super().__init__()
#         self.tokenizer = tokenizer
#         self.label_dict = label_dict
#         self.input_size = input_size
#         self.min_segment_size = min_segment_size
#         self.file_names = file_names
#         self.shuffle = shuffle
#         self.gpu_id = None # Set in a separate method to allow initialization without GPU information
# 
#     def __iter__(self):
#         worker_info = torch.utils.data.get_worker_info()
#         file_name_slice = itertools.islice(self.file_names, worker_info.id, None, worker_info.num_workers)
#         self.group_text_gen = group_text_gen(self.gpu_id, file_name_slice)
#         example_gen = (to_example(header=header, segment=segment, input_size=self.input_size, label_dict=self.label_dict, cls=self.tokenizer.cls_token_id, sep=self.tokenizer.sep_token_id) for header, segment in segment(self.tokenizer, self.input_size, self.min_segment_size, self.group_text_gen))
#         filtered = (example for example in example_gen if example['year'] != None) # Filter out documents with missing years
#         return chunk_shuffle(filtered, 10000) if self.shuffle else filtered
# 
#     def set_gpu_id(self, gpu_id):
#         self.gpu_id = gpu_id

class EccoDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, label_dict, input_size, min_segment_size, file_names, dataset_size, shuffle, keep_structure):
        super().__init__()
        self.tokenizer = tokenizer
        self.label_dict = label_dict
        self.input_size = input_size
        self.min_segment_size = min_segment_size
        self.shuffle = shuffle
        self.keep_structure = keep_structure

        example_gen = (to_example(header=header, segment=segment, input_size=self.input_size, label_dict=self.label_dict, cls=self.tokenizer.cls_token_id, sep=self.tokenizer.sep_token_id) for header, segment in segment(self.tokenizer, self.input_size, self.min_segment_size, group_text_gen(file_names, self.keep_structure)))
        filtered = (example for example in example_gen if example['year'] != None and len(example['input_ids']) >= self.min_segment_size) # Filter out segments that are shorter than minimum size or don't have a year
        # filtered = (example for example in example_gen if example['year'] != None and len(example['input_ids']) >= self.min_segment_size and example['collection_id'] == 'ecco2')
        # filtered = (example for example in example_gen if example['year'] != None)
        self.data_list = list(itertools.islice(filtered, dataset_size))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

# A Dataset class for taking equally sized samples of documents.
class EccoDocumentLevelDataset(EccoDataset):
    def __init__(self, segments_per_document, **args):
        super().__init__(dataset_size=None, **args)
        data_dict = {}
        for segment in self.data_list:
            data_dict.setdefault(segment['document_id'], []).append(segment)

        document_lengths = [len(v) for v in data_dict.values()]
        print(f"Total number of segments: {len(self.data_list)}, number of documents: {len(data_dict)}, shortest document: {min(document_lengths)} segments, longest document: {max(document_lengths)} segments")

        if segments_per_document:
            print(f"Clipping to length: {segments_per_document}")
            data_dict = {k: v[:segments_per_document] for k, v in data_dict.items()}

        self.data_list = [s for l in list(data_dict.values()) for s in l]
        print(f"Total number of segments after clipping: {len(self.data_list)}")

def pad_with_value(vals, padding_value):
    vals=[torch.LongTensor(v) for v in vals]
    return torch.nn.utils.rnn.pad_sequence(vals, batch_first=True, padding_value=padding_value)

def collate(itemlist, pad_token_id):
    batch = {}

    for k in ['input_ids', 'token_type_ids', 'attention_mask']:
        batch[k] = pad_with_value([item[k] for item in itemlist], pad_token_id)

    for k, dtype in [('module', torch.long), ('year', torch.float), ('uncertain', torch.long)]:
        batch[k] = torch.tensor([item[k] for item in itemlist], dtype=dtype)

    for k in 'document_id', 'collection_id':
        batch[k] = [item[k] for item in itemlist]

    return batch

class EccoDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, tokenizer, input_size, min_segment_size, label_dict, train_files, dev_fname, train_size, dev_size, keep_structure):
        super().__init__()
        self.batch_size = batch_size
        self.train_files = train_files
        self.dev_fname = dev_fname
        self.tokenizer = tokenizer
        self.input_size = input_size
        self.min_segment_size = min_segment_size
        self.collate_fn = lambda x: collate(x, self.tokenizer.pad_token_id)
        self.label_dict = label_dict
        self.train_size = train_size
        self.dev_size = dev_size
        self.keep_structure = keep_structure

    def setup(self, stage=None):
        print(f"Train size: {self.train_size}, dev size: {self.dev_size}")
        self.train_data = EccoDataset(input_size=self.input_size, min_segment_size=self.min_segment_size, tokenizer=self.tokenizer, label_dict=self.label_dict, file_names=train_files, dataset_size=self.train_size, shuffle=True, keep_structure=self.keep_structure)
        self.dev_data = EccoDataset(input_size=self.input_size, min_segment_size=self.min_segment_size, tokenizer=self.tokenizer, label_dict=self.label_dict, file_names=[self.dev_fname], dataset_size=self.dev_size, shuffle=False, keep_structure=self.keep_structure)
        years = torch.tensor([float(d['year']) for d in self.train_data.data_list])
        self.mean = torch.mean(years).item()
        self.stdev = torch.std(years).item()
        print(f"Training data year mean: {self.mean:.2f}, stdev: {self.stdev:.2f}")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data, collate_fn=self.collate_fn, batch_size=self.batch_size, num_workers=1, pin_memory=True, sampler=torch.utils.data.distributed.DistributedSampler(self.train_data, shuffle=True))

    def val_dataloader(self): 
        # Note that the evaluation dataset isn't deterministic due to the random segment splits.
        return torch.utils.data.DataLoader(self.dev_data, collate_fn=self.collate_fn, batch_size=self.batch_size, num_workers=1, pin_memory=True)

class CheckpointSteps(pl.callbacks.Callback):
    def __init__(self, out_dir, every_n_steps):
        self.out_dir = out_dir
        self.every_n_steps = every_n_steps

    def on_train_batch_end(self, trainer, *args):
        if (trainer.global_step + 1) % self.every_n_steps == 0:
            trainer.save_checkpoint(self.out_dir + f'/step-{trainer.global_step+1}.ckpt')

class CheckpointTrainEnd(pl.callbacks.Callback):
    def __init__(self, out_dir):
        self.out_dir = out_dir

    def on_train_end(self, trainer, *args):
        trainer.save_checkpoint(self.out_dir + f'/epoch-{trainer.current_epoch+1}.ckpt')

# Training
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tokenizer', help="Path to the tokenizer vocabulary file.")
    parser.add_argument('bert_path', help="Path to the pretrained BERT model.")
    parser.add_argument('eval', help="A gzip compressed TSV file containing the evaluation texts.")
    parser.add_argument('out_dir', help="A directory to which the finetuned model is saved.")
    parser.add_argument('--train', help="A gzip compressed TSV file containing the training texts.")
    parser.add_argument('--load_checkpoint', help="A path to a checkpoint file to load.")
    parser.add_argument('--gpus', type=int, default=1, help="Number of GPUs per node")
    parser.add_argument('--nodes', type=int, default=1, help="Number of nodes")
    parser.add_argument('--keep_structure', action='store_true', help="Inform the model of the subdocument and linebreak structure with special tokens. Header line is expected as the first line of every subdocument.")

    args = parser.parse_args()

    lr = 1e-5
    max_steps = None
    eval_batches = 500
    gpus = args.gpus
    num_nodes = args.nodes
    train_files = [args.train]
    random.shuffle(train_files)

    additional_special_tokens = ['[ECCO_SB]', '[ECCO_LB]'] if args.keep_structure else [] 
    tokenizer = transformers.BertTokenizerFast.from_pretrained(args.tokenizer, additional_special_tokens=additional_special_tokens)
    input_size = 512
    min_segment_size = input_size
    batch_size = 24
    accumulate_grad_batches = 1

    # General Reference
    # History and Geography
    # Law
    # Literature and Language
    # Medicine, Science and Technology
    # Religion and Philosophy
    # Social Sciences and Fine Arts
    label_dict = {'General Reference': 0, 'History and Geography': 1, 'Law': 2, 'Literature and Language': 3, 'Medicine, Science and Technology': 4, 'Religion and Philosophy': 5, 'Social Sciences and Fine Arts': 6}
    label_dict_full = {**label_dict, **{'Science, Medicine and Technology': 4, 'Fine Arts': 6, 'Social Sciences': 6}}

    data = EccoDataModule(batch_size=batch_size, tokenizer=tokenizer, input_size=input_size, min_segment_size=min_segment_size, label_dict=label_dict_full, train_files=train_files, dev_fname=args.eval, train_size=max_steps if not max_steps else int(max_steps*batch_size), dev_size=eval_batches*batch_size, keep_structure=args.keep_structure)
    print(f"Number of nodes {num_nodes}, GPUs per node {gpus}, batch size {batch_size}, accumulate_grad_batches {accumulate_grad_batches}, learning rate {lr}, training steps {max_steps}")
    print(f"Tokenizer {args.tokenizer} ({'fast' if tokenizer.is_fast else 'slow'}), input size {input_size}, minimum segment size {min_segment_size}")
    print(f"Model {args.bert_path}, {'keeping newline structure' if args.keep_structure else 'not keeping newline structure'}")

    data.setup()

    if args.load_checkpoint:
        model = EccoBERTPredict.load_from_checkpoint(checkpoint_path=args.load_checkpoint)
        print(f"Model loaded from checkpoint: {args.load_checkpoint}")
    else:
        steps_train = max_steps
        model = EccoBERTPredict(bert_model=args.bert_path, num_labels=len(label_dict), lr=lr, steps_train=len(data.train_data), mean=data.mean, stdev=data.stdev, vocab_size=len(tokenizer))

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')

    # train BERT and evaluate
    trainer = pl.Trainer(
        num_nodes=num_nodes,
        gpus=gpus,
        auto_select_gpus=True,
        accelerator='ddp',
        precision=16,
        val_check_interval=1000,
        # val_check_interval=0.0,
        # limit_val_batches=500,
        # val_check_interval=1.0,
        num_sanity_val_steps=5,
        # max_steps=max_steps,
        max_epochs=1,
        accumulate_grad_batches=accumulate_grad_batches,
        progress_bar_refresh_rate=50, # Large value prevents crashing in colab
        # callbacks=[CheckpointSteps(out_dir=args.out_dir, every_n_steps=max_steps), lr_monitor],
        callbacks=[CheckpointTrainEnd(out_dir=args.out_dir), lr_monitor],
        checkpoint_callback=False,
        resume_from_checkpoint=args.load_checkpoint
    )

    print(f"Train file: {args.train}")
    if args.train:
        trainer.fit(model, datamodule=data)

    # one-hot encoding of words
    train_batches = list(itertools.islice(data.train_dataloader(), 100))
    eval_batches = [batch for batch in data.val_dataloader()]
    train_encoded = [' '.join(tokenizer.convert_ids_to_tokens(e)) for batch in train_batches for e in batch['input_ids']]
    train_labels = [e for batch in train_batches for e in batch['year']]
    eval_encoded = [' '.join(tokenizer.convert_ids_to_tokens(e)) for batch in eval_batches for e in batch['input_ids']]
    eval_labels = [e for batch in eval_batches for e in batch['year']]
   
    train_avg = torch.mean(torch.tensor(train_labels))
    print(f"Train average: {train_avg:.2f}")
    avg_baseline_preds = len(eval_labels)*[train_avg]
    print(f"Average baseline mean squared error: {mean_squared_error(eval_labels, avg_baseline_preds)}")
    print(f"Average baseline explained variance: {explained_variance_score(eval_labels, avg_baseline_preds)}")
 
    vectorizer = CountVectorizer()
    vectorizer.fit(train_encoded)
    train_data = vectorizer.transform(train_encoded)
    eval_data = vectorizer.transform(eval_encoded)
    regr = svm.SVR()
    regr.fit(train_data, train_labels)
    svr_pred = regr.predict(eval_data)
    # print(f"Year predictions: {[f'{i:.2f}' for i in year_preds_display]}")
    # print(f"Year target: {year_target_display}")
    print(f"SVR mean squared error: {mean_squared_error(eval_labels, svr_pred)}")
    print(f"SVR explained variance: {explained_variance_score(eval_labels, svr_pred)}")

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
    
    print("Evaluating on document level.")
    with torch.no_grad():
        year_preds = []
        year_target = []
        document_ids = []
        collection_ids = []
        for batch in doc_dataloader:
            output = model({k: v if type(v) == list else v.cuda() for k, v in batch.items()})
            year_preds.append(output.regression)
            year_target.append(batch['year'])
            document_ids.append(batch['document_id'])
            collection_ids.append(batch['collection_id'])

    year_preds = [e.item() for t in year_preds for e in t]
    year_target = [e.item() for t in year_target for e in t]
    document_ids = [e for t in document_ids for e in t]
    collection_ids = [e for t in collection_ids for e in t]
    document_dict = {}
    for doc_id, col_id, pred, target in zip(document_ids, collection_ids, year_preds, year_target):
        document_dict.setdefault(doc_id, []).append((pred, target, col_id))

    year_preds = [[t[0] for t in l] for l in document_dict.values()]
    year_target = [[t[1] for t in l] for l in document_dict.values()]
    collection_ids = [[t[2] for t in l][0] for l in document_dict.values()]
    document_lengths = [len(l) for l in year_preds]
    year_preds = [sum(l)/len(l) for l in year_preds]
    year_target = [l[0] for l in year_target]
    year_difference = [p - t for p, t in zip(year_preds, year_target)]

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

    g = sns.jointplot(x=year_target, y=year_preds, kind='hist', xlim=(1700, 1800), ylim=(1675, 1825), joint_kws=dict(binwidth=4), binwidth=4)
    g.plot_joint(sns.regplot, scatter=False, color='m')
    g.ax_joint.plot([1700, 1800], [1700, 1800], '--b', linewidth=2)
    g.fig.suptitle("Document year prediction - EccoBERT vs gold", fontsize=16)
    g.set_axis_labels('Correct year', 'Predicted year', fontsize=14)
    g.fig.tight_layout()
    plt.show()
    plt.savefig(path / 'document_year_heatmap.pdf')
    plt.close()

    g = sns.jointplot(x=year_target, y=year_difference, kind='hist', xlim=(1700, 1800), ylim=(-80, 100), joint_kws=dict(binwidth=4))
    g.fig.suptitle("EccoBERT document prediction error by year", fontsize=16)
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
