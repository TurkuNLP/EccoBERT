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

from sklearn.metrics import classification_report, mean_squared_error, explained_variance_score
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
    def __init__(self, bert_model, num_labels, steps_train, lr):
        super().__init__()
        self.save_hyperparameters()
        self.bert = transformers.BertModel.from_pretrained(bert_model)
        self.cls_layer = torch.nn.Linear(self.bert.config.hidden_size, num_labels)
        self.reg_layer = torch.nn.Linear(self.bert.config.hidden_size, 1)
        # self.relu1 = torch.nn.ReLU()
        # self.ff1 = torch.nn.Linear(128, 128)
        # self.tanh1 = torch.nn.Tanh()
        # self.ff2 = torch.nn.Linear(128, 1)
        self.steps_train = steps_train
        self.lr = lr
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
        year_regression = (torch.tanh(self.reg_layer(enc['last_hidden_state'][:, 0])[:, 0]) * 250) + 1760
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
def group_text_gen(fnames):
    for fname in fnames:
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
    min_len = min_segment_size
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
                split = random.randrange(i+min_len, i+max_len)
            yield header, t_ids[i:split]
            i = split

def to_example(header, segment, input_size, label_dict, cls, sep):
    document_id, collection_id, module, year, uncertain = header
    if year:
        year = int(year)
    else:
        year = None
    return {'input_ids': [cls] + segment + [sep], 'token_type_ids': [0]*(len(segment)+2), 'attention_mask': [1]*(len(segment)+2), 'module': label_dict[module], 'year': year, 'uncertain': uncertain == 'TRUE'}

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
    def __init__(self, tokenizer, label_dict, input_size, min_segment_size, file_names, dataset_size):
        super().__init__()
        self.tokenizer = tokenizer
        self.label_dict = label_dict
        self.input_size = input_size
        self.min_segment_size = min_segment_size

        example_gen = (to_example(header=header, segment=segment, input_size=self.input_size, label_dict=self.label_dict, cls=self.tokenizer.cls_token_id, sep=self.tokenizer.sep_token_id) for header, segment in segment(self.tokenizer, self.input_size, self.min_segment_size, group_text_gen(file_names)))
        filtered = (example for example in example_gen if example['year'] != None) # Filter out documents with missing years
        self.data_list = list(itertools.islice(filtered, dataset_size))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

def pad_with_value(vals, padding_value):
    vals=[torch.LongTensor(v) for v in vals]
    return torch.nn.utils.rnn.pad_sequence(vals, batch_first=True, padding_value=padding_value)

def collate(itemlist, pad_token_id):
    batch = {}

    for k in ['input_ids', 'token_type_ids', 'attention_mask']:
        batch[k] = pad_with_value([item[k] for item in itemlist], pad_token_id)

    for k, dtype in [('module', torch.long), ('year', torch.float), ('uncertain', torch.long)]:
        batch[k] = torch.tensor([item[k] for item in itemlist], dtype=dtype)

    return batch

class EccoDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, tokenizer, input_size, min_segment_size, label_dict, train_files, dev_fname, train_size, dev_size):
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

    def setup(self, stage=None):
        print(f"Train size: {self.train_size}, dev size: {self.dev_size}")
        self.train_data = EccoDataset(input_size=self.input_size, min_segment_size=self.min_segment_size, tokenizer=self.tokenizer, label_dict=self.label_dict, file_names=train_files, dataset_size=self.train_size)
        self.dev_data = EccoDataset(input_size=self.input_size, min_segment_size=self.min_segment_size, tokenizer=self.tokenizer, label_dict=self.label_dict, file_names=[self.dev_fname], dataset_size=self.dev_size)
        years = torch.tensor([float(d['year']) for d in self.train_data.data_list])
        self.mean = torch.mean(years).item()
        self.std = torch.std(years).item()
        print(f"Training data year mean: {self.mean:.2f}, stdev: {self.std:.2f}")

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

    args = parser.parse_args()

    lr = 1e-5
    max_steps = 4e4
    eval_batches = 5000
    gpus = args.gpus
    num_nodes = args.nodes
    train_files = [args.train]
    random.shuffle(train_files)

    tokenizer = transformers.BertTokenizerFast.from_pretrained(args.tokenizer)
    input_size = 128
    min_segment_size = input_size // 2
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

    data = EccoDataModule(batch_size=batch_size, tokenizer=tokenizer, input_size=input_size, min_segment_size=min_segment_size, label_dict=label_dict_full, train_files=train_files, dev_fname=args.eval, train_size=int(max_steps*batch_size), dev_size=eval_batches*batch_size)
    print(f"Number of nodes {num_nodes}, GPUs per node {gpus}, batch size {batch_size}, accumulate_grad_batches {accumulate_grad_batches}, learning rate {lr}, training steps {max_steps}")
    print(f"Tokenizer {args.tokenizer} ({'fast' if tokenizer.is_fast else 'slow'}), input size {input_size}, minimum segment size {min_segment_size}")
    print(f"Model {args.bert_path}")

    data.setup()

    if args.load_checkpoint:
        model = EccoBERTPredict.load_from_checkpoint(checkpoint_path=args.load_checkpoint)
        print(f"Model loaded from checkpoint: {args.load_checkpoint}")
    else:
        steps_train = max_steps
        model = EccoBERTPredict(bert_model=args.bert_path, num_labels=len(label_dict), lr=lr, steps_train=steps_train)

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')

    # train BERT and evaluate
    trainer = pl.Trainer(
        num_nodes=num_nodes,
        gpus=gpus,
        auto_select_gpus=True,
        accelerator='ddp',
        precision=16,
        val_check_interval=1000,
        # limit_val_batches=500,
        # val_check_interval=1.0,
        num_sanity_val_steps=5,
        max_steps=max_steps,
        accumulate_grad_batches=accumulate_grad_batches,
        progress_bar_refresh_rate=50, # Large value prevents crashing in colab
        callbacks=[CheckpointSteps(out_dir=args.out_dir, every_n_steps=40000), lr_monitor],
        checkpoint_callback=False,
        resume_from_checkpoint=args.load_checkpoint
    )

    print(args.train)
    if args.train:
        trainer.fit(model, datamodule=data)

    # one-hot encoding of words
    train_batches = list(itertools.islice(data.train_dataloader(), 1000))
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
        for batch in batches:
            output = model({k: v.cuda() for k, v in batch.items()})
            # preds.append(output.logits.argmax(-1))
            # labels.append(batch['module'])
            year_preds.append(output.regression)
            year_target.append(batch['year'])

    # preds = [e.item() for t in preds for e in t]
    # target = [e.item() for t in labels for e in t]
    year_preds = [e.item() for t in year_preds for e in t]
    year_target = [e.item() for t in year_target for e in t]
    # print(f"Predictions: {preds[:10]}")
    # print(f"Target: {target[:10]}")
    year_preds_display, year_target_display = list(zip(*random.sample(list(zip(year_preds, year_target)), 50)))
    print(f"Year predictions: {[f'{i:.2f}' for i in  year_preds_display]}")
    print(f"Year target: {year_target_display}")
    print(f"Mean squared error: {mean_squared_error(year_target, year_preds)}")
    print(f"Explained variance: {explained_variance_score(year_target, year_preds)}")
    # print(classification_report(target, preds, digits=4, labels=list(label_dict.values()), target_names=list(label_dict.keys())))

    year_difference = [p - t for p, t in zip(year_preds, year_target)]
    yd = torch.tensor([float(n) for n in year_difference])
    mean = torch.mean(yd).item()
    stdev = torch.std(yd).item()
    print(f"Year error mean: {mean}, stdev: {stdev}")

    # sns.set_theme(style='darkgrid')
    path = Path('plots')
    path.mkdir(parents=True, exist_ok=True)
    g = sns.jointplot(x=year_target, y=year_preds, kind='hist', xlim=(1700, 1800), ylim=(1675, 1825), binwidth=4)
    g.plot_joint(sns.regplot, scatter=False, color='m')
    g.ax_joint.plot([1700, 1800], [1700, 1800], '--b', linewidth=2)
    g.fig.suptitle("Text segment year prediction - EccoBERT vs gold", fontsize=16)
    g.set_axis_labels('Correct year', 'Predicted year', fontsize=14)
    g.fig.tight_layout()
    plt.show()
    plt.savefig(path / 'year_heatmap.pdf')
    plt.close()

    g = sns.jointplot(x=year_target, y=year_difference, kind='hist', xlim=(1700, 1800), ylim=(-80, 100))
    g.fig.suptitle("EccoBERT prediction error by year", fontsize=16)
    g.set_axis_labels('Year', 'Error (years)', fontsize=14)
    g.fig.tight_layout()
    plt.show()
    plt.savefig(path / 'year_error_heatmap.pdf')
    plt.close()
    # g.plot_joint(sns.scatterplot, s=5, zorder=0)
