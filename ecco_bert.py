import pytorch_lightning as pl
import transformers
import torch
import argparse
import random
import collections
import itertools
import gzip
import tqdm 
import sys
import math

from sklearn.metrics import classification_report

def pad(l, size):
    return l + [0]*(size - len(l))

def transpose(l):
  return [list(t) for t in zip(*l)]

def load_fields(fn):
    with gzip.open(fn, 'rt') as f:
        return transpose([l.rstrip('\n').split('\t') for l in f])

class EccoBERT(pl.LightningModule):

    def __init__(self, model_type, vocab_size, lr, steps_train=None):
        super().__init__()
        self.save_hyperparameters()
        self.steps_train = steps_train
        args = {}
        if model_type == 'bert':
            model_name = "TurkuNLP/bert-base-finnish-cased-v1"
            config_class = transformers.BertConfig
            model_class = transformers.BertForPreTraining
        elif model_type == 'bert-very-large':
            model_name = 'bert-large-cased'
            config_class = transformers.BertConfig
            model_class = transformers.BertForPreTraining
            args = {'hidden_size': 1032, 'intermediate_size': 4096, 'num_attention_heads': 24, 'num_hidden_layers': 40}
            print(args)
        elif model_type == 'bigbird':
            model_name = "google/bigbird-roberta-base"
            config_class = transformers.BigBirdConfig
            model_class = transformers.BigBirdForPreTraining
        
        configuration = config_class.from_pretrained(model_name, vocab_size=vocab_size, **args)
        self.model = model_class(configuration)
        self.lr = lr
        # self.accuracy = pl.metrics.Accuracy()
        # self.val_accuracy = pl.metrics.Accuracy()

    def forward(self,batch):
        return self.model(input_ids=batch['input_ids'],
                          attention_mask=batch['attention_mask'],
                          token_type_ids=batch['token_type_ids'],
                          labels=batch['label'],
                          next_sentence_label=batch['next_sentence_label']) #BxS_LENxSIZE; BxSIZE

    def training_step(self,batch,batch_idx):
        outputs = self(batch)
        # self.accuracy(y_hat, batch["label"])
        # self.log("train_acc", self.accuracy, prog_bar=True, on_step=True, on_epoch=True)
        # self.log("linear_scheduler", )
        self.log("loss", outputs.loss)
        # loss_function = torch.nn.CrossEntropyLoss()
        # self.log('masked_lm_loss', loss_function(outputs.prediction_logits.view(-1, self.bert.config.vocab_size), batch['label'].view(-1)))
        # self.log('next_sentence_loss', loss_function(outputs.seq_relationship_logits.view(-1, 2), batch['next_sentence_label'].view(-1)))
        return outputs.loss

    def validation_step(self,batch,batch_idx):
        outputs = self(batch)
        self.log("val_loss", outputs.loss, prog_bar=True)
        # TODO: Calculate whole word accuracy

        # loss_function = torch.nn.CrossEntropyLoss()
        # self.log('val_masked_lm_loss', loss_function(outputs.prediction_logits.view(-1, self.bert.config.vocab_size), batch['label'].view(-1)), prog_bar=True)
        # self.log('val_next_sentence_loss', loss_function(outputs.seq_relationship_logits.view(-1, 2), batch['next_sentence_label'].view(-1)), prog_bar=True)
        # self.log("val_acc", self.val_accuracy, prog_bar=True, on_epoch=True)
        # return outputs

    def configure_optimizers(self):
        optimizer = transformers.optimization.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        scheduler = transformers.optimization.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10000, num_training_steps=self.steps_train)
        scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
        return [optimizer], [scheduler]

#class EccoBigBird(EccoBERT):
#    def __init__(self, block_size, **args):
#        super().__init__(**args)
#        self.save_hyperparameters()
#        self.model = transformers.BigBirdForPreTraining(args['configuration'])
#        self.block_size = block_size
#
#    def forward(self, batch):
#        return super().forward(aligned)

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

# Segment a list of strings into tokenized chunks, with a maximum length of 75 % of the maximum input size.
# Segments shorter than 128 tokens are generated only at end-of-document positions.
def segment(tokenizer, input_size, group_texts):
    min_len = 128
    max_len = math.floor(0.75*input_size)
    for text in group_texts:
        t_ids = tokenizer(text, add_special_tokens=False)['input_ids']
        t_tokens = tokenizer.convert_ids_to_tokens(t_ids)
        i = 0
        while(i < len(t_ids)):
            # Find all period-space-capital formations within the target range
            matches = [j for j, (p, u) in enumerate(sliding_window(t_tokens[i+min_len-1:i+max_len], 2)) if p[-1] == '.' and u[0].isupper()]
            if matches:
                split = random.choice(matches) + i + min_len
            else:
                split = random.randrange(i+min_len, i+max_len)
            yield t_ids[i:split]
            i = split

def to_example(input_size, cls, sep, window):
    w1 = window[0]
    w2, next_sentence_label = (window[1], 0) if random.random() > 0.5 else (random.choice(window[2:]), 1)
    effective_input_size = input_size - 3
    if len(w1) + len(w2) > effective_input_size:
        min_length = min(len(w1), len(w2))
        if min_length > effective_input_size / 2:
            max_allowed = effective_input_size / 2
        else:
            max_allowed = effective_input_size - min_length
        
        # Trim from opposite ends to keep the sentence boundary intact for NSP.
        # Use ceiling and floor in case min_length > effective_input_size / 2, to get the otherwise unused
        # last token position when effective_input_size is odd.
        # For example, input_size=512 -> effective_input_size=509 -> len(w1)=255, len(w2)=254
        w1 = w1[-math.ceil(max_allowed):]
        w2 = w2[:math.floor(max_allowed)]
        
    return {'input_ids': [cls] + w1 + [sep] + w2 + [sep], 'token_type_ids': [0]*(len(w1)+2) + [1]*(len(w2)+1), 'attention_mask': [1]*(len(w1)+len(w2)+3), 'next_sentence_label': next_sentence_label}

# TODO: Optimize performance by reading the next chunk while yielding from chunk
def chunk_shuffle(iterable, chunk_size):
    args = [iter(iterable)] * chunk_size
    chunks = (list(chunk) for chunk in zip(*args))
    for chunk in chunks:
        random.shuffle(chunk)
        yield from chunk

#def chunk_shuffle(iterable, chunk_size):
#    chunk = [e for _, e in zip(range(chunk_size), iterable)]
#    while True:
#        random.shuffle(chunk)
#        next_chunk = []
#        for i in range(chunk_size):
#            yield chunk[i]
#            try:
#                next_chunk.append(next(iterable))
#            except StopIteration:
#                yield from chunk[i+1:]
#                return
#        chunk = next_chunk

def group_text_gen(gpu_id, fnames):
    for fname in fnames:
        print(f"GPU {gpu_id} worker {torch.utils.data.get_worker_info().id} opening file {fname}", flush=True)
        group_text = []
        for text in load_fields(fname)[0]:
            if not text:
                yield ' '.join(group_text)
                group_text = []
            else:
                group_text.append(text)

# Cycle a list, with each cycle being a random permutation of the list.
def random_permutation_cycle(ls):
    while True:
        shuffled = random.sample(ls, len(ls))
        for e in shuffled:
            yield e

class EccoDataset(torch.utils.data.IterableDataset):
    def __init__(self, gpu_id, tokenizer, input_size, file_names, shuffle):
        super().__init__()
        self.tokenizer = tokenizer
        self.input_size = input_size
        self.gpu_id = gpu_id
        self.file_names = file_names
        self.shuffle = shuffle

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        file_name_slice = itertools.islice(self.file_names, worker_info.id, None, worker_info.num_workers)
        self.group_text_gen = group_text_gen(self.gpu_id, file_name_slice)
        example_gen = (to_example(input_size=self.input_size, cls=self.tokenizer.cls_token_id, sep=self.tokenizer.sep_token_id, window=window) for window in sliding_window(segment(self.tokenizer, self.input_size, self.group_text_gen), 50))
        return chunk_shuffle(example_gen, 250000) if self.shuffle else example_gen

def pad_with_value(vals, padding_value):
    vals=[torch.LongTensor(v) for v in vals]
    return torch.nn.utils.rnn.pad_sequence(vals, batch_first=True, padding_value=padding_value)

def collate(itemlist, data_collator, pad_token_id):
    batch = {}
    masked = data_collator(itemlist)

    batch['attention_mask'] = pad_with_value([item['attention_mask'] for item in itemlist], pad_token_id) # Here pad_token_id=3
    batch['token_type_ids'] = pad_with_value([item['token_type_ids'] for item in itemlist], 0) # Can't use pad_token_id since index out of bounds
    batch['input_ids'] = masked['input_ids']
    batch['label'] = masked['labels']

    batch['next_sentence_label'] = torch.tensor([item['next_sentence_label'] for item in itemlist], dtype=torch.long)
 
    return batch

# Pad batch tensors to minimum length and block size alignment.
def collate_bigbird(itemlist, data_collator, pad_token_id, block_size):
    batch = collate(itemlist, data_collator, pad_token_id)
    # TODO: Calculate minimum length from num_random_blocks of the Big Bird configuration
    block_aligned_length = max(math.ceil(batch['input_ids'].size(1) / block_size), 12) * block_size
    for k, fill in [('attention_mask', pad_token_id), ('token_type_ids', 0), ('input_ids', pad_token_id), ('label', pad_token_id)]:
        padded = torch.full([batch[k].size(0), block_aligned_length], fill)
        padded[:, :batch[k].size(1)] = batch[k]
        batch[k] = padded

    return batch

class EccoDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, tokenizer, input_size, train_files, dev_fname):
        super().__init__()
        self.batch_size = batch_size
        self.train_files = train_files
        self.dev_fname = dev_fname
        self.tokenizer = tokenizer
        self.input_size = input_size
        self.data_collator = transformers.DataCollatorForWholeWordMask(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
        self.collate_fn = lambda x: collate(x, self.data_collator, self.tokenizer.pad_token_id)

    def train_dataloader(self):
        num_devices = self.trainer.num_nodes*self.trainer.num_gpus
        intervals = [round(n * (len(self.train_files)/num_devices)) for n in range(num_devices)] + [len(self.train_files)]
        device_id = self.trainer.node_rank*self.trainer.num_gpus + self.trainer.root_gpu
        fns = random_permutation_cycle(self.train_files[intervals[device_id]:intervals[device_id+1]]) # Infinite dataset
        print(f"Node: {self.trainer.node_rank}, GPU: {self.trainer.root_gpu}, loading files between: {(intervals[device_id], intervals[device_id+1])}", flush=True)
        dataset = EccoDataset(gpu_id=self.trainer.root_gpu, input_size=self.input_size, tokenizer=self.tokenizer, file_names=fns, shuffle=True)
        return torch.utils.data.DataLoader(dataset, collate_fn=self.collate_fn, batch_size=self.batch_size, num_workers=8, pin_memory=True)

    def val_dataloader(self): 
        # Note that the evaluation dataset isn't deterministic due to the random segment splits and sliding window NSP approach.
        dataset = EccoDataset(gpu_id=self.trainer.root_gpu, input_size=self.input_size, tokenizer=self.tokenizer, file_names=[self.dev_fname], shuffle=False)
        return torch.utils.data.DataLoader(dataset, collate_fn=self.collate_fn, batch_size=self.batch_size, num_workers=1, pin_memory=True)

class EccoBigBirdDataModule(EccoDataModule):
    def __init__(self, block_size, **args):
        super().__init__(**args)
        self.collate_fn = lambda x: collate_bigbird(x, self.data_collator, self.tokenizer.pad_token_id, block_size)

class CheckpointSteps(pl.callbacks.Callback):
    def __init__(self, out_dir, every_n_steps):
        self.out_dir = out_dir
        self.every_n_steps = every_n_steps

    def on_train_batch_end(self, trainer, *args):
        if (trainer.global_step + 1) % self.every_n_steps == 0:
            trainer.save_checkpoint(self.out_dir + f'/step-{trainer.global_step+1}.ckpt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tokenizer', help="Path to the tokenizer vocabulary file.")
    parser.add_argument('train', help="A text file containing paths to the training files.")
    parser.add_argument('eval', help="A TSV file containing the full unannotated evaluation texts.")
    parser.add_argument('out_dir', help="A directory to which the model is saved.")
    parser.add_argument('--model', default='bert', help="The type of model to use ('bert' or 'bigbird')")
    parser.add_argument('--load_checkpoint', help="A path to a checkpoint file to load.")
    parser.add_argument('--only-loop-data', action="store_true", default=False, help="Just loop over the data do nothing else")
    parser.add_argument('--gpus', type=int, default=1, help="Number of GPUs per node")
    parser.add_argument('--nodes', type=int, default=1, help="Number of nodes")

    args = parser.parse_args()

    lr = 6e-5
    max_steps = 1e6
    # max_steps = 1500
    gpus = args.gpus
    num_nodes = args.nodes
    train_files = [l.rstrip('\n') for l in open(args.train).readlines()]
    random.shuffle(train_files)

    if args.model == 'bert' or args.model == 'bert-very-large':
        tokenizer = transformers.BertTokenizerFast.from_pretrained(args.tokenizer)
        input_size = 512
        batch_size = 24 if args.model == 'bert' else 3
        accumulate_grad_batches = 4 if args.model == 'bert' else 32
        data = EccoDataModule(batch_size=batch_size, tokenizer=tokenizer, input_size=input_size, train_files=train_files, dev_fname=args.eval)
    else:
        # tokenizer = transformers.BigBirdTokenizerFast.from_pretrained(args.tokenizer)
        tokenizer = transformers.BertTokenizerFast.from_pretrained(args.tokenizer)
        input_size = 4096
        block_size = 64 # TODO: Get from configuration or model
        batch_size = 2
        accumulate_grad_batches = 16
        data = EccoBigBirdDataModule(batch_size=batch_size, tokenizer=tokenizer, input_size=input_size, train_files=train_files, dev_fname=args.eval, block_size=block_size)

    print(f"Number of nodes {num_nodes}, GPUs per node {gpus}, batch size {batch_size}, accumulate_grad_batches {accumulate_grad_batches}, learning rate {lr}")
    print(f"Model {args.model}, tokenizer {args.tokenizer} ({'fast' if tokenizer.is_fast else 'slow'})")

    data.setup()

    if args.only_loop_data:
        for e in range(200):
            data.current_epoch=e
            for batch in tqdm.tqdm(data.train_dataloader()):
                pass
                #print(list(batch.keys()))
        sys.exit(0)

    if args.load_checkpoint:
        model = EccoBERT.load_from_checkpoint(checkpoint_path=args.load_checkpoint)
        print("Model loaded from checkpoint.")
    else:
        steps_train = 1e6
        model = EccoBERT(model_type=args.model, vocab_size=len(tokenizer), lr=lr, steps_train=steps_train)

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')

    # train BERT and evaluate
    trainer = pl.Trainer(
        num_nodes=num_nodes,
        gpus=gpus,
        auto_select_gpus=True,
        accelerator='ddp',
        precision=16,
        val_check_interval=5000,
        #limit_val_batches=300,
        # val_check_interval=1.0,
        num_sanity_val_steps=5,
        max_steps=max_steps,
        # max_epochs=2,
        accumulate_grad_batches=accumulate_grad_batches,
        progress_bar_refresh_rate=50, # Large value prevents crashing in colab
        callbacks=[CheckpointSteps(out_dir=args.out_dir, every_n_steps=1000), lr_monitor],
        checkpoint_callback=False,
        # reload_dataloaders_every_epoch=True, # TODO: Will be removed in Pytorch Lightning v1.6. Replace with the line below.
        ## reload_dataloaders_every_n_epochs=1,
        resume_from_checkpoint=args.load_checkpoint
    )

    trainer.fit(model, datamodule=data)

    model.eval()
    model.cuda()
    print("Evaluating.")
    with torch.no_grad():
         preds = []
         for batch in data.val_dataloader():
             output = model({k: v.cuda() for k, v in batch.items()})
             preds.append(output.logits.argmax(-1))

    print([tokenizer.convert_ids_to_tokens(t) for t in preds[2]])
