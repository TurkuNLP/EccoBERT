import pytorch_lightning as pl
import transformers
import torch
import argparse
import bisect
import random
import collections
import itertools
import gzip
import tqdm 
import sys

def pad(l, size):
    return l + [0]*(size - len(l))

def transpose(l):
  return [list(t) for t in zip(*l)]

def load_fields(fn):
    with gzip.open(fn, 'rt') as f:
        return transpose([l.rstrip('\n').split('\t') for l in f])

class EccoBERT(pl.LightningModule):

    def __init__(self, bert_model, steps_train=None):
        super().__init__()
        self.save_hyperparameters()
        configuration = transformers.BertConfig.from_pretrained(bert_model)
        self.steps_train = steps_train
        self.bert=transformers.BertForPreTraining(configuration)
        # self.accuracy = pl.metrics.Accuracy()
        # self.val_accuracy = pl.metrics.Accuracy()

    def forward(self,batch):
        return self.bert(input_ids=batch['input_ids'],
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
        return outputs.loss

    def validation_step(self,batch,batch_idx):
        outputs = self(batch)
        self.log("val_loss", outputs.loss, prog_bar=True)
        # self.log("val_acc", self.val_accuracy, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = transformers.optimization.AdamW(self.parameters(), lr=8e-5, weight_decay=0.01)
        scheduler = transformers.optimization.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10000, num_training_steps=self.steps_train)
        scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
        return [optimizer], [scheduler]

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

# Segment a list of strings into tokenized chunks, with a maximum length of 256 tokens.
# Segments shorter than 150 tokens are generated only at end-of-document positions.
def segment(tokenizer, group_texts):
    min_len = 150
    max_len = 255
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

def to_example(window, cls, sep):
    w1 = window[0]
    w2, next_sentence_label = (window[1], 0) if random.random() > 0.5 else (random.choice(window[2:]), 1)
    return {'input_ids': [cls] + w1 + [sep] + w2 + [sep], 'token_type_ids': [0]*(len(w1)+2) + [1]*(len(w2)+1), 'attention_mask': [1]*(len(w1)+len(w2)+3), 'next_sentence_label': next_sentence_label}

class EccoDataset(torch.utils.data.IterableDataset):
    def __init__(self, tokenizer, input_size, group_texts):
        super().__init__()
        self.tokenizer = tokenizer
        self.input_size = input_size
        self.group_texts = group_texts

    @classmethod
    def from_newline_delimited(cls, fname, tokenizer, input_size):
        group_texts_raw = load_fields(fname)[0]
        group_texts = []
        group_text = []
        for text in group_texts_raw:
            if not text:
                group_texts.append(' '.join(group_text))
                group_text = []
            else:
                group_text.append(text)

        return cls(tokenizer=tokenizer, input_size=input_size, group_texts=group_texts)

    def __iter__(self):
        return (to_example(window, cls=self.tokenizer.cls_token_id, sep=self.tokenizer.sep_token_id) for window in sliding_window(segment(self.tokenizer, self.group_texts), 50))

def pad_with_value(vals, padding_value):
    vals=[torch.LongTensor(v) for v in vals]
    return torch.nn.utils.rnn.pad_sequence(vals, batch_first=True, padding_value=padding_value)

def collate(itemlist, data_collator, pad_token_id):
    batch={}
    masked = data_collator(itemlist)

    batch['attention_mask'] = pad_with_value([item['attention_mask'] for item in itemlist], pad_token_id) # Here pad_token_id=3
    batch['token_type_ids'] = pad_with_value([item['token_type_ids'] for item in itemlist], 0) # Can't use pad_token_id since index out of bounds
    batch['input_ids'] = masked['input_ids']
    batch['label'] = masked['labels']
    batch['next_sentence_label'] = torch.tensor([item['next_sentence_label'] for item in itemlist], dtype=torch.long)
    
    return batch

class MiniEpochDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, tokenizer, input_size, train_files, dev_fname):
        super().__init__()
        self.batch_size = batch_size
        self.train_files = train_files
        self.dev_fname = dev_fname
        self.tokenizer = tokenizer
        self.input_size = input_size
        self.data_collator = transformers.DataCollatorForWholeWordMask(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
        self.collate_fn = lambda x: collate(x, self.data_collator, self.tokenizer.pad_token_id)

    #def setup(self, stage=None):
    #    self.dev_data = EccoDataset.from_newline_delimited(self.dev_fname, input_size=self.input_size, tokenizer=self.tokenizer)

    def train_dataloader(self):
        if not self.trainer:
            epoch_num=self.current_epoch #this is only set when --only-loop-data is used
        else:
            epoch_num=self.trainer.current_epoch
        fn = self.train_files[epoch_num % len(self.train_files)]
        print(f"Current epoch: {epoch_num}, loading file: {fn}", flush=True)
        dataset = EccoDataset.from_newline_delimited(fn, input_size=self.input_size, tokenizer=self.tokenizer)
        return torch.utils.data.DataLoader(dataset, collate_fn=self.collate_fn, batch_size=self.batch_size, num_workers=1, pin_memory=True)

    def val_dataloader(self): 
        dataset=EccoDataset.from_newline_delimited(self.dev_fname, input_size=self.input_size, tokenizer=self.tokenizer)
        return torch.utils.data.DataLoader(dataset, collate_fn=self.collate_fn, batch_size=self.batch_size, num_workers=1, pin_memory=True)

class CheckpointEpoch(pl.callbacks.Callback):
    def __init__(self, out_dir, every_n_epochs):
        self.out_dir = out_dir
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, _):
        if (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            trainer.save_checkpoint(self.out_dir + f'/epoch-{trainer.current_epoch}.ckpt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tokenizer', help="Path to the tokenizer vocabulary file.")
    parser.add_argument('train', help="A text file containing paths to the training files.")
    parser.add_argument('eval', help="A TSV file containing the full unannotated evaluation texts.")
    parser.add_argument('out_dir', help="A directory to which the model is saved.")
    parser.add_argument('--load_checkpoint', help="A path to a checkpoint file to load.")
    parser.add_argument('--only-loop-data', action="store_true", default=False, help="Just loop over the data do nothing else")
    parser.add_argument('--gpus', type=int, default=1, help="Number of GPUs per node")
    parser.add_argument('--nodes', type=int, default=1, help="Number of nodes")

    args = parser.parse_args()

    input_size = 512
    batch_size = 8
    max_steps = 9e5
    # max_steps = 1500
    model_name = "TurkuNLP/bert-base-finnish-cased-v1"
    gpus = args.gpus
    num_nodes = args.nodes
    # TODO: THIS MUST BE SHUFFLED BECAUSE WE CANNOT KNOW WHETHER THE FILES DONT COME IN SOME DESTRUCTIVE ORDER -Filip
    train_files = [l.rstrip('\n') for l in open(args.train).readlines()]

    # TODO: Get the special tokens from the vocabulary file
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_file=args.tokenizer, unk_token='[UNK]', cls_token='[CLS]', sep_token='[SEP]', pad_token='[PAD]', mask_token='[MASK]')
    data = MiniEpochDataModule(batch_size=batch_size, tokenizer=tokenizer, input_size=input_size, train_files=train_files, dev_fname=args.eval)
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
        model = EccoBERT(bert_model=model_name, steps_train=steps_train)

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
        accumulate_grad_batches=11,
        progress_bar_refresh_rate=5, # Large value prevents crashing in colab
        callbacks=[CheckpointEpoch(out_dir=args.out_dir, every_n_epochs=100), lr_monitor],
        # reload_dataloaders_every_epoch=True, # TODO: Will be removed in Pytorch Lightning v1.6. Replace with the line below.
        # reload_dataloaders_every_n_epochs=1,
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
