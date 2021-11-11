import transformers
import random
import sys
import argparse
import gzip

from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers import normalizers
from tokenizers.normalizers import Lowercase, NFD, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordPieceTrainer

def gzip_iterator(files):
    for fn in files:
        with gzip.open(fn, 'rt') as f:
            yield from f

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--filelist', help='List of files to train from')
    parser.add_argument('--N', type=int, default=10000, help="How many files?")
    parser.add_argument('--out', help='Where to save?')

    args = parser.parse_args()

    bert_tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    bert_tokenizer.normalizer = normalizers.Sequence([NFD()])
    bert_tokenizer.pre_tokenizer = Whitespace()
    bert_tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", 1),
            ("[SEP]", 2),
        ],
    )

    files=[]
    with open(args.filelist) as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            files.append(line)

    random.shuffle(files)
    print(f"Got {len(files)} files",file=sys.stderr,flush=True)

    trainer = WordPieceTrainer(vocab_size=500, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    bert_tokenizer.train_from_iterator(gzip_iterator(files[:args.N]), trainer)
    bert_tokenizer.save(args.out)
