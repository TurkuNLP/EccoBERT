import argparse
# import torch
# import transformers
# import pytorch_lightning as pl
import ecco_bert as eb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help="Path to the pretrained model.")
    parser.add_argument('out_path', help="A directory to which the BERT model is saved.")

    args = parser.parse_args()

    # tokenizer = transformers.BertTokenizer.from_pretrained('TurkuNLP/bert-base-finnish-cased-v1')
    model = eb.EccoBERT.load_from_checkpoint(checkpoint_path=args.model) # TODO: REMOVE token_vocabulary_size

    model.model.save_pretrained(args.out_path)
