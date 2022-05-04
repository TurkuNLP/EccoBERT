import argparse
# import torch
# import transformers
# import pytorch_lightning as pl
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt
import ecco_bert as eb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help="Path to the pretrained model.")
    parser.add_argument('out_path', help="A directory to which the BERT model is saved.")

    args = parser.parse_args()

    # tokenizer = transformers.BertTokenizer.from_pretrained('TurkuNLP/bert-base-finnish-cased-v1')
    model = eb.EccoBERT.load_from_checkpoint(checkpoint_path=args.model) # TODO: REMOVE token_vocabulary_size

    # print(model.noise_layer.weight)
    # path = pathlib.Path('plots')
    # path.mkdir(parents=True, exist_ok=True)
    # with sns.axes_style('ticks'):
    #     ax = sns.heatmap(model.noise_layer.weight.detach().numpy(), center=0, linewidths=0, linecolor='black', xticklabels=10, yticklabels=10)
    #     plt.show()
    #     plt.savefig(path / 'noise_layer_heatmap.pdf')
    # plt.close()

    model.model.save_pretrained(args.out_path)
