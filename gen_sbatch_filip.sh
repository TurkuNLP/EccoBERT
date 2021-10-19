#echo "sbatch -e eo/ecco_bert.err -o eo/ecco_bert.out -J ecco_bert ecco_bert_sbatch.sh data/EccoBERT_tokenizer.json /projappl/project_2005072/rastasii/EccoBERT/train_list.txt /projappl/project_2005072/rastasii/EccoBERT/ecco_rand_part_000000_dev.gz /scratch/project_2002820/filip/checkpoints/ecco_bert"

echo "sbatch -e eo/ecco_bert.err -o eo/ecco_bert.out -J ecco_bert ecco_bert_sbatch.sh data/EccoBERT_tokenizer.json /projappl/project_2005072/rastasii/EccoBERT/train_list.txt ecco_rand_part_10K_dev.gz /scratch/project_2002820/filip/checkpoints/ecco_bert"
