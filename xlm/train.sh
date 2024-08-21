export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

for lang in am ang cdo crh eml frr km kn lij ps sa sd si so olo ta tl tt;
do
    python train_tgt_mlm.py --lang $lang --seed 42 --lr 5e-4 --mlm_epoch 50
done
for lang in bzd hch shp tar;
do
    python train_tgt_mlm.py --lang $lang --seed 42 --lr 5e-4 --mlm_epoch 50
done
for lang in aym cni gn nah oto quy;
do
    python train_tgt_mlm.py --lang $lang --seed 42 --lr 7e-4 --mlm_epoch 50
done

for lang in ar ja en zh;
do
    python train_src_mlm.py --lang $lang --seed 42 --lr 1e-3 --epoch 50
done
python train_src_mlm.py --lang ru --seed 42 --lr 2e-4 --epoch 50

for lang in en ru zh;
do
    python train_src_ner.py --lang $lang --lr 1e-3 --seed 42 --epoch 10
done
for lang in ar ja;
do
    python train_src_ner.py --lang $lang --lr 5e-4 --seed 42 --epoch 10
done
python train_src_pos.py --subset en_ewt+en_lines+en_partut+en_gum --lr 1e-3 --epoch 10
python train_src_pos.py --subset ar_padt --lr 1e-3 --epoch 10
python train_src_pos.py --subset ja_gsd --lr 1e-3 --epoch 10
python train_src_pos.py --subset ru_gsd+ru_syntagrus+ru_taiga --lr 1e-3 --epoch 10
python train_src_pos.py --subset zh_gsd --lr 1e-3 --epoch 10

for lang in ar ru;
do
    python train_src_nli.py --lang $lang --lr 5e-4 --epoch 10
done
for lang in en zh;
do
    python train_src_nli.py --lang $lang --lr 1e-4 --epoch 10
done