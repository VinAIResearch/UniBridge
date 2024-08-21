export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

for lang in am ang cdo crh eml frr km kn lij ps sa sd si so olo ta tl tt aym bzd cni gn hch nah oto quy shp tar;
do
    python build.py --lang $lang
done