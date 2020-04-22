export WORKSPACE="/home/aistudio/work/nlp_2020"
export DATADIR="${WORKSPACE}/data"
python nlp_2020/classification/train.py --data_dir ${DATADIR}/classification \
--model_name_or_path ${DATADIR}/model \
--output_dir ${DATADIR}/output \
--cache_dir ${DATADIR}/cache \
--embed_path ${DATADIR}/sgns.sogounews.bigram-char
