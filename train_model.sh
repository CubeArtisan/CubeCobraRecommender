export BUCKET_NAME=cubecobratesting

aws s3 sync s3://$BUCKET_NAME/data/ data/

python -m pip install scikit-build
python setup.py install

python -m src.ml.train_metapath -e 32 --name $(date +"%Y%m%d") --batch-size 512 --embed-dims 64 --metapath-dims 32 --num-heads 8 --dropout 0.0 --noise 0.4 --noise-stddev 0.1 --learning-rate 4e-03 --decks-weight 0.5 --l1-weight 0.1 --l2-weight 0.5 --seed 43 -j 3 --xla --mixed --profile

aws s3 sync ml_files/ s3://$BUCKET_NAME/ml_files/
aws s3 sync logs/ s3://$BUCKET_NAME/logs/
