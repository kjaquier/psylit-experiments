for f in data/raw/$0 ; do python src/data_prepare.py -x $f data/interim/$0 ; done
