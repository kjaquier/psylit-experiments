set -f
set -x
for f in data/interim/$1 ; do python src/data_process.py -x $f data/processed/$1 ; done
