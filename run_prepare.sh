set -f
set -x
for f in data/raw/$1$(echo "/*.txt") ; do python src/data_prepare.py -x $f data/interim/$1 ; done
