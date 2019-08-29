set -f
set -x
for f in data/processed/$1/$(echo '*.*') ; do python src/run_experiment.py $2 $f data/analytics ; done

