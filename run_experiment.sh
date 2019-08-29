set -f
set -x
for f in data/processed/$1 ; do python src/run_experiment.py $2 $f data/analytics/blockent_stimres ; done

