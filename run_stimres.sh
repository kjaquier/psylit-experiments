set -x
set -f
python src/run_experiment.py -k StimulusResponse data/processed/$1/EN_* data/analytics
