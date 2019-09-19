set -x
set -f
python src/run_experiment.py -ek TransferEntropy_StimulusResponse data/analytics/stimres_batch$1/* data/experiments
