set -x
set -f
python src/run_experiment.py -ek TransferEntropy_StimulusResponse data/analytics/stimres/* data/analytics
