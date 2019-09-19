set -x
set -f
python src/run_experiment.py -k StimulusResponse_RandSubjUnif data/processed/**/EN_*.csv.zip data/experiments_rsubjun &&
python src/run_experiment.py -e TransferEntropy_StimulusResponse data/experiments_rsubjun/stimres/* data/experiments_rsubjun &&
python src/run_experiment.py -e BlockEntropy_StimulusResponse data/experiments_rsubjun/stimres/* data/experiments_rsubjun