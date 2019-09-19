set -x
set -f
python src/run_experiment.py -k StimulusResponse_RandSubjProb data/processed/**/EN_*.csv.zip data/experiments_rsubjpr &&
python src/run_experiment.py -e TransferEntropy_StimulusResponse data/experiments_rsubjpr/stimres/* data/experiments_rsubjpr &&
python src/run_experiment.py -e BlockEntropy_StimulusResponse data/experiments_rsubjpr/stimres/* data/experiments_rsubjpr