set -x
set -f
# python src/run_experiment.py -k StimulusResponse_NoSemanticRole data/processed/**/EN_*.csv.zip data/experiments_nosemrole &&
python src/run_experiment.py -e BlockEntropy_StimulusResponse data/experiments_nosemrole/stimres_nosemrole/* data/experiments_nosemrole && 
python src/run_experiment.py -e TransferEntropy_StimulusResponse data/experiments_nosemrole/stimres_nosemrole/* data/experiments_nosemrole