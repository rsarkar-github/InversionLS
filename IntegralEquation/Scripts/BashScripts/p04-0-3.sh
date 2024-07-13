# Run from within this directory

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd ~
source .bashrc
conda activate py39

cd $SCRIPT_DIR
cd ../../../..

python -m InversionLS.IntegralEquation.Scripts.p04_lse_solves 0 3 1 0
python -m InversionLS.IntegralEquation.Scripts.p04_lse_solves 0 3 2 0
python -m InversionLS.IntegralEquation.Scripts.p04_lse_solves 0 3 1 1
python -m InversionLS.IntegralEquation.Scripts.p04_lse_solves 0 3 2 1
python -m InversionLS.IntegralEquation.Scripts.p04_lse_solves 0 3 1 2
python -m InversionLS.IntegralEquation.Scripts.p04_lse_solves 0 3 2 2
