# Run from within this directory


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd ~
source .bashrc
conda activate py39

cd $SCRIPT_DIR
cd ../../../..

python -m InversionLS.IntegralEquation.Scripts-Horizontal-Well.p09_perform_update 7 40 0.000001
python -m InversionLS.IntegralEquation.Scripts-Horizontal-Well.p09_perform_update 8 40 0.000001
python -m InversionLS.IntegralEquation.Scripts-Horizontal-Well.p09_perform_update 9 40 0.0000001