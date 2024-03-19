# Run from within this directory


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd ~
source .bashrc
conda activate py39

cd $SCRIPT_DIR
cd ../../../..

#python -m InversionLS.IntegralEquation.Scripts-Horizontal-Well.p09_perform_update 12 100 0.0
python -m InversionLS.IntegralEquation.Scripts-Horizontal-Well.p09_perform_update 13 100 0.0