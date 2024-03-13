# Run from within this directory


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd ~
source .bashrc
conda activate py39

cd $SCRIPT_DIR
cd ../../../..

python -m InversionLS.IntegralEquation.Scripts-Horizontal-Well.p07_perform_update 2 20 0.0001
python -m InversionLS.IntegralEquation.Scripts-Horizontal-Well.p07_perform_update 3 20 0.00001
python -m InversionLS.IntegralEquation.Scripts-Horizontal-Well.p07_perform_update 4 20 0.00001
python -m InversionLS.IntegralEquation.Scripts-Horizontal-Well.p07_perform_update 5 20 0.000001
python -m InversionLS.IntegralEquation.Scripts-Horizontal-Well.p07_perform_update 6 40 0.000001