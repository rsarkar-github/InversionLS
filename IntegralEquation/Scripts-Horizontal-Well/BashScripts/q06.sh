# Run from within this directory

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd ~
source .bashrc
conda activate py39

cd $SCRIPT_DIR
cd ../../../..

python -m InversionLS.IntegralEquation.Scripts-Horizontal-Well.q06_display_obj -1
python -m InversionLS.IntegralEquation.Scripts-Horizontal-Well.q06_display_obj 0
python -m InversionLS.IntegralEquation.Scripts-Horizontal-Well.q06_display_obj 1
python -m InversionLS.IntegralEquation.Scripts-Horizontal-Well.q06_display_obj 6