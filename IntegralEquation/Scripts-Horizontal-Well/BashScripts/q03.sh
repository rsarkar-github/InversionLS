# Run from within this directory

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd ~
source .bashrc
conda activate py39

cd $SCRIPT_DIR
cd ../../../..

python -m InversionLS.IntegralEquation.Scripts-Horizontal-Well.q03_display_true_wavefield 5 50
python -m InversionLS.IntegralEquation.Scripts-Horizontal-Well.q03_display_true_wavefield 12 50
python -m InversionLS.IntegralEquation.Scripts-Horizontal-Well.q03_display_true_wavefield 19 50
python -m InversionLS.IntegralEquation.Scripts-Horizontal-Well.q03_display_true_wavefield 26 50