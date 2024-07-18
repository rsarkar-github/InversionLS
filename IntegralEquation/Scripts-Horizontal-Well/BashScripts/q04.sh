# Run from within this directory

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd ~
source .bashrc
conda activate py39

cd $SCRIPT_DIR
cd ../../../..

python -m InversionLS.IntegralEquation.Scripts-Horizontal-Well.q04_display_iter_wavefield 5 50 1
python -m InversionLS.IntegralEquation.Scripts-Horizontal-Well.q04_display_iter_wavefield 12 50 1
python -m InversionLS.IntegralEquation.Scripts-Horizontal-Well.q04_display_iter_wavefield 19 50 1
python -m InversionLS.IntegralEquation.Scripts-Horizontal-Well.q04_display_iter_wavefield 26 50 1


python -m InversionLS.IntegralEquation.Scripts-Horizontal-Well.q04_display_iter_wavefield 5 50 6
python -m InversionLS.IntegralEquation.Scripts-Horizontal-Well.q04_display_iter_wavefield 12 50 6
python -m InversionLS.IntegralEquation.Scripts-Horizontal-Well.q04_display_iter_wavefield 19 50 6
python -m InversionLS.IntegralEquation.Scripts-Horizontal-Well.q04_display_iter_wavefield 26 50 6


python -m InversionLS.IntegralEquation.Scripts-Horizontal-Well.q04_display_iter_wavefield 5 50 9
python -m InversionLS.IntegralEquation.Scripts-Horizontal-Well.q04_display_iter_wavefield 12 50 9
python -m InversionLS.IntegralEquation.Scripts-Horizontal-Well.q04_display_iter_wavefield 19 50 9
python -m InversionLS.IntegralEquation.Scripts-Horizontal-Well.q04_display_iter_wavefield 26 50 9
