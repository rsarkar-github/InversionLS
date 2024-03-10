# Run from within this directory

cd /home/rsarkar/
source .bashrc
conda activate py39
cd /home/rsarkar/Research/Thesis

#python -m InversionLS.IntegralEquation.Scripts-Horizontal-Well.p07_perform_update 0 20
python -m InversionLS.IntegralEquation.Scripts-Horizontal-Well.p07_perform_update 1 20
