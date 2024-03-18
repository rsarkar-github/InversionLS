#!/usr/bin/bash
#SBATCH --job-name=p01_
#SBATCH --output=p01_.%j.out
#SBATCH --error=p01_.%j.err
#SBATCH --time=1:00:00
#SBATCH -p cpu
#SBATCH --ntasks=1
#SBATCH -c 32

echo "hostname: " $HOSTNAME

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo $SCRIPT_DIR

cd ~
source .bashrc
conda activate py39

cd $SCRIPT_DIR
cd ../../../..

python -m InversionLS.IntegralEquation.Scripts-Marmousi.p01_create_params_jsonfile