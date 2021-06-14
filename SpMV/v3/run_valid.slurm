#! /bin/bash
#SBATCH -J baseline
#SBATCH -p PilotCup
#SBATCH -N 1
#SBATCH --gres=dcu:4
module switch compiler/rocm/2.9 compiler/rocm/3.9.1
hipcc -Dgpu -I/public/software/compiler/rocm/rocm-3.9.1/rocsparse/include -I ./  -L/public/software/compiler/rocm/rocm-3.9.1/rocsparse/lib/ -lrocsparse -O3 main.cpp -o Csrsparse
echo "----------------af23560----------------"
echo "---------------------------------------"
./Csrsparse ../data/af23560.csr
echo "---------------------------------------"

echo "----------------bayer10----------------"
echo "---------------------------------------"
./Csrsparse ../data/bayer10.csr
echo "---------------------------------------"

echo "----------------bcsstk18----------------"
echo "---------------------------------------"
./Csrsparse ../data/bcsstk18.csr
echo "---------------------------------------"

echo "----------------coater2----------------"
echo "---------------------------------------"
./Csrsparse ../data/coater2.csr
echo "---------------------------------------"

echo "----------------dw4096----------------"
echo "---------------------------------------"
./Csrsparse ../data/dw4096.csr
echo "---------------------------------------"

echo "----------------epb1----------------"
echo "---------------------------------------"
./Csrsparse ../data/epb1.csr
echo "---------------------------------------"

echo "----------------exdata_1----------------"
echo "---------------------------------------"
./Csrsparse ../data/exdata_1.csr
echo "---------------------------------------"

echo "----------------nemeth03----------------"
echo "---------------------------------------"
./Csrsparse ../data/nemeth03.csr
echo "---------------------------------------"

echo "----------------poli_large----------------"
echo "---------------------------------------"
./Csrsparse ../data/poli_large.csr
echo "---------------------------------------"

echo "----------------rajat03----------------"
echo "---------------------------------------"
./Csrsparse ../data/rajat03.csr
echo "---------------------------------------"
