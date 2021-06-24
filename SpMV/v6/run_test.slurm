#! /bin/bash
#SBATCH -J baseline
#SBATCH -p PilotCup
#SBATCH -N 1
#SBATCH --gres=dcu:4
module switch compiler/rocm/2.9 compiler/rocm/3.9.1
hipcc -Dgpu -I/public/software/compiler/rocm/rocm-3.9.1/rocsparse/include -I ./  -L/public/software/compiler/rocm/rocm-3.9.1/rocsparse/lib/ -lrocsparse -O3 main.cpp -o Csrsparse
echo "----------------af23560----------------"
echo "---------------------------------------"
./Csrsparse /public/home/liutengyu/csr_data/boneS10.csr
echo "---------------------------------------"

echo "----------------bayer10----------------"
echo "---------------------------------------"
./Csrsparse /public/home/liutengyu/csr_data/Bump_2911.csr
echo "---------------------------------------"

echo "----------------bcsstk18----------------"
echo "---------------------------------------"
./Csrsparse /public/home/liutengyu/csr_data/Cube_Coup_dt6.csr
echo "---------------------------------------"

echo "----------------coater2----------------"
echo "---------------------------------------"
./Csrsparse /public/home/liutengyu/csr_data/dielFilterV3real.csr
echo "---------------------------------------"

echo "----------------dw4096----------------"
echo "---------------------------------------"
./Csrsparse /public/home/liutengyu/csr_data/Ga41As41H72.csr
echo "---------------------------------------"

echo "----------------epb1----------------"
echo "---------------------------------------"
./Csrsparse /public/home/liutengyu/csr_data/Hardesty3.csr
echo "---------------------------------------"

echo "----------------exdata_1----------------"
echo "---------------------------------------"
./Csrsparse /public/home/liutengyu/csr_data/largebasis.csr
echo "---------------------------------------"

echo "----------------nemeth03----------------"
echo "---------------------------------------"
./Csrsparse /public/home/liutengyu/csr_data/RM07R.csr
echo "---------------------------------------"

echo "----------------poli_large----------------"
echo "---------------------------------------"
./Csrsparse /public/home/liutengyu/csr_data/TSOPF_RS_b2383.csr
echo "---------------------------------------"

echo "----------------rajat03----------------"
echo "---------------------------------------"
./Csrsparse /public/home/liutengyu/csr_data/vas_stokes_2M.csr
echo "---------------------------------------"
