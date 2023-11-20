#!/bin/bash

for f in xyzs/*.xyz
do

    if [ ! -f $(basename $f .xyz)/output.txt ]
    then
    
    mkdir -p $(basename $f .xyz)
    mkdir -p $(basename $f .xyz)/relax
    cat <<EOF > $(basename $f .xyz)/relax/orca.inp
!B3LYP LANL2DZ OPT
%PAL NPROCS 24 END
%geom
MaxIter 200
TolE 4e-5 # Energy change (a.u.) (about 1e-3 eV)
TolRMSG 2e-4 # RMS gradient (a.u.)
TolMaxG 4e-4 # Max. element of gradient (a.u.) (about 0.02 ev/A)
TolRMSD 4e-2 # RMS displacement (a.u.)
TolMaxD 8e-2 # Max. displacement (a.u.)
END
* xyzfile 0 3 ../../$f

EOF

    mkdir -p $(basename $f .xyz)/triplet
    cat <<EOF > $(basename $f .xyz)/triplet/orca.inp
!B3LYP LANL2DZ
%PAL NPROCS 24 END
* xyzfile 0 3 ../relax/orca.xyz

EOF
    
    
# %BASIS  NEWGTO  IR "def2-TZVP" END
# END
    
    cat <<EOF > $(basename $f .xyz)/orca.inp
!B3LYP LANL2DZ
%PAL NPROCS 24 END
* xyzfile 0 1 relax/orca.xyz

EOF

    cat<<EOF > $(basename $f .xyz)/run.sh
#!/bin/bash -x
#SBATCH --account=rrg-ovoznyy
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=24
#SBATCH --exclusive
#SBATCH --export=ALL
#SBATCH --time=06:00:00
#SBATCH -o $(basename $f .xyz)/output.txt
#SBATCH -e $(basename $f .xyz)/error.txt
#SBATCH --job-name=$(basename $f .xyz)

cd $(basename $f .xyz)/relax
EOF

    cat<<'EOF' >> $(basename $f .xyz)/run.sh

${EBROOTORCA}/orca orca.inp > output.txt
cd ../triplet
${EBROOTORCA}/orca orca.inp > output.txt
cd ..
${EBROOTORCA}/orca orca.inp > output.txt

EOF

    sh $(basename $f .xyz)/run.sh

    fi
    
done
