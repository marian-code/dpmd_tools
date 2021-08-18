for d in */ 
do
    cd $d
    echo $d
    dpmd-tools dev2ase -t trajectory.lammps -df 0.1 1 -p 0.005 -li in.lammps -pi plumed.dat
    cd ..
done
