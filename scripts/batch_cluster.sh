paths=(
    md_Ge136_10kbar
    md_Ge136_11kbar
    md_Ge136_12.5kbar
    md_Ge136_15kbar
    md_Ge136_20kbar
    md_Ge136_30kbar
    md_Ge136_50kbar
    md_Ge136_267
    md_Ge136_269
    md_Ge136_274
)
# approx 1000 structures per cluster
clusters=(
    70
    90
    30
    30
    20
    10
    20
    10
    10
    10
)

for index in ${!paths[*]}; do 

    md="${paths[$index]}"
    clusters="${clusters[$index]}"

    # move to system dir
    cd $md
    echo dir: `pwd`
    echo number of clusters: $clusters
    # read and partition data
    # to_deepmd -p vasp_files -e 10 -v 10 31 -n -5 -2 -a -gp '[Path.cwd() / "OUTCAR"]'
    # move to exported data dir
    cd deepmd_data/all/*
    # fingerprint dataset
    cluster_deepmd take-prints -bs 100000 -p
    # assign clusters
    cluster_deepmd select -p 5000 -bs 100000 -nc $clusters
    cd ../../../..
done	
