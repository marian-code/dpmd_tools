paths=(
    md_bc8_600K
    md_bct5_600K
    md_btin_600K
    md_cd_600K
    md_hex_dia_600K
    md_Imma_600K
    md_R8_600K
    md_sh_600K
    md_ST12_600K
    md_Ge136_10kbar
    md_Ge136_11kbar
    md_Ge136_12.5kbar
    md_Ge136_15kbar
    md_Ge136_20kbar
    #md_Ge136_30kbar
    #md_Ge136_50kbar
    md_Ge136_267
    md_Ge136_269
    md_Ge136_274
    mtd_Ge136
    ea_Ge8_0GPa
)

for index in ${!paths[*]}; do 

    md="${paths[$index]}"

    # move to exported data dir
    cd $md/deepmd_data
    echo
    echo +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    echo dir: `pwd`
    # read and select data
    to_deepmd \
        --parser dpmd_raw \
        --graphs ../../../selective_train1/gen3/train3_*/ge_all*.pb \
        --volume 10 31 \
        --energy -5 -2 \
        --per-atom \
        --dev-force 0.1 1 \
        --std-method \
        --fingerprint-use \
        --mode append \
        --max-select 5% \
        --cache-predictions \
        --block-pbs \
        --auto-save \
        --wait_for ../../../selective_train1/gen3/train3_4/ge_all_s3_4.pb
    cd ../..
done