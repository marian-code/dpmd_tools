paths=(
    #md_bc8_600K
    #md_bct5_600K_0GPa
    #md_btin_600K
    #md_cd_600K
    #md_hex_dia_600K
    #md_Imma_600K
    #md_R8_600K
    md_sc_300K_h_64
    #md_sh_600K
    #md_ST12_600K
    #md_Ge136_10kbar
    #md_Ge136_11kbar
    #md_Ge136_12.5kbar
    #md_Ge136_15kbar
    #md_Ge136_20kbar
    #md_Ge136_30kbar
    #md_Ge136_50kbar
    #md_Ge136_267
    #md_Ge136_269
    #md_Ge136_274
    #mtd_Ge136
    #ea_Ge8_0GPa
)

for index in ${!paths[*]}; do 

    md="${paths[$index]}"

    # move to exported data dir
    cd $md/deepmd_data
    echo
    echo +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    echo dir: `pwd`
    # read and select data
    dpmd-tools to-deepmd \
        --parser dpmd_raw \
        --press -5 100 \
        --energy -5 -2 \
        --per-atom \
        --every 10 \
        --mode append \
        --force-iteration 0 \
        --get-paths 'list((Path.cwd() / "all").glob("Ge*"))' \
        --save yes\
        #--max-select 5% \
        #--cache-predictions \
        #--block-pbs \
        #--dev-force 0.1 1 \
        #--std-method \
        #--fingerprint-use \
        #--graphs ../../../selective_train1/gen5/train*/ge_all*.pb \
        #--volume 10 31 \
        #--wait_for ../../../selective_train1/gen5/train5_4/ge_all_s5_4.pb
    cd ../..
done
