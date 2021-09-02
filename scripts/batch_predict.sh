paths=(
    md_bc8_600K
    #md_bct5_600K_0GPa
    #md_btin_600K
    #md_cd_600K
    #md_hex_dia_600K
    #md_Imma_600K
    #md_R8_600K
    #md_sc_50K
    #md_sc_150K
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
    #mtd_Ge136_50kbar
    #mtd_Ge136_100kbar
    #####!ea_Ge8_0GPa
    #ea_recompute_0GPa
    #ea_recompute_10GPa
    #ea_recompute_20GPa
    #continued_83_100kbar
    #continued_83_100kbar_heat
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
        --mode append \
        --save no\
        --max-select 5% \
        --cache-predictions \
        --block-pbs \
        --dev-force 0.1 1 \
        --std-method \
        --fingerprint-use \
        --graphs ../../../selective_train2/gen5/train5_[5-8]/ge_all*.pb \
        --volume 10 31 \
        --wait-for ../../../selective_train2/gen5/train5_5/ge_all_s5_5.pb ../../../selective_train2/gen5/train5_6/ge_all_s5_6.pb ../../../selective_train2/gen5/train5_7/ge_all_s5_7.pb ../../../selective_train2/gen5/train5_8/ge_all_s5_8.pb
    cd ../..
done


#*1. generacia
# vsetky 5000 krokove md
#*2. generacia
# vsetky 5000 krokove md, ea 0GPa, mtd 100kbar
#*3. generacia
# ea 10 GPa, md 11kbar, md 50kbar, md 267, mtd 50kbar, continued 83 100kbar
#*4. generacia
# ea 20 GPa, md 10kbar, md 274, md 50kbar, Imma, R8, sh, continued 83 100kbar heat
#*5. generacia
# 

'list(Path.cwd().glob("md_*0K/deepmd_data/all/*"))
[Path("ea_recompute_0GPa/deepmd_data/all/Ge8")]
[Path("mtd_Ge136_100kbar/deepmd_data/all/Ge136")]
[Path("ea_recompute_10GPa/deepmd_data/all/Ge8")]
[Path("md_Ge136_11kbar/deepmd_data/all/Ge136")]
[Path("md_Ge136_50kbar/deepmd_data/all/Ge136")]
[Path("md_Ge136_267/deepmd_data/all/Ge136")]
[Path("mtd_Ge136_50kbar/deepmd_data/all/Ge136")]
[Path("continued_83_100kbar/deepmd_data/all/Ge136")]
[Path("ea_recompute_20GPa/deepmd_data/all/Ge8")]
[Path("md_Ge136_10kbar/deepmd_data/all/Ge136")]
[Path("md_Ge136_274/deepmd_data/all/Ge136")]
[Path("continued_83_100kbar_heat/deepmd_data/all/Ge136")]'