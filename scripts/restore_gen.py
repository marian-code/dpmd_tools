import numpy as np
from tqdm import tqdm
from pathlib import Path


dirs = {
    #"md_bc8_600K": -1,
    #"md_bct5_600K_0GPa": -1,
    #"md_btin_600K": -1,
    #"md_cd_600K": -1,
    #"md_hex_dia_600K": -1,
    #"md_Imma_600K": -1,
    #"md_R8_600K": -1,
    #"md_sh_600K": -1,
    #"md_ST12_600K": -1,
    #"md_Ge136_10kbar": -1,
    #"md_Ge136_11kbar": -1,
    "md_Ge136_12.5kbar": -1,
    #"md_Ge136_15kbar": -1,
    #"md_Ge136_20kbar": -1,
    #"md_Ge136_30kbar": -1,
    #"md_Ge136_50kbar": -1,
    #"md_Ge136_267": -1,
    #"md_Ge136_269": -1,
    #"md_Ge136_274": -1,
    "mtd_Ge136_50kbar": -1,
    "mtd_Ge136_100kbar": -1,
    #"ea_0GPa_zero_step": -1,
    #"ea_10GPa_zero_step": -1,
    #"ea_20GPa_zero_step": -1,
    #"continued_83_100kbar": -1,
    #"continued_83_100kbar_heat": -1,
}

for name, it in tqdm(dirs.items()):

    used = list((Path.cwd() / name).glob("deepmd_data/all/*/used.raw"))[0]
    data = np.loadtxt(used)
    #print(name)
    #print(data[:15])
    
    if it == 0 or (len(data.shape) == 1 and it == -1):
        data = np.atleast_2d(np.zeros((data.shape[0]))).T
    else:
        data = data[:, :it]
    np.savetxt(used, data, fmt="%i")
    #print("------------")
    #print(data[:15])
    #print("**************")

