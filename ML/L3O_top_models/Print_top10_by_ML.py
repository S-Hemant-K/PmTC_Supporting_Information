#L3O Models
from joblib import load
import pandas as pd

ENERGY_COLS = [
    'ener', 'ey', 'sf', 'vw', 'el', 'deltaSurf', 'EnerSc',
    'OMM_iPE', 'OMM_fPE',
    'Sand_Elec', 'Sand_VDW', 'Sand_Elec14', 'Sand_VDW14', 'Sand_TOTAL',
    'PMD_HBF', 'PMD_PTF'
]

VHL_CONTACTS = [
    'VAL_62','LEU_63','ARG_64','SER_65','VAL_66','ASN_67','SER_68','ARG_69',
    'GLU_70','PRO_71','SER_72','GLN_73','VAL_74','ILE_75','PHE_76','CAS_77',
    'ASN_78','ARG_79','SER_80','PRO_81','ARG_82','VAL_83','LEU_85','PRO_86',
    'VAL_87','TRP_88','LEU_89','ASN_90','PHE_91','ASP_92','GLY_93','GLU_94',
    'PRO_95','GLN_96','PRO_97','TYR_98','PRO_99','THR_100','LEU_101','PRO_102',
    'PRO_103','GLY_104','THR_105','GLY_106','ARG_107','ARG_108','ILE_109',
    'HIE_110','SER_111','TYR_112','ARG_113','TRP_117','ARG_120','ASP_121',
    'ALA_122','GLY_123','THR_124','HIE_125','ASP_126','SER_139','LEU_140',
    'ASN_141','VAL_142','ASP_143','GLY_144','GLN_145','PRO_146','ILE_147',
    'PHE_148','ARG_167','LYS_171','PRO_172','GLU_173','TYR_175','LEU_188',
    'GLU_189','ASP_190','HIE_191','PRO_192','ASN_193','VAL_194','GLN_195',
    'LYS_196','GLU_199','THR_202'
]


full_cols = ENERGY_COLS + VHL_CONTACTS

top_models = ["6hay_8bdx_9rkj_LightGBM_Full", "6hay_7znt_8qw6_LightGBM_Full"]

ML_mdl_prfmnc_dict = {x:[] for x in top_models}

usr_inp = input("Enter the path to the input CSV/XLSX file: ")
dfinp = pd.read_csv(usr_inp) if usr_inp.endswith('.csv') else pd.read_excel(usr_inp)
    
for mdl_ in top_models:
    
    pipeline = load(f"Models/{mdl_}_pipeline.joblib")
    scaler   = load(f"Models/{mdl_}_scaler.joblib")

    if "Contacts" not in mdl_:
        X_new = dfinp[full_cols].copy().fillna(0)
        energy_present = [c for c in ENERGY_COLS if c in X_new.columns]
        X_new[energy_present] = scaler.transform(X_new[energy_present])

    else:
        X_new = dfinp[VHL_CONTACTS].copy().fillna(0)

    y_prob = pipeline.predict_proba(X_new)[:, 1]

    dfinp[f'{mdl_}_prob'] = y_prob
    df_sorted = dfinp.sort_values(f'{mdl_}_prob', ascending=False)
    ranked_list = df_sorted.i.to_list()

    dfinp[f'{mdl_}_prob'] = y_prob
    df_sorted = dfinp.sort_values(f'{mdl_}_prob', ascending=False)

    mdls_top10 = df_sorted.head(10).i.to_list()

    print(f"\nTop 10 predictions for model {mdl_}:")
    print(df_sorted.head(10)[['i', f'{mdl_}_prob']].to_string(index=False))
