import pandas as pd
import LMI_main as lmi
# run_max.py
# import subprocess
# #
# # # Define command and arguments
# # command = 'Rscript'
# # path2script = 'processed_data/R_output/Rscript.R'
# #
# # # Variable number of args in a list
# # args = ['11', '3', '9', '42']
# #
# # # Build subprocess command
# # cmd = [command, path2script] + args
# #
# # # check_output will run the command and store to result
# # x = subprocess.check_output(cmd, universal_newlines=True)
# #
# # print('The maximum of the numbers is:', x)

def agg_ethnic(grp):
    grp  = grp.apply(pd.Series.value_counts).fillna(0).loc[1.0] / grp.shape[0] * 100
    return grp

df = lmi.LCA_in()
df2 = df[['amer_indian_alskn_native', 'blck_afr_amer', 'east_asian', 'latino', 'middl_eastern', 'hawii_pac_islnd', 'south_asian', 'white', 'nan_ethnic', 'other_ethnic','class']]
df2 = df2.fillna(0)
df_out = df2.groupby('class').apply(agg_ethnic).transpose()
df_out['All'] = df2.apply(pd.Series.value_counts).fillna(0).loc[1.0] / df2.shape[0] * 100
df_out = df_out.drop('class')
