import pandas as pd
from string import ascii_lowercase
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

#todo: add emploed/unemployed Q


#this was a preliminary step
# #def pre_consolidate_cv17():
#     df = read_cv17()
#     q_nums = q_num2q_str()
#     q_nums = q_nums[q_nums.isin(list(df))]
#     return df[q_nums]


def q_num2q_cv17_str():
    q_nums = pd.read_csv('C:/Users/james.tamerius/PycharmProjects/CVRP_LMIprofile/cv17_q_nums.csv', header=None)
    tmp_qs = ['cvrpid_cv17']
    for n in q_nums[0].astype('str').tolist():
        tmp_qs.append('q' + n + '_cv17')
        for c in ascii_lowercase:
            tmp_qs.append('q'+ n + '_' + c  + '_cv17')
    return pd.Series(tmp_qs)


def q_num2q_cv16_str():
    q_nums = pd.read_csv('C:/Users/james.tamerius/PycharmProjects/CVRP_LMIprofile/cv16_q_nums.csv', header=None)
    tmp_qs = ['cvrpid_cv16']
    for n in q_nums[0].astype('str').tolist():
        tmp_qs.append('q' + n + '_cv16')
        for c in ascii_lowercase:
            tmp_qs.append('q'+ n + '_' + c  + '_cv16')

    tmp_qs = pd.Series(tmp_qs)
    df = read_cv16()
    df = list(df)
    tmp_qs = tmp_qs[tmp_qs.isin(df)]
    tmp_qs.to_csv('Final_variables_cv2016.csv')
    #this needs to be checked by hand to make sure all variable labels were grabbed


def read_cv17():
    df = pd.read_excel('20181204_6month_2017-18_consumersurvey_PendingQC.xlsx', sheet_name='6-month FINAL')
    return df


def read_cv16():
    df = pd.read_excel('20170927_finalweighted2016-17consumersurvey_DONOTEDIT.xlsx', sheet_name='FINAL')
    return df


def consolidate_cv17():
    df = read_cv17()
    q_nums = pd.read_csv('Final_variables_cv16_cv17_map.csv')
    df = df[q_nums['cvrp_17']]
    df.columns = q_nums['description']
    df['cvrp_17'] = 1
    return df


def consolidate_cv16():
    df = read_cv16()
    q_nums = pd.read_csv('Final_variables_cv16_cv17_map.csv')
    df = df[q_nums['cvrp_16']]
    df.columns = q_nums['description']
    df['cvrp_17'] = 0
    return df


def concat_cv16_cv17(df_cv16, df_cv17):
    df = pd.concat([df_cv17, df_cv16], axis=0)
    return df


def read_cvrp_app_data():
    df_app = pd.read_csv('2019-01-22_lmi_profile_program.csv')
    df_app = df_app[['Application: Application Number', 'Vehicle: Vehicle Category', 'Vehicle Purchase or Lease', 'Vehicle: Vehicle Model', 'Total Amount', 'Purchase Price', 'Age Range', 'Air District', 'County', 'Postal Code', 'Census Tract']]
    df_app.columns=['cvrp_id', 'tech_type', 'purch_lease', 'veh_model', 'tot_rebate', 'purch_price', 'age', 'air_dstrct', 'cnty', 'zipcode', 'census_trct']
    return df_app


def merge_df_and_app(df_app, df_survey):
    df = df_app.merge(df_survey, on='cvrp_id')
    df['tech_type_x'] = df['tech_type_x'].fillna(df['tech_type_y']) # replace missing values in survey data with values from application data
    df['tech_type'] = df['tech_type_x']
    df['age'] = df['age_y']
    df = df.drop(['tech_type_x', 'tech_type_y', 'age_x', 'age_y'], axis=1)

    return df


def process_data():
    df_cv17 = consolidate_cv17()
    df_cv16 = consolidate_cv16()
    df_survey = concat_cv16_cv17(df_cv16, df_cv17)
    df_app = read_cvrp_app_data()
    df = merge_df_and_app(df_app, df_survey)

    output_file = 'processed_data/processed.csv'
    df.to_csv(output_file, index=False)
    return df


def recode_data_binomial(df):
    # many of the categories I combined were due to small numbers, i.e., many cats have small n which is problematic for calssification algorithms

    #In many of the fields below I replace missing values with random values to preserve as many records as possible. The missing values are replaced by a random value that is represntative of the dataset. I.e., if 10% of tech_types were BEV and 90% were PHEV, a missing value has a 10% chance of being replaced by "BEV" and 90% chance of "PHEV".
    def add_missing_vals(ds):
        p = (ds.value_counts() / ds.value_counts().sum()).tolist()
        ins = ds.isna()
        for i in range(sum(ins)):
            rnd_val = np.random.multinomial(1, p, 1)
            r, c = np.where(rnd_val == 1)
            ds = ds.fillna(c[0], limit=1)  # replace missing values with random value (0/1) where porbbaility is determined from mix of gas/alternative from completed date records
        return ds

    #keep track of the number of missing data fields in each record for sensitivty analyses
    df['missing'] = 0
    def add_missing(df, var):
        df.loc[df[var].isna(), 'missing'] = df.loc[df[var].isna(), 'missing'] + 1
        return df


    #Previous vehicle technology type: consolidate and fill in missing values with random values
    df['prev_tech_type'].replace([2, 1], 0, inplace=True) #combine gasoline and diesel
    df['prev_tech_type'].replace([5, 6, 7, 8, 4, -99], 1, inplace=True) #combine alternative fuel (EV, PHEV, HEV, etc)
    df = add_missing(df, 'prev_tech_type')
    df['prev_tech_type'] = add_missing_vals(df['prev_tech_type'])

    # vehicle technology type: code BEV =1 / PHEV = 0
    df = df[df['tech_type'] != 'FCEV'].copy() #drop FCEV
    df.loc[:, 'tech_type'] = (df['tech_type'] == 'BEV') #code BEV =1 / PHEV = 0

    #What was the most important factor for purchasing an EV?
    df = add_missing(df, 'reason')
    df.loc[:, 'reason_saving_fuel'] = df['reason'] == 1  #Saving money on fuel costs
    df.loc[:, 'reason_saving_overall'] = df['reason'] == 2  # Saving money overall
    df.loc[:, 'reason_environ_imp'] = df['reason'] == 3  # Reducing environmental impacts
    df.loc[:, 'reason_other'] = (df['reason'] >= 4)   #Carpool or High Occupancy Vehicle (HOV) lane access, Increased energy independence, Convenience of charging, vehicle performance, style, desire for the newest technology

    # residence type: coded for detached house or other (detached house = 1)
    df['residence_type'].replace([2, 3, -99], 0, inplace=True)
    df['residence_type'].replace([-77], np.nan, inplace=True)
    df = add_missing(df, 'residence_type')
    df.loc[:, 'residence_type'] = add_missing_vals(df['residence_type'])

    #do you have solar panels: coded as boolean (1 = True)
    df['solar_panels'].replace([3, 4, 5, -99], 0, inplace=True)
    df['solar_panels'].replace(2, 1, inplace=True)
    df['solar_panels'].replace([-77], np.nan, inplace=True)
    df = add_missing(df, 'solar_panels')
    df.loc[:, 'solar_panels'] = add_missing_vals(df['solar_panels'])

    # number of vehicles in HH: 0=1-2, 1=>2
    df['num_cars_HH'].replace([1, 2], 0, inplace=True)
    df['num_cars_HH'].replace([3, 4], 1, inplace=True)
    df['num_cars_HH'].replace([-77], np.nan, inplace=True)
    df = add_missing(df, 'num_cars_HH')
    df.loc[:, 'num_cars_HH'] = add_missing_vals(df['num_cars_HH'])

    # age: break age groups into <30, 30-39, 50+
    df['age'].replace([-77], np.nan, inplace=True)
    df = add_missing(df, 'age')
    df.loc[:, 'age'] = add_missing_vals(df['age'])
    df.loc[:, 'age_16_30'] = df['age'] <= 2
    df.loc[:, 'age_30_40'] = df['age'] == 3
    df.loc[:, 'age_40_50'] = df['age'] == 4
    df.loc[:, 'age_50_60'] = df['age'] == 5
    df.loc[:, 'age_60_70'] = df['age'] == 6
    df.loc[:, 'age_70_90'] = df['age'] >= 7

    # gender: coded to boolean; "prefer no to answer", "transgender" treated as missing (randomly assigned values)
    df['gender'].replace([2], 0, inplace=True)
    df['gender'].replace([1], 1, inplace=True)
    df['gender'].replace([3, -99, -77], np.nan, inplace=True)
    df = add_missing(df, 'gender')
    df.loc[:, 'gender'] = add_missing_vals(df['gender'])

    # HH education: "graudate degree"=2, "bachelors degree"=1, "less than bachelor's degree"=0
    df.loc[:, 'HH_educ_graduate_deg'] = df['HH_education'] == 6
    df.loc[:, 'HH_educ_bachelor_deg'] = df['HH_education'] == 5
    df.loc[:, 'HH_educ_less_bachelor_deg'] = df['HH_education'] <=4
    df.loc[:, 'HH_education'].replace([3, -99, -77], np.nan, inplace=True)
    df = add_missing(df, 'HH_education')
    df.loc[:, 'HH_education'] = add_missing_vals(df['HH_education'])

    # income
    df['HH_income'].replace([-77], np.nan, inplace=True)
    df = add_missing(df, 'HH_income')
    df.loc[:, 'HH_income'] = add_missing_vals(df['HH_income'])
    df.loc[:, 'HH_income_less_50k'] = df['HH_income'] <= 2
    df.loc[:, 'HH_income_50_75k'] = df['HH_income'] == 3
    df.loc[:, 'HH_income_gtr_75k'] = df['HH_income'] >= 4

    #HH size
    df['HH_size'].replace([-77], np.nan, inplace=True)
    df = add_missing(df, 'HH_size')
    df.loc[:, 'HH_size'] = add_missing_vals(df['HH_size'])
    df.loc[:, 'HH_size_1'] = df['HH_size'] == 1
    df.loc[:, 'HH_size_2'] = df['HH_size'] == 2
    df.loc[:, 'HH_size_3_or_grtr'] = df['HH_size'] >= 3

    ##Total Rebate >4000 = 1; <=3500 = 0
    df = df[df['tot_rebate'] >= 3000]
    df.loc[:, 'tot_rebate'] = (df['tot_rebate'] > 3500).astype('int')

    ##purchase = 1 / lease = 0
    df.loc[:, 'purch_lease'] = (df['purch_lease'] == 'Purchase').astype('int')

    ##purchase price : exclude values of greater than $50, 000
    df = df[df['purch_price'] < 50000]

    #q78: Ethnicity
    df.loc[:, 's_e_asian'] = (df['east_asian'].fillna(0) + df['south_asian'].fillna(0))>0
    df.loc[:, 'other_ethnic'] = (df['other_ethnic'].fillna(0) + df['amer_indian_alskn_native'].fillna(0) +df['blck_afr_amer'].fillna(0)+df['middl_eastern'].fillna(0)+df['hawii_pac_islnd'].fillna(0)) > 0
    df = df.drop(['east_asian', 'south_asian', 'amer_indian_alskn_native', 'blck_afr_amer', 'middl_eastern', 'hawii_pac_islnd', 'NA'], axis=1)

    mssng = df[['latino', 'white', 'other_ethnic', 's_e_asian']].sum(axis=1) == 0
    for index, row in df.iterrows():
        if row[['latino', 'white', 'other_ethnic', 's_e_asian']].sum() == 0:
            indx = df['cvrp_id'] == row['cvrp_id']
            df.loc[indx, ['latino', 'white', 'other_ethnic', 's_e_asian']] = df.loc[~mssng, ['latino', 'white', 'other_ethnic', 's_e_asian']].sample(n=1, axis=0)

    df = df.drop(['num_lic_drvr_HH', 'HH_education', 'HH_size', 'HH_income', 'age', 'reason', 'tax_filing_status'], axis=1)
    df.loc[:, 'prev_tech_type':] = (df.loc[:, 'prev_tech_type':]).fillna(0).astype(int)

    df = df[df['missing']<3] #only leaving in records that have less than three missing values (this does not include ethnicity)

    today = dt.datetime.today().strftime('%m_%d_%Y')
    output_file = 'processed_data/processed_recode_{}.csv'.format(today)
    df.to_csv(output_file, index=False)

    return df


def recode_data(df):
    # many of the categories I combined were due to small numbers, i.e., many cats have small n which is problematic for calssification algorithms

    #In many of the fields below I replace missing values with random values to preserve as many records as possible. The missing values are replaced by a random value that is represntative of the dataset. I.e., if 10% of tech_types were BEV and 90% were PHEV, a missing value has a 10% chance of being replaced by "BEV" and 90% chance of "PHEV".
    def add_missing_vals(ds):
        p = (ds.value_counts() / ds.value_counts().sum()).tolist()
        ins = ds.isna()
        for i in range(sum(ins)):
            rnd_val = np.random.multinomial(1, p, 1)
            r, c = np.where(rnd_val == 1)
            ds = ds.fillna(c[0]+1, limit=1)  # replace missing values with random value (0/1) where porbbaility is determined from mix of gas/alternative from completed date records
        return ds

    #keep track of the number of missing data fields in each record for sensitivty analyses
    df['missing'] = 0
    def add_missing(df, var):
        df.loc[df[var].isna(), 'missing'] = df.loc[df[var].isna(), 'missing'] + 1
        return df


    #Previous vehicle technology type: consolidate and fill in missing values with random values
    df['prev_tech_type'].replace([2, 1], 1, inplace=True) #combine gasoline and diesel
    df['prev_tech_type'].replace([5, 6, 7, 8, 4, -99], 2, inplace=True) #combine alternative fuel (EV, PHEV, HEV, etc)
    df['prev_tech_type'].replace([-77], np.nan, inplace=True)
    df = add_missing(df, 'prev_tech_type')
    df['prev_tech_type'] = add_missing_vals(df['prev_tech_type'])

    # vehicle technology type: code BEV =1 / PHEV = 0
    df = df[df['tech_type'] != 'FCEV'].copy() #drop FCEV
    df.loc[:, 'tech_type'] = (df['tech_type'] == 'BEV') + 1 #code BEV =1 / PHEV = 0


    #What was the most important factor for purchasing an EV?
    df['reason'].replace([-99], np.nan, inplace=True)
    df['reason'].replace([4,5,6,7,8], 4, inplace=True)
    df = add_missing(df, 'reason')
    df['reason'] = add_missing_vals(df['reason'])
    # 1 Saving money on fuel costs
    # 2 Saving money overall
    # 3 Reducing environmental impacts
    # 4 Carpool or High Occupancy Vehicle (HOV) lane access, Increased energy independence, Convenience of charging, vehicle performance, style, desire for the newest technology

    # residence type: coded for detached house or other (detached house = 1)
    df['residence_type'].replace([2, 3, -99], 2, inplace=True)
    df['residence_type'].replace([-77], np.nan, inplace=True)
    df = add_missing(df, 'residence_type')
    df.loc[:, 'residence_type'] = add_missing_vals(df['residence_type'])

    #do you have solar panels: coded as boolean (1 = True)
    df['solar_panels'].replace([1, 2], 2, inplace=True)
    df['solar_panels'].replace([3, 4, 5, -99], 1, inplace=True)
    df['solar_panels'].replace([-77], np.nan, inplace=True)
    df = add_missing(df, 'solar_panels')
    df.loc[:, 'solar_panels'] = add_missing_vals(df['solar_panels'])

    # number of vehicles in HH: 0=1-2, 1=>2
    df['num_cars_HH'].replace([1, 2], 0, inplace=True)
    df['num_cars_HH'].replace([3, 4], 1, inplace=True)
    df['num_cars_HH'].replace([-77], np.nan, inplace=True)
    df = add_missing(df, 'num_cars_HH')
    df.loc[:, 'num_cars_HH'] = add_missing_vals(df['num_cars_HH'])
    df['num_cars_HH'] = df['num_cars_HH'] + 1

    # age: break age groups
    df['age'].replace([-77], np.nan, inplace=True)
    df = add_missing(df, 'age')
    df.loc[:, 'age'] = add_missing_vals(df['age'])
    df['age'].replace([1, 2, 3], 1, inplace=True)  #age_16_30
    #df['age'].replace([3], 2, inplace=True)  # age_30_40
    df['age'].replace([4], 2, inplace=True)  # age_40_50
    df['age'].replace([5,6,7,8,9], 3, inplace=True)  #  age_50_60
    #df['age'].replace([6], 5, inplace=True)  #  age_60_70
    #df['age'].replace([7, 8, 9], 6, inplace=True)  # age_70+

    # gender: coded to boolean; "prefer no to answer", "transgender" treated as missing (randomly assigned values)
    df['gender'].replace([2], 0, inplace=True)
    df['gender'].replace([1], 1, inplace=True)
    df['gender'].replace([3, -99, -77], np.nan, inplace=True)
    df = add_missing(df, 'gender')
    df.loc[:, 'gender'] = add_missing_vals(df['gender'])
    df['gender'] = df['gender']+1

    # HH education: "graudate degree"=2, "bachelors degree"=1, "less than bachelor's degree"=0
    df['HH_education'].replace([1,2,3,4], 1, inplace=True)  # HH_educ_less_bachelor_deg
    df['HH_education'].replace([5], 2, inplace=True)  # HH_educ_bachelor_deg
    df['HH_education'].replace([6], 3, inplace=True)  # HH_educ_graduate_deg
    df['HH_education'].replace([-99, -77], np.nan, inplace=True)
    df = add_missing(df, 'HH_education')
    df.loc[:, 'HH_education'] = add_missing_vals(df['HH_education'])

    # income
    df['HH_income'].replace([-77], np.nan, inplace=True)
    df = add_missing(df, 'HH_income')
    df.loc[:, 'HH_income'] = add_missing_vals(df['HH_income'])
    df['HH_income'].replace([1, 2], 1, inplace=True)  # HH_income_less_50k
    df['HH_income'].replace([3], 2, inplace=True)  # HH_income_50_75k
    df['HH_income'].replace([4,5,6,7,8,9,10,11,12,13,14,15], 3, inplace=True)  # HH_income_gtr_75k

    #HH size
    df['HH_size'].replace([-77], np.nan, inplace=True)
    df = add_missing(df, 'HH_size')
    df.loc[:, 'HH_size'] = add_missing_vals(df['HH_size'])
    df['HH_size'].replace([1], 1, inplace=True)  # HH_size_1
    df['HH_size'].replace([2], 2, inplace=True)  # HH_size_2
    df['HH_size'].replace([3,4,5,6,7,8,9], 3, inplace=True)  # HH_size_3_or_grtr

    ##Total Rebate >$4000 = 1; $3000-3500 = 0
    df = df[df['tot_rebate'] >= 3000] #remove rebates that are less than $3000
    df.loc[:, 'tot_rebate'] = (df['tot_rebate'] > 3500).astype('int') + 1

    ##purchase = 1 / lease = 0
    df.loc[:, 'purch_lease'] = (df['purch_lease'] == 'Purchase').astype('int')

    ##purchase price : exclude values of greater than $50, 000
    df = df[df['purch_price'] < 50000]

    #q78: Ethnicity
    df.loc[:, 's_e_asian'] = (df['east_asian'].fillna(0) + df['south_asian'].fillna(0))>0
    df.loc[:, 'other_ethnic'] = (df['other_ethnic'].fillna(0) + df['amer_indian_alskn_native'].fillna(0) +df['blck_afr_amer'].fillna(0)+df['middl_eastern'].fillna(0)+df['hawii_pac_islnd'].fillna(0)) > 0
    #df = df.drop(['east_asian', 'south_asian', 'amer_indian_alskn_native', 'blck_afr_amer', 'middl_eastern', 'hawii_pac_islnd', 'NA'], axis=1)

    mssng = df[['latino', 'white', 'other_ethnic', 's_e_asian']].sum(axis=1) == 0

    for index, row in df.iterrows():
        if row[['latino', 'white', 'other_ethnic', 's_e_asian']].sum() == 0:
            indx = df['cvrp_id'] == row['cvrp_id']
            df.loc[indx, ['latino', 'white', 'other_ethnic', 's_e_asian']] = df.loc[~mssng, ['latino', 'white', 'other_ethnic', 's_e_asian']].sample(n=1, axis=0)


    df[['latino', 'white', 'other_ethnic', 's_e_asian']] = df[['latino', 'white', 'other_ethnic', 's_e_asian']].fillna(0).astype('int') + 1


    #df = df.drop(['num_lic_drvr_HH', 'tax_filing_status'], axis=1)


    df = df[df['missing']<3] #only leaving in records that have less than X missing values (this does not include ethnicity)

    output_file = 'processed_data/processed_recode.csv'
    df.to_csv(output_file, index=False)

    return df


def out_for_LCA(df):
    #df = df.drop(['missing', 'cvrp_17', 'purch_lease', 'veh_model', 'tot_rebate', 'purch_price', 'air_dstrct', 'cnty', 'zipcode', 'census_trct'], axis=1)
    output_file = 'processed_data/LCA_input.csv'
    df.to_csv(output_file, index=False)
    return df


def LCA_in():

    def reverse_code(df):
        # df = pd.read_csv('processed_data/processed.csv')

        # previous technology
        df['prev_tech_type'].replace([1, 2], 'gas/diesel', inplace=True)
        df['prev_tech_type'].replace([3, 4, 5, 6, 7, 8, 9, -99], 'alt. fuel', inplace=True)

        # reason
        df['reason'].replace([1,2], 'saving $', inplace=True)
        df['reason'].replace([3], 'env. impact', inplace=True)
        df['reason'].replace([4], 'carpool', inplace=True)
        df['reason'].replace([5], 'enrgy ind.', inplace=True)
        df['reason'].replace([6], 'convenience', inplace=True)
        df['reason'].replace([7], 'performance', inplace=True)
        df['reason'].replace([8], 'styling', inplace=True)
        df['reason'].replace([9], 'new tech', inplace=True)
        df['reason'].replace([-99], 'other', inplace=True)

        # residence type
        df['residence_type'].replace([1], 'detached house', inplace=True)
        df['residence_type'].replace([2], 'attached house', inplace=True)
        df['residence_type'].replace([3], 'apt/condo', inplace=True)
        df['residence_type'].replace([-77, -99], 'other/NA', inplace=True)

        # solar panels
        df['solar_panels'].replace([1, 2], 'yes', inplace=True)
        df['solar_panels'].replace([3, 4, 5], 'no', inplace=True)
        df['solar_panels'].replace([-99], 'NA', inplace=True)

        # cars in household
        df['num_cars_HH'].replace([1], 'one', inplace=True)
        df['num_cars_HH'].replace([2], 'two', inplace=True)
        df['num_cars_HH'].replace([3], 'three', inplace=True)
        df['num_cars_HH'].replace([4], 'four or more', inplace=True)

        # licensed drivers household
        df['num_lic_drvr_HH'].replace([1], 'one', inplace=True)
        df['num_lic_drvr_HH'].replace([2], 'two', inplace=True)
        df['num_lic_drvr_HH'].replace([3], 'three', inplace=True)
        df['num_lic_drvr_HH'].replace([4, 5, 6, 7, 8, 9], 'four or more', inplace=True)

        # gender
        df['gender'].replace([1], 'male', inplace=True)
        df['gender'].replace([2], 'female', inplace=True)
        df['gender'].replace([3, -99, -77], 'NA', inplace=True)

        # HH_education
        df['HH_education'].replace([1], 'some HS', inplace=True)
        df['HH_education'].replace([2], 'HS grad', inplace=True)
        df['HH_education'].replace([3], 'some college', inplace=True)
        df['HH_education'].replace([4], 'assoc. deg.', inplace=True)
        df['HH_education'].replace([5], 'bach. deg.', inplace=True)
        df['HH_education'].replace([6], 'graduate. deg.', inplace=True)
        df['HH_education'].replace([-77], 'NA', inplace=True)

        # income
        df['HH_income'].replace([1], 'less $25k', inplace=True)
        df['HH_income'].replace([2], '$25-50k', inplace=True)
        df['HH_income'].replace([3], '$50-75k', inplace=True)
        df['HH_income'].replace([4], '$75-99k', inplace=True)
        df['HH_income'].replace([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 'grtr $100k', inplace=True)
        df['HH_income'].replace([-77], 'NA', inplace=True)

        # tax_filing_status
        df['tax_filing_status'].replace([1], 'single', inplace=True)
        df['tax_filing_status'].replace([2, 3], 'married', inplace=True)
        df['tax_filing_status'].replace([4, 5], 'other', inplace=True)
        df['tax_filing_status'].replace([-77], 'NA', inplace=True)

        # licensed drivers household
        df['HH_size'].replace([1], 'one', inplace=True)
        df['HH_size'].replace([2], 'two', inplace=True)
        df['HH_size'].replace([3], 'three', inplace=True)
        df['HH_size'].replace([4, 5, 6, 7, 8, 9], 'four or more', inplace=True)

        # ethnicity
        cols = ['amer_indian_alskn_native', 'blck_afr_amer', 'east_asian', 'latino', 'middl_eastern', 'hawii_pac_islnd',
         'south_asian', 'white', 'nan_ethnic', 'other_ethnic']
        df[cols].replace([1], 'yes', inplace=True)
        df[cols].replace([2], 'no', inplace=True)
        df = df.rename(columns={'nan_ethnic': 'NA'})

        # age
        df['age'].replace([1], '16-20', inplace=True)
        df['age'].replace([2], '21-29', inplace=True)
        df['age'].replace([3], '30-39', inplace=True)
        df['age'].replace([4], '40-49', inplace=True)
        df['age'].replace([5], '50-59', inplace=True)
        df['age'].replace([6], '60-69', inplace=True)
        df['age'].replace([7, 8], '70+', inplace=True)
        df['age'].replace([-77], 'NA', inplace=True)

        return df

    def filter_processed():
        df_pr = pd.read_csv('processed_data/processed.csv')
        df_rc = pd.read_csv('processed_data/processed_recode.csv')
        df_pr = df_pr[df_pr['cvrp_id'].isin(df_rc['cvrp_id'])]
        return df_pr.reset_index()

    df_lca = pd.read_csv('processed_data/R_output/lca5.csv', names=['class'], skiprows=1)
    df_pr = filter_processed()
    df_pr['class'] = df_lca['class']
    df_pr = reverse_code(df_pr)
    return df_pr


def mk_fig(tmp,var,air_distrct):
    cols = tmp.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    tmp = tmp[cols]
    tmp = tmp.round(3) * 100

    tmp.columns = ['All', 'Group-1', 'Group-2', 'Group-3', 'Group-4']
    ax = tmp.plot.bar(color=['black', 'indianred', 'forestgreen', 'cornflowerblue', 'gold'])
    ax.set_ylabel('% Of Group')
    plt.tight_layout()

    if air_distrct == 'All':
        out_name = 'figures/' + var + '.png'
    else:
        out_name = 'figures/' + air_distrct + '/' + var + '.png'

    plt.savefig(out_name)
    plt.close()
    return tmp


def crosstab_it(df, var, air_distrct):

    if var == 'HH_income':
        tmp = pd.crosstab(df[var], df['class'], normalize='columns', margins=True).reindex(['less $25k', '$25-50k', '$50-75k', '$75-99k', 'grtr $100k', 'NA'])
    elif var == 'tot_rebate':
        tmp = pd.crosstab(df[var], df['class'], normalize='columns', margins=True).reindex(['$3000', '$3500', '$4000', '$4500'])
    elif var == 'reason':
        tmp = pd.crosstab(df[var], df['class'], normalize='columns', margins=True).reindex(['carpool', 'convenience', 'enrgy ind.', 'env. impact', 'new tech', 'performance', 'saving $', 'styling', 'other'])
    elif var == 'residence_type':
        tmp = pd.crosstab(df[var], df['class'], normalize='columns', margins=True).reindex(['detached house', 'attached house', 'apt/condo', 'other/NA'])
    elif (var == 'num_cars_HH') | (var == 'num_lic_drvr_HH') | (var == 'HH_size'):
        tmp = pd.crosstab(df[var], df['class'], normalize='columns', margins=True).reindex(['one', 'two', 'three','four or more'])
    elif var == 'gender':
        tmp = pd.crosstab(df[var], df['class'], normalize='columns', margins=True).reindex(['male','female','NA'])
    elif var == 'HH_education':
        tmp = pd.crosstab(df[var], df['class'], normalize='columns', margins=True).reindex(['some HS', 'HS grad', 'some college', 'assoc. deg.', 'bach. deg.', 'graduate. deg.', 'NA'])
    elif var == 'tax_filing_status':
        tmp = pd.crosstab(df[var], df['class'], normalize='columns', margins=True).reindex(['single', 'married', 'other', 'NA'])
    elif var == 'air_dstrct':
        df = subset_by_AD(df, ['Bay Area', 'San Diego', 'South Coast','San Joaquin Valley Unified'])
        tmp = pd.crosstab(df[var], df['class'], normalize='columns', margins=True).reindex(['Bay Area' , 'San Diego', 'South Coast','San Joaquin Valley Unified'])

    else:
        tmp = pd.crosstab(df[var], df['class'], normalize='columns', margins=True)

    tmp = mk_fig(tmp, var, air_distrct)

    return tmp


def results_to_excel(air_distrct):
    def write_ethnic_table(writer,df,air_distrct):

        def agg_ethnic(grp):
            grp = grp.apply(pd.Series.value_counts).fillna(0).loc[1.0] / grp.shape[0]
            return grp

        df2 = df[['amer_indian_alskn_native', 'blck_afr_amer', 'east_asian', 'latino', 'middl_eastern', 'hawii_pac_islnd',
                  'south_asian', 'white', 'NA', 'other_ethnic', 'class']]
        df2 = df2.fillna(0)
        tmp = df2.groupby('class').apply(agg_ethnic).transpose()
        tmp['All'] = df2.apply(pd.Series.value_counts).fillna(0).loc[1.0] / df2.shape[0]
        tmp = tmp.drop('class')
        tmp = mk_fig(tmp, 'ethnicity',air_distrct)
        tmp.index.names = ['race/ethnicity']
        tmp.to_excel(writer, sheet_name='ethnicity_race')

    df = LCA_in()
    df = subset_by_AD(df, air_distrct)

    #crosstab tables: class versus variables
    writer = pd.ExcelWriter('RawResults.xlsx', engine='xlsxwriter')
    vars = ['age', 'HH_education', 'HH_income', 'HH_size']
    for i in vars:
        tmp = crosstab_it(df, i, air_distrct)
        tmp.to_excel(writer, sheet_name=i)

    write_ethnic_table(writer,df, air_distrct)

    vars = ['prev_tech_type', 'residence_type', 'tax_filing_status', 'reason', 'gender', 'purch_lease', 'veh_model', 'tot_rebate', 'solar_panels', 'num_cars_HH', 'num_lic_drvr_HH', 'tech_type', 'air_dstrct']
    for i in vars:
        tmp = crosstab_it(df, i, air_distrct)
        tmp.to_excel(writer, sheet_name=i)

    writer.save()


def subset_by_AD(df, air_distrct):
    if (air_distrct != 'All') & (isinstance(air_distrct,list)):
        df = df[df['air_dstrct'].isin(air_distrct)]

    if (air_distrct != 'All') & (not isinstance(air_distrct,list)):
        df = df[df['air_dstrct'].isin([air_distrct])]

    return df


def mk_AD_fig():
    df = LCA_in()
    df = df.replace('San Joaquin Valley Unified', 'San Joaquin Valley')
    air_dstrct = ['Bay Area', 'San Diego', 'South Coast', 'San Joaquin Valley']
    df = subset_by_AD(df, air_dstrct)


    tmp = df['air_dstrct'].value_counts()
    ax = tmp.plot.bar(color='darkgrey')
    ax.set_ylabel('% Of LMI Rebate Consumers')
    plt.tight_layout()
    out_name = 'figures/GroupByAD.png'
    plt.savefig(out_name)
    plt.close()


    color = color=[ 'indianred', 'forestgreen', 'cornflowerblue', 'gold']
    tmp = pd.crosstab(df['class'], df.air_dstrct, normalize='columns', margins=True).transpose()
    tmp.columns = ['Group-1', 'Group-2', 'Group-3', 'Group-4']
    ax = tmp.plot.bar(color=color)
    ax.legend(loc=1)
    ax.set_ylabel('% Of Region')
    plt.tight_layout()

    out_name = 'figures/GroupByAD.png'

    plt.savefig(out_name)
    plt.close()



