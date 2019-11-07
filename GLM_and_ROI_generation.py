#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 05:34:04 2017

@author: brain
"""

##### This scriptsruns GLM import os
import os
#import matplotlib
#matplotlib.use('Qt5Agg')
#import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
plt.ioff() #http://matplotlib.org/faq/usage_faq.html (interactive mode)
import numpy as np
import nibabel as nb
from nilearn import image
from nilearn.input_data import NiftiMasker
from nilearn.mass_univariate import permuted_ols
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
from sklearn import linear_model
from scipy import stats
from scikits.bootstrap import bootstrap
from sklearn.utils import resample as sk_resample
from sklearn.model_selection import StratifiedShuffleSplit
import subprocess
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from nipype.interfaces.fsl import Cluster
from scipy import stats
from nipype.interfaces.fsl import Threshold
from nipype.interfaces.fsl import ImageStats
from nipype.interfaces.fsl import Merge
from joblib import Parallel, delayed 
from nilearn import surface
#matplotlib.use('qt5agg')


#mpl.rcParams.update({'font.size': 7})
def Nan_ing_the_outliers_of_a_column(Table, Column_name, Outling_criteria = 3):
    Count_outliers = len(Table[Table[Column_name] > Table[Column_name].mean() + Outling_criteria*Table[Column_name].std()])
    Count_outliers = Count_outliers + len(Table[Table[Column_name] < Table[Column_name].mean() - Outling_criteria*Table[Column_name].std()])
    
    if Count_outliers > 0:
        Table.loc[Table[Column_name] > Table[Column_name].mean() + Outling_criteria*Table[Column_name].std(),Column_name]=np.nan
        Table.loc[Table[Column_name] < Table[Column_name].mean() - Outling_criteria*Table[Column_name].std(),Column_name]=np.nan
        
    return Table, Count_outliers





#### Read in the tables and create sub_samples:

## Help functions for 

def importing_table_from_csv(Table_DIR_full_path):
    #os.chdir('/Users/shahrzad/sciebo/JÃ¼lich_analysis_F1000/Fz100/')
    Original_data_table=pd.read_csv(Table_DIR_full_path, sep=',') #All_data_Fz100=pd.read_excel('FZJ100.xlsx')
    return Original_data_table

## End of help files for "cretae_test_specific_subsamples
def create_test_var_specific_subsamples(Base_working_dir,Main_Sample_info_table_CSV_full_path, Confounders_names,\
                                        test_variable_name, run ='', Other_important_variables =[],\
                                        exclusion_criteria = 'loose'):
    
    """ 
    #%%
    Other_important_variables: may include: 'T1_weighted_useful'
    
    
    exclusion_criteria= 'loose', 'moderate' or 'strict'
    So input fixed structure: 
    
    Base_working_dir: this is where my working directory is: /Shahrzad'sPersonalFolder/HCP/test_var_name_partial_corr/ 
    It assumes that the folder will be from now like this structured: 
        
      /Shahrzad'sPersonalFolder/Main_study/%test_var_name_partial_corr/
                                               /sample_CSV/main_sample_%test_var_name.csv      # Info_table_CSV
                                                           subsample_1_Subjects_info_table.csv
                                                           subsample_2_Subjects_info_table.csv
                                                           
    
    """
    Diagnosis_Exclusion = exclusion_criteria + "_exclusion"  
    ### Initialization:
    # Folders:
    # Base directory of analysis: 
    Working_dir = os.path.join(Base_working_dir, test_variable_name +'_partial_corr'+ run)
    
    try:
            os.makedirs(Working_dir)
    except OSError:
        if not os.path.isdir(Working_dir):
            raise
     
    # Where the Csv file needs to be located? (FIXED path) 
    Info_table_dir = os.path.join(Working_dir, 'sample_CSV')
    try:
            os.makedirs(Info_table_dir)
    except OSError:
        if not os.path.isdir(Info_table_dir):
            raise
    # name of Info table of variables of the sample it should be a csv (with Fix column names for sex, age and TIV) 
    Table_main_original = importing_table_from_csv(Main_Sample_info_table_CSV_full_path)
    # These are variables that should not be nan:
    Important_variables = [test_variable_name] + Confounders_names + Other_important_variables
    # keep only subjects that are not nan for all the variables in the Important_variables list_
    
    Table_main_filtered = Table_main_original.dropna(subset=[Important_variables])
    ### Here I exclude based on the Diagnosis table, subjects not healthy enough:
    if Diagnosis_Exclusion != '_exclusion':
        Table_main_filtered = Table_main_filtered[Table_main_filtered[Diagnosis_Exclusion].isin(["Exclude"]) == False].reset_index(drop = True)
    
    ## Now save this as main table of the sample. 
    Table_main_filtered_full_path = os.path.join(Info_table_dir, 'main_sample_'+test_variable_name+'.csv') 
    Table_main_filtered.to_csv(Table_main_filtered_full_path, index = False)

    return Table_main_filtered_full_path, Working_dir
##############
    

def read_Total_volumes_from_files(Image_top_DIR, Original_data_table_Fullpath,sample_name):
    
    Sample_Table = importing_table_from_csv(Original_data_table_Fullpath) #"/Users/shahrzad/sciebo/Jülich_analysis_F1000/Fz100/", 'FZJ100.xlsx')
    TIV_df = pd.DataFrame()
    
    if 'NKI' in sample_name:
        Subjects = Sample_Table['Subject'].tolist()
        Subj_col_name = 'Subject'
    elif 'HCP' in sample_name:
        Subjects = Sample_Table['ID'].tolist()
        Subj_col_name = 'ID'
    #Subjects_df = pd.DataFrame(data =Subjects,  columns = [Subj_col_name]) 
    s2 = pd.Series(np.nan)
    for Subj in np.arange(len(Subjects)).tolist():
        ##### Modified as tets does not have such folder structure#####
        if 'NKI' in sample_name:
            print(Subj)
            files_path = os.path.join(Image_top_DIR,Sample_Table['Subject'][Subj], Sample_Table['Session'][Subj], \
                                      'CAT/TIV_' + Sample_Table['Subject'][Subj] +'_'+ Sample_Table['Session'][Subj]+'_T1w.txt')
            
            if os.path.isfile(files_path):
                Temp_df = pd.read_csv(files_path, sep = " ", header=None)
                TIV_df = pd.concat([TIV_df, Temp_df], axis = 0)
            else:
                TIV_df = TIV_df.append(s2, ignore_index=True)
                print(files_path)
    
    TIV_df.reset_index(drop=True, inplace=True) 
    TIV_df.columns= ['TIV', 'TGM', 'TWM', 'TCSF', 'a', 'b', 'WMH'] 
    Sample_Table = pd.concat([Sample_Table, TIV_df[['TIV', 'TGM', 'TWM', 'TCSF']]], axis=1)
    Base_name = os.path.basename(Original_data_table_Fullpath)
    Base_dir = os.path.dirname(Original_data_table_Fullpath)
    TIV_added_Table_Full_path = os.path.join(Base_dir, 'TIV_added_'+ Base_name)
    Sample_Table.to_csv(TIV_added_Table_Full_path, sep=',', index = False)
    
    
    return TIV_added_Table_Full_path
    
    

## Several_split index generation:
def several_split_train_test_index_generation(subsampling_scripts_base_dir, Table_main_filtered_full_path, N_splits = 10, Sex_col_name = 'Sex',\
                                              Age_col_name = 'Age_current', Age_step_size = 10,\
                                              test_size = 0.5, gender_selection = None):
    os.chdir(subsampling_scripts_base_dir)
    import Mod_01_binning_shahrzad
    main_Table = importing_table_from_csv(Table_main_filtered_full_path)
    ### For the following code, I need sex: 1:female, 0:male
    #if len(main_Table[Sex_col_name].map({1.0: 0, 2.0: 1}).isnull()) >0:
    #main_Table[Sex_col_name+'_num'] = main_Table[Sex_col_name].map({"FEMALE": 0, "MALE": 1})
    try:
        a=main_Table[main_Table[Sex_col_name].str.startswith('F')][Sex_col_name].unique()[0]
    except:
        a=main_Table[main_Table[Sex_col_name].str.startswith('f')][Sex_col_name].unique()[0]
    try:
        b=main_Table[main_Table[Sex_col_name].str.startswith('M')][Sex_col_name].unique()[0]
    except:
        b=main_Table[main_Table[Sex_col_name].str.startswith('m')][Sex_col_name].unique()[0]
        
    main_Table[Sex_col_name+'_num'] = main_Table[Sex_col_name].map({a: -1, b: 1}) # This is new for the 
#    main_Table[Sex_col_name+'_num'] = main_Table[Sex_col_name].map({"F": -1, "M": 1}) # This is new for the 
    
    
    gender_specific_tag = 0
    
    if gender_selection == 'Male':
        
      main_Table= main_Table[main_Table[Sex_col_name+'_num'] == 1]
      main_Table = main_Table.reset_index(drop= True)
      gender_specific_tag = 1

    elif gender_selection == 'Female':
        
       main_Table= main_Table[main_Table[Sex_col_name+'_num'] == -1]
       main_Table = main_Table.reset_index(drop= True)
       gender_specific_tag = 1
    else:
        pass
    
    Age = Age_col_name
    
    min_age = main_Table[Age].min()

    max_age = main_Table[Age].max()
    main_Table, catrgory_num = Mod_01_binning_shahrzad.binning_age(main_Table, min_age, max_age,  Age, step_size= Age_step_size)
            
    female = main_Table[main_Table[Sex_col_name+'_num']==-1]
    male = main_Table[main_Table[Sex_col_name+'_num']==1]
    female_index = female.index.values
    female_age_group = np.array(female.age_group_num)
    female_test_size = test_size#round(len(female_index)/2)
    male_index = male.index.values
    male_age_group = np.array(male.age_group_num)
    male_test_size =  test_size#round(len(male_index)/2)
    female_splits = StratifiedShuffleSplit(n_splits=N_splits, test_size=female_test_size, random_state=0)
    female_splits.get_n_splits(female_index, female_age_group)
    split_female_index_df_train = pd.DataFrame(index=range(np.int(np.floor(len(female_index)*(1- female_test_size)))))
    split_female_index_df_test = pd.DataFrame(index= range(np.int(np.ceil(len(female_index)* (female_test_size)))))
    i = 0
    
    for female_train_index, female_test_index in female_splits.split(female_index, female_age_group):
        print("TRAIN:", female_train_index, "TEST:", female_test_index)
        split_female_index_df_train['train_'+str(i)]= female_index[female_train_index]
        split_female_index_df_test['test_'+str(i)] = female_index[female_test_index]
        
        i = i+1
        
        
    # male_splits
    male_splits = StratifiedShuffleSplit(n_splits=N_splits, test_size=male_test_size, random_state=0)
    male_splits.get_n_splits(male_index, male_age_group)
    split_male_index_df_train = pd.DataFrame(index=range(np.int(np.floor(len(male_index)*(1- male_test_size)))))
    split_male_index_df_test = pd.DataFrame(index= range(np.int(np.ceil(len(male_index)*(male_test_size)))))
    i = 0
    
    for male_train_index, male_test_index in male_splits.split(male_index, male_age_group):
        print("TRAIN:", male_train_index, "TEST:", male_test_index)
        split_male_index_df_train['train_'+str(i)]= male_index[male_train_index]
        split_male_index_df_test['test_'+str(i)] = male_index[male_test_index]
        
        i = i+1
        
    # merging indexes for male and female.
    train_frames = [split_female_index_df_train,split_male_index_df_train]   
    Train_index = pd.concat(train_frames, ignore_index= True)
    test_frames = [split_female_index_df_test,split_male_index_df_test]   
    Test_index = pd.concat(test_frames, ignore_index= True)

    
    return Train_index, Test_index
    
## Now several splits for ADNI (i.e. SITE matching)
def several_split_ADNI_Site_matched_train_test_index_generation(subsampling_scripts_base_dir, Table_main_filtered_full_path, N_splits = 10, Sex_col_name = 'PTGENDER',\
                                                           Age_col_name = 'AGE',SITE_col_name = '', n_bins_age = 2,\
                                                           test_size = 0.5, gender_selection = None):
    os.chdir(subsampling_scripts_base_dir)
    import Mod_01_binning_shahrzad
    main_Table = importing_table_from_csv(Table_main_filtered_full_path)
    ### For the following code, I need sex: 1:female, 0:male
    #if len(main_Table[Sex_col_name].map({1.0: 0, 2.0: 1}).isnull()) >0:
#    main_Table[Sex_col_name+'_num'] = main_Table[Sex_col_name].map({"Female": -1, "Male": 1})
    #main_Table[Sex_col_name+'_num'] = main_Table[Sex_col_name].map({"F": -1, "M": 1})
    try:
        a=main_Table[main_Table[Sex_col_name].str.startswith('F')][Sex_col_name].unique()[0]
    except:
        a=main_Table[main_Table[Sex_col_name].str.startswith('f')][Sex_col_name].unique()[0]
    try:
        b=main_Table[main_Table[Sex_col_name].str.startswith('M')][Sex_col_name].unique()[0]
    except:
        b=main_Table[main_Table[Sex_col_name].str.startswith('m')][Sex_col_name].unique()[0]
        
    main_Table[Sex_col_name+'_num'] = main_Table[Sex_col_name].map({a: -1, b: 1}) # This is new for the 

    gender_specific_tag = 0
    
    if gender_selection == 'Male':
        
      main_Table= main_Table[main_Table[Sex_col_name+'_num'] == 1]
      main_Table = main_Table.reset_index(drop= True)
      gender_specific_tag = 1

    elif gender_selection == 'Female':
        
       main_Table= main_Table[main_Table[Sex_col_name+'_num'] == -1]
       main_Table = main_Table.reset_index(drop= True)
       gender_specific_tag = 1
    else:
        pass
    
    Age = Age_col_name
    
    min_age = main_Table[Age].min()

    max_age = main_Table[Age].max()
    main_Table, catrgory_num = Mod_01_binning_shahrzad.binning_age(main_Table, min_age, max_age,  Age, step_size = 0, n_bins_age = n_bins_age)
        
    main_Table_young = main_Table[main_Table.age_group_num == 0] 
    main_Table_old =   main_Table[main_Table.age_group_num == 1] 
    female_young = main_Table_young[main_Table_young[Sex_col_name+'_num']==-1] 
    male_young = main_Table_young[main_Table_young[Sex_col_name+'_num']==1]
    
    # This ection discards all sites with less than 2 female or 2 male 
    f_SITE = np.array(female_young[SITE_col_name])
    for i in range(len(f_SITE)):
        if len(female_young[female_young[SITE_col_name] ==f_SITE[i]])<2:
            female_young= female_young[female_young[SITE_col_name] !=f_SITE[i]]
    m_SITE = np.array(male_young[SITE_col_name])
    for i in range(len(m_SITE)):
        if len(male_young[male_young[SITE_col_name] ==m_SITE[i]])<2:
            male_young= male_young[male_young[SITE_col_name] !=m_SITE[i]]
            
            
        
    female_young_index = female_young.index.values
    female_young_SITE = np.array(female_young[SITE_col_name])
    female_young_test_size = test_size#round(len(female_young_index)/2)
    male_young_index = male_young.index.values
    male_young_SITE = np.array(male_young[SITE_col_name])
    male_young_test_size =  test_size#round(len(male_young_index)/2)
    
    female_young_splits = StratifiedShuffleSplit(n_splits=N_splits, test_size=female_young_test_size, random_state=0)
    female_young_splits.get_n_splits(female_young_index, female_young_SITE)
    split_female_young_index_df_train = pd.DataFrame(index=range(np.int(np.floor(len(female_young_index)*(1- female_young_test_size)))))
    split_female_young_index_df_test = pd.DataFrame(index= range(np.int(np.ceil(len(female_young_index)* (female_young_test_size)))))
    i = 0
    
    for female_young_train_index, female_young_test_index in female_young_splits.split(female_young_index, female_young_SITE):
        print("TRAIN:", female_young_train_index, "TEST:", female_young_test_index)
        split_female_young_index_df_train['train_'+str(i)]= female_young_index[female_young_train_index]
        split_female_young_index_df_test['test_'+str(i)] = female_young_index[female_young_test_index]
        
        i = i+1
        
        
    # male_young_splits
    male_young_splits = StratifiedShuffleSplit(n_splits=N_splits, test_size=male_young_test_size, random_state=0)
    male_young_splits.get_n_splits(male_young_index, male_young_SITE)
    split_male_young_index_df_train = pd.DataFrame(index=range(np.int(np.floor(len(male_young_index)*(1- male_young_test_size)))))
    split_male_young_index_df_test = pd.DataFrame(index= range(np.int(np.ceil(len(male_young_index)*(male_young_test_size)))))
    i = 0
    
    for male_young_train_index, male_young_test_index in male_young_splits.split(male_young_index, male_young_SITE):
        print("TRAIN:", male_young_train_index, "TEST:", male_young_test_index)
        split_male_young_index_df_train['train_'+str(i)]= male_young_index[male_young_train_index]
        split_male_young_index_df_test['test_'+str(i)] = male_young_index[male_young_test_index]
        
        i = i+1
        
        
        
    female_old = main_Table_old[main_Table_old[Sex_col_name+'_num']==-1]
    male_old = main_Table_old[main_Table_old[Sex_col_name+'_num']==1]
    
        # This ection discards all sites with less than 2 female or 2 male 
    f_O_SITE = np.array(female_old[SITE_col_name])
    for i in range(len(f_O_SITE)):
        if len(female_old[female_old[SITE_col_name] ==f_O_SITE[i]])<2:
            female_old= female_old[female_old[SITE_col_name] !=f_O_SITE[i]]
    m_O_SITE = np.array(male_old[SITE_col_name])
    for i in range(len(m_O_SITE)):
        if len(male_old[male_old[SITE_col_name] ==m_O_SITE[i]])<2:
            male_old= male_old[male_old[SITE_col_name] !=m_O_SITE[i]]
            

    
    
    female_old_index = female_old.index.values
    female_old_SITE = np.array(female_old[SITE_col_name])
    female_old_test_size = test_size#round(len(female_old_index)/2)
    male_old_index = male_old.index.values
    male_old_SITE = np.array(male_old[SITE_col_name])
    male_old_test_size =  test_size#round(len(male_old_index)/2)
    female_old_splits = StratifiedShuffleSplit(n_splits=N_splits, test_size=female_old_test_size, random_state=0)
    female_old_splits.get_n_splits(female_old_index, female_old_SITE)
    split_female_old_index_df_train = pd.DataFrame(index=range(np.int(np.floor(len(female_old_index)*(1- female_old_test_size)))))
    split_female_old_index_df_test = pd.DataFrame(index= range(np.int(np.ceil(len(female_old_index)* (female_old_test_size)))))
    i = 0
    
    for female_old_train_index, female_old_test_index in female_old_splits.split(female_old_index, female_old_SITE):
        print("TRAIN:", female_old_train_index, "TEST:", female_old_test_index)
        split_female_old_index_df_train['train_'+str(i)]= female_old_index[female_old_train_index]
        split_female_old_index_df_test['test_'+str(i)] = female_old_index[female_old_test_index]
        
        i = i+1
        
        
    # male_old_splits
    male_old_splits = StratifiedShuffleSplit(n_splits=N_splits, test_size=male_old_test_size, random_state=0)
    male_old_splits.get_n_splits(male_old_index, male_old_SITE)
    split_male_old_index_df_train = pd.DataFrame(index=range(np.int(np.floor(len(male_old_index)*(1- male_old_test_size)))))
    split_male_old_index_df_test = pd.DataFrame(index= range(np.int(np.ceil(len(male_old_index)*(male_old_test_size)))))
    i = 0
    
    for male_old_train_index, male_old_test_index in male_old_splits.split(male_old_index, male_old_SITE):
        print("TRAIN:", male_old_train_index, "TEST:", male_old_test_index)
        split_male_old_index_df_train['train_'+str(i)]= male_old_index[male_old_train_index]
        split_male_old_index_df_test['test_'+str(i)] = male_old_index[male_old_test_index]
        
        i = i+1
        
    # merging indexes for male_old and female_old.
    train_frames = [split_female_old_index_df_train,split_male_old_index_df_train, split_female_young_index_df_train,split_male_young_index_df_train]   
    Train_index = pd.concat(train_frames, ignore_index= True)
    test_frames = [split_female_old_index_df_test,split_male_old_index_df_test,split_female_young_index_df_test,split_male_young_index_df_test]   
    Test_index = pd.concat(test_frames, ignore_index= True)

    
    return Train_index, Test_index
    
    

        

    
    
#### Create subsamples

    
def train_test_sample_generation(Table_main_filtered_full_path, subsampling_scripts_base_dir, Sex_col_name = 'Sex',\
                                 Age_col_name = 'Age_current',VarX_name = '', Age_step_size = 10, n_bins_age = 0,\
                                 n_bins_VarX = 0, gender_selection = None):
    ### Matchfor age, sex groupping
    os.chdir(subsampling_scripts_base_dir)
    import Mod_01_binning_shahrzad
    main_Table = importing_table_from_csv(Table_main_filtered_full_path)
    ### For the following code, I need sex: 1:female, 0:male
    #if len(main_Table[Sex_col_name].map({1.0: 0, 2.0: 1}).isnull()) >0:
    main_Table[Sex_col_name+'_num'] = main_Table[Sex_col_name].map({"FEMALE": -1, "MALE": 1})
    gender_specific_tag = 0
    
    if gender_selection == 'Male':
        
      main_Table= main_Table[main_Table[Sex_col_name+'_num'] == -1]
      main_Table = main_Table.reset_index(drop= True)
      gender_specific_tag = 1

    elif gender_selection == 'Female':
        
       main_Table= main_Table[main_Table[Sex_col_name+'_num'] == 1]
       main_Table = main_Table.reset_index(drop= True)
       gender_specific_tag = 1
    else:
        pass
    
    
    First_group , Second_group = Mod_01_binning_shahrzad.Create_ageAndsexAndVarx_matched_sugroups_from_DF(DF=main_Table,Age = Age_col_name, \
                                                                                                          sex = Sex_col_name+'_num',\
                                                                                                          VarX = VarX_name, \
                                                                                                          step_size_age=Age_step_size, \
                                                                                                          n_bins_age=n_bins_age, \
                                                                                                          n_bins_VarX=n_bins_VarX)
    
    
    main_Table['Which_sample'] = np.zeros(len(main_Table))
    main_Table.loc[main_Table.index.isin(First_group.index),'Which_sample'] = 1
    main_Table.loc[main_Table.index.isin(Second_group.index),'Which_sample'] = 2
    main_Table.loc[(main_Table['Which_sample'] == 0) == True,'Which_sample'] = 3  # This refers to all the subjects that are in neither groups. So, to get all subjects that are in 
    
    First_group = First_group.reset_index(drop= True)
    Second_group = Second_group.reset_index(drop= True)
    
    new_samples_base_name = os.path.basename(Table_main_filtered_full_path).strip("main_sample_")
    #main_table_gender_specific_full_path = Table_main_filtered_full_path
    
    main_table_gender_specific_full_path = os.path.join(os.path.dirname(Table_main_filtered_full_path),'grouped_main_sample_' + new_samples_base_name  )
    main_Table.to_csv(main_table_gender_specific_full_path, index = False)
    first_group_full_path = os.path.join(os.path.dirname(Table_main_filtered_full_path),'Sample_1_' + new_samples_base_name)
    First_group.to_csv(first_group_full_path, sep=',', index = False)
    second_group_full_path = os.path.join(os.path.dirname(Table_main_filtered_full_path),'Sample_2_' + new_samples_base_name)
    Second_group.to_csv(second_group_full_path, sep=',', index = False)
    
    return first_group_full_path, second_group_full_path, main_table_gender_specific_full_path


######

def Create_gray_matter_map_filenames(Image_top_DIR, Sample_Table, Sample_name, modulation_method = 'non_linearOnly', Smoothing_kernel_FWHM=8):
    
    ''' 
    This is a very bad way of defining the preprocessed file paths, but I need to keep it like this until Felix/I make 
    a simple, same strcture for all the preprocessed images
    '''
    if 'NKI' in Sample_name:
        Subjects = Sample_Table['Subject'].tolist()    
    elif 'HCP' in Sample_name:
        Subjects = Sample_Table['Subject'].tolist()
        #Subjects = Sample_Table['ID'].tolist()
    elif Sample_name == 'ADNI':
        Subjects = Sample_Table['subject_folder'].tolist()
    elif Sample_name == 'ADNI_new':
        Subjects = Sample_Table['SubjectID'].tolist()
    elif Sample_name == 'SCZ':
        Subjects = Sample_Table['participant_id'].tolist()
    elif Sample_name=='':
        
        Subjects = Sample_Table['Subject'].tolist()    

        
        
        
    if modulation_method == 'non_linearOnly':
       smoothed_preprocessed_prefix =  'sm0wp1'
       
    elif modulation_method == 'fully_modulated':
       smoothed_preprocessed_prefix =  'smwp1'
   
    #i=0
    files_list =[]
    for Subj in np.arange(len(Subjects)).tolist():
        ##### Modified as tets does not have such folder structure#####
        if 'NKI' in Sample_name:
            files_list.append(os.path.join(Image_top_DIR,Sample_Table['Subject'][Subj], Sample_Table['Session'][Subj], \
                                           'CAT/mri/' + smoothed_preprocessed_prefix + Sample_Table['Subject'][Subj] +'_'+ Sample_Table['Session'][Subj]+'_T1w.nii'))
        elif Sample_name =='HCP':
            ##This is because VBM8 apparently had an extra -r in the naming
#            if modulation_method == 'non_linearOnly':
#                smoothed_preprocessed_prefix =  'sm0wrp1'
#            elif modulation_method == 'fully_modulated':
#                smoothed_preprocessed_prefix =  'smwrp1'       
#           
#            files_list.append(os.path.join(Image_top_DIR,Sample_Table['ID'][Subj], '3D/'+smoothed_preprocessed_prefix+ Sample_Table['ID'][Subj]+ '.nii'))
#            #/data/Heisenberg_HDD/MultiState/DATA/HCP/100610/3D/sm0wrp1100610.nii 
                        ##This is felix's new naming
            if modulation_method == 'non_linearOnly':
                if Smoothing_kernel_FWHM >0:
                    smoothed_preprocessed_prefix =  's'+ str(Smoothing_kernel_FWHM) +'m0wp1'
                else: 
                    smoothed_preprocessed_prefix =  'm0wp1'
            elif modulation_method == 'fully_modulated':
                if Smoothing_kernel_FWHM >0:
                    smoothed_preprocessed_prefix =  's'+ str(Smoothing_kernel_FWHM) +'mwp1'
                else:
                    smoothed_preprocessed_prefix = 'mwp1'
            
            files_list.append(os.path.join(Image_top_DIR,str(Sample_Table['Subject'][Subj]),'mri',smoothed_preprocessed_prefix+ str(Sample_Table['Subject'][Subj]) + '.nii'))
        elif Sample_name == 'SCZ':
            if modulation_method == 'non_linearOnly':
                if Smoothing_kernel_FWHM >0:
                    smoothed_preprocessed_prefix =  's'+ str(Smoothing_kernel_FWHM) +'m0wp1'
                else: 
                    smoothed_preprocessed_prefix =  'm0wp1'
            elif modulation_method == 'fully_modulated':
                if Smoothing_kernel_FWHM >0:
                    smoothed_preprocessed_prefix =  's'+ str(Smoothing_kernel_FWHM) +'mwp1'
                else:
                    smoothed_preprocessed_prefix = 'mwp1'
            files_list.append(os.path.join(Image_top_DIR, str(Sample_Table['site_detail'][Subj]),str(Sample_Table['participant_id'][Subj]), 'mri',smoothed_preprocessed_prefix+ str(Sample_Table['participant_id'][Subj])+ '.nii'))
        elif Sample_name =='':
            ##This is because VBM8 apparently had an extra -r in the naming
#            if modulation_method == 'non_linearOnly':
#                smoothed_preprocessed_prefix =  'sm0wrp1'
#            elif modulation_method == 'fully_modulated':
#                smoothed_preprocessed_prefix =  'smwrp1'       
#           
#            files_list.append(os.path.join(Image_top_DIR,Sample_Table['ID'][Subj], '3D/'+smoothed_preprocessed_prefix+ Sample_Table['ID'][Subj]+ '.nii'))
#            #/data/Heisenberg_HDD/MultiState/DATA/HCP/100610/3D/sm0wrp1100610.nii 
                        ##This is felix's new naming
            if modulation_method == 'non_linearOnly':
                if Smoothing_kernel_FWHM >0:
                    smoothed_preprocessed_prefix =  's'+ str(Smoothing_kernel_FWHM) +'m0wp1'
                else: 
                    smoothed_preprocessed_prefix =  'm0wp1'
            elif modulation_method == 'fully_modulated':
                if Smoothing_kernel_FWHM >0:
                    smoothed_preprocessed_prefix =  's'+ str(Smoothing_kernel_FWHM) +'mwp1'
                else:
                    smoothed_preprocessed_prefix = 'mwp1'
            
            files_list.append(os.path.join(Image_top_DIR,str(Sample_Table['Subject'][Subj]),'mri',smoothed_preprocessed_prefix+ str(Sample_Table['Subject'][Subj]) + '.nii'))


            
        elif 'SPM12_HCP' in Sample_name:
            files_list.append(os.path.join(Image_top_DIR,Sample_Table['ID'][Subj], 'CAT/mri/'+smoothed_preprocessed_prefix+ Sample_Table['ID'][Subj]+ '.nii'))
        elif Sample_name == 'ADNI':
            files_list.append(os.path.join(Image_top_DIR,Sample_Table.subject_folder[Subj],Sample_Table.Imaging_sess[Subj], \
                                           'mri/Dartel_Template_1_IXI555_MNI152_rr1.2_default', \
                                           smoothed_preprocessed_prefix+ Sample_Table.subject_folder[Subj]+'_'+Sample_Table.Imaging_sess[Subj]+'.nii'))
        elif Sample_name == 'ADNI_new':
            ##This is felix's new naming
            if modulation_method == 'non_linearOnly':
                smoothed_preprocessed_prefix =  's'+ str(Smoothing_kernel_FWHM) +'m0wp1'
            elif modulation_method == 'fully_modulated':
                smoothed_preprocessed_prefix =  's'+ str(Smoothing_kernel_FWHM) +'mwp1'
            files_list.append(os.path.join(Image_top_DIR,Sample_Table.SubjectID[Subj],Sample_Table.SessionID[Subj],\
                                           'mri', smoothed_preprocessed_prefix+ Sample_Table.SubjectID[Subj]+'_'+Sample_Table.SessionID[Subj]+'.nii'))
                    
                    
        #files_list.append(os.path.join(Image_top_DIR,'smwp1'+Sample_Table['T-No.'][Subj]+'.nii'))
    
    #####END of modification######
    
    Files_datafram = pd.DataFrame()
    Files_datafram['NIFTI_file_path_and_name'] = files_list

    return Files_datafram, files_list

def create_merged_file(files_list, save_dir, merged_file_name): # 
    #os.chdir(save_dir)
    #merger=Merge()
    #merger.inputs.in_files=files_list
    #merger.inputs.dimension='t'
    #merger.inputs.merged_file= merged_file_name + '.nii.gz'
   # merger.inputs.output_type='NIFTI_GZ'
    cmd_merge = "/bin/bash -c 'source /usr/share/fsl/5.0/etc/fslconf/fsl.sh; fslmerge -t " + os.path.join(save_dir, merged_file_name + '.nii.gz') + " `cat "+ files_list +"`'"
       
    subprocess.call(cmd_merge, shell=True)
            #print(merger.cmdline)
    #merger.run()
    return os.path.join(save_dir, merged_file_name + '.nii.gz')

def load_NIFTI_DATA_FROM_TABLE(Image_top_DIR, Original_data_table_with_TIV,merge_save_dir , merge_file_name, Mask_file, sample_name, \
                               modulation_method = 'non_linearOnly', load_nifti_masker_Flag=0, Smoothing_kernel_FWHM=8):
    
    Table = Original_data_table_with_TIV #"/Users/shahrzad/sciebo/Jülich_analysis_F1000/Fz100/", 'FZJ100.xlsx')
    gray_matter_map_filenames, files_list  = Create_gray_matter_map_filenames(Image_top_DIR, Table, sample_name,modulation_method, Smoothing_kernel_FWHM) # Creates a table and also a list for filenames
    
    with open(os.path.join(merge_save_dir, merge_file_name+'_list.txt'), 'w') as f:
        for item in files_list:
            f.write("%s\n" % item)
    nifti_masker = NiftiMasker(standardize=False, mask_img=Mask_file, memory='nilearn_cache')
    merged_file_path=create_merged_file(os.path.join(merge_save_dir, merge_file_name+'_list.txt'), merge_save_dir, merge_file_name)
    gm_maps_masked = ''
    D = ''
    
    if load_nifti_masker_Flag ==1:
        gm_maps_masked = nifti_masker.fit_transform(merged_file_path)
        D = nifti_masker.inverse_transform(gm_maps_masked)
    output = {'merged_file_path':merged_file_path, 'files_list':files_list, 'nifti_masker': nifti_masker,'D': D,'gm_maps_masked':gm_maps_masked}
        #D.to_filename(os.path.join(Stats_saving_DIR, 'GM_'+ Stats_saving_file_name))
    return output

####### LOAD merged_thickness_FROM_TABLE

def create_merged_thickness_FROM_TABLE(Subjects_DIR, Sample_Table, merge_save_dir, merge_base_file_name, Sample_name = '', \
                                       smoothing_kernel_FWHM = 10, meas = 'thickness', Target_surface = 'fsaverage'):
    #INPUTS:
    # SUBJECTSDIR = Subjects_DIR
    # smoothing kernel FWHM mm.
    # IMPORTANT: fsaverage dir is the same as the SUBJECTSDIR
    # tables to get Ids from 
    
    #A make a list of all IDs that need to be merged. 
    #B Save it in a text file 
    #C use mri_preproc command to generate a smoothed merged file 
    if 'NKI' in Sample_name:
        Subjects = Sample_Table['Subject'].tolist()    
    elif 'HCP' in Sample_name:
        Subjects = Sample_Table['Subject'].tolist()
    elif Sample_name == 'ADNI':
        Subjects = Sample_Table['subject_folder'].tolist()
    elif Sample_name == 'ADNI_new':
        Subjects = Sample_Table['SubjectID'].tolist()
    elif Sample_name=='':
        
        Subjects = Sample_Table['Subject'].tolist()    

        
   
    #i=0
    files_list =[]
    for Subj in np.arange(len(Subjects)).tolist():
        if 'HCP' in Sample_name:
            files_list.append(Sample_Table['Subject'][Subj])
    Ids_file_path = os.path.join(merge_save_dir, merge_base_file_name + '_id.txt')
    with open(Ids_file_path, 'w') as f:
        for item in files_list:
            f.write('%s\n' % item)
    
    # Now I need to basically make commands for bash and run  it to create the merged files:
    merged_smoothed_filemname_lh = merge_base_file_name + '_' + meas + '_lh.sm' + str(smoothing_kernel_FWHM) + '.mgz'
    FULL_path_merged_smoothed_filemname_lh = os.path.join(merge_save_dir, merged_smoothed_filemname_lh)
    merged_smoothed_filemname_rh = merge_base_file_name + '_' + meas + '_rh.sm' + str(smoothing_kernel_FWHM) + '.mgz'
    FULL_path_merged_smoothed_filemname_rh = os.path.join(merge_save_dir, merged_smoothed_filemname_rh)
    cmd_FSenv = 'export SUBJECTS_DIR=' + Subjects_DIR
    cmd_preproc_left = 'mris_preproc --f ' + Ids_file_path + ' --target ' + Target_surface + ' --hemi lh --meas ' + meas + ' --fwhm ' + str(smoothing_kernel_FWHM) + ' --out ' + FULL_path_merged_smoothed_filemname_lh
    cmd_preproc_right = 'mris_preproc --f ' + Ids_file_path + ' --target ' + Target_surface + ' --hemi rh --meas ' + meas + ' --fwhm ' + str(smoothing_kernel_FWHM) + ' --out ' + FULL_path_merged_smoothed_filemname_rh
    cmd_c = cmd_FSenv + ' ; ' + cmd_preproc_left + ' ; ' + cmd_preproc_right
    subprocess.call(cmd_c, shell=True)        
    output = {'left_merged_file_path':FULL_path_merged_smoothed_filemname_lh, 'right_merged_file_path':FULL_path_merged_smoothed_filemname_rh, 'files_list':files_list}
        #D.to_filename(os.path.join(Stats_saving_DIR, 'GM_'+ Stats_saving_file_name))
    return output
        
#%%    
def Create_GLM_columns(Table_dataframe, test_variables, confounders):

    variables_of_interest = np.array(Table_dataframe[test_variables])
    confounding_vars = np.array(Table_dataframe[confounders])
    return variables_of_interest, confounding_vars


def permutation_based_stats_and_saving_p_vals(gm_maps_masked, nifti_masker, variables_of_interest, confounding_vars, n_perm, n_jobs, \
                                              Stats_saving_DIR, test_var_name, Stats_saving_file_name):

    variance_threshold = VarianceThreshold(threshold=0.00001)
    data = variance_threshold.fit_transform(gm_maps_masked)
    neg_log_pvals, t_scores_original_data, h0_fmax = permuted_ols(
        tested_vars=variables_of_interest, target_vars=data, confounding_vars=confounding_vars,  # + intercept as a covariate by default
        model_intercept =True, n_perm=n_perm,  # 1,000 in the interest of time; 10000 would be better
        n_jobs=n_jobs) # It can be used to use more than one cpu!
    signed_neg_log_pvals = neg_log_pvals * np.sign(t_scores_original_data)
    signed_neg_log_pvals_unmasked = nifti_masker.inverse_transform(
        variance_threshold.inverse_transform(signed_neg_log_pvals))
    t_scores_original_data_unmasked = nifti_masker.inverse_transform(
        variance_threshold.inverse_transform(t_scores_original_data))
    #t_scores_original_data_unmasked.header.set_data_offset(0)
    #t_scores_original_data_unmasked.header._structarr['vox_offset'] = 0
    T_file_name = os.path.join(Stats_saving_DIR, 'T_map_'+ test_var_name +'_'+ Stats_saving_file_name + '.nii.gz')
    P_file_name = os.path.join(Stats_saving_DIR, 'p_val_'+ test_var_name +'_'+ Stats_saving_file_name + '.nii.gz')
    try:
        os.remove(T_file_name)
    except OSError:
        pass
    try:
        os.remove(P_file_name)
    except OSError:
        pass
    signed_neg_log_pvals_unmasked.to_filename(P_file_name)
    t_scores_original_data_unmasked.to_filename(T_file_name)
    return h0_fmax , T_file_name, P_file_name

def Create_design_text(design_saving_dir, variables_of_interest, confounders = 0.00, add_intercept = False):
    ''' 
    design_saving_dir: usually is similar to where T-tests are located. 
    
    
    output:
        Full path of design.txt file
    '''
    
        
    if len(np.shape(variables_of_interest)) == 1:
        variables_of_interest = variables_of_interest[:, np.newaxis]
    
    if len(np.shape(confounders)) == 1:
        confounders = confounders[:, np.newaxis]
    
    if len(np.shape(confounders)) > 0:
        if add_intercept == True:
            intercept_arr = np.ones((variables_of_interest.shape[0], 1))
            Temp = np.hstack((intercept_arr , variables_of_interest, confounders))
        else:
            Temp = np.hstack((variables_of_interest, confounders))
        design_text_fullpath = os.path.join(design_saving_dir, 'design.txt')
        np.savetxt(design_text_fullpath, Temp)
        
    else:
        if add_intercept == True:
            intercept_arr = np.ones((variables_of_interest.shape[0], 1))
            Temp = np.hstack((intercept_arr , variables_of_interest))
        else:
            Temp = variables_of_interest
        design_text_fullpath = os.path.join(design_saving_dir, 'design.txt')
        np.savetxt(design_text_fullpath, Temp)
    return design_text_fullpath
        
def Create_contrast_text(design_saving_dir, variables_of_interest, confounders = 0.00, add_intercept = False):
    ''' 
    This function at the moment only works for one test_variable. 
    design_saving_dir: usually is similar to where T-tests are located. 
    
    
    output:
        Full path of contrast.txt file
    '''

        
    if len(np.shape(variables_of_interest)) == 1:
        variables_of_interest = variables_of_interest[:, np.newaxis]
    
    if len(np.shape(confounders)) == 1:
        confounders = confounders[:, np.newaxis]
    
    if len(np.shape(confounders)) > 0:
        if add_intercept == True:
            Temp_pos= np.array([0] + [1] +  [0 for i in np.arange(np.shape(confounders)[1])])
            
            Temp_neg= np.array([0] + [-1] +  [0 for i in np.arange(np.shape(confounders)[1])])
        
            
        else:
            Temp_pos= np.array([1] +  [0 for i in np.arange(np.shape(confounders)[1])])
            
            Temp_neg= np.array([-1] +  [0 for i in np.arange(np.shape(confounders)[1])])
        
        Temp = np.vstack((Temp_pos, Temp_neg))
        
        contrast_text_fullpath = os.path.join(design_saving_dir, 'contrast.txt')
        np.savetxt(contrast_text_fullpath, Temp)
        
    else:
        if add_intercept == True:
            Temp_pos= np.array([0] + [1])
            Temp_neg= np.array([0] + [-1])
        else:
            Temp_pos= np.array([1])
            
            Temp_neg= np.array([-1])
        
        Temp = np.vstack((Temp_pos, Temp_neg))
        
        contrast_text_fullpath = os.path.join(design_saving_dir, 'contrast.txt')
        np.savetxt(contrast_text_fullpath, Temp)
    return contrast_text_fullpath


def Separate_nilearn_Tstats_maps_to_sides(T_file_name, Stats_saving_DIR, Base_File_name_Stats, test_var_name):
    
    """
    gets a T_map of nilearn (Including both positive and negative T-values (raw)) and cuts it into
    two maps: 
        t1: looking at positive t-values only (i.e. regions  positively correlated with test var)
        t2: looking at negative t-values only (i.e. regions  negatively correlated with test var)
        t1 & t2 at the end of this function should only consist of positive values.
        Fixed outut names: %test_var_name_nilearn_stats_'tstat1/2.nii.gz'  # Fixed Outputs saved
        
    """        
        
    
    for j in np.arange(2).tolist():
        
        
        if j == 0:
            
            t1_map_name = test_var_name + '_' + Base_File_name_Stats+'_' + 'tstat1.nii.gz'
            Temp_T_contrast = image.math_img("1*img", img = T_file_name)
            t1_full_path = os.path.join(Stats_saving_DIR, t1_map_name)
            Temp_T_contrast.to_filename(t1_full_path)
        elif j ==1:
            t2_map_name = test_var_name + '_' + Base_File_name_Stats+'_' + 'tstat2.nii.gz'
            inverted_Stats_image = image.math_img("-1*img", img = T_file_name)
            t2_full_path = os.path.join(Stats_saving_DIR, t2_map_name)
            inverted_Stats_image.to_filename(t2_full_path)
            

    return t1_full_path, t2_full_path

def Separate_nilearn_corrPstats_maps_to_sides(P_file_name, Stats_saving_DIR,Base_File_name_Stats, test_var_name):
    """
    
    gets a P_map of nilearn (negative_log_signed_FWE_cluster level corrected p-map) and cuts it into
    two maps: 
        p1: looking at positive p-values only (i.e. regions significantly positively correlated with test var)
        p2: looking at negative p-values only (i.e. regions significantly negatively correlated with test var)
        P1 & p2 at the end of this function should only consist of positive values.
        Fixed outputs saved:  %test_var_name_nilearn_stats_FWE_corrp_tstat1/2.nii.gz  # Fixed Outputs saved 
    """
    
    
    for j in np.arange(2).tolist():
                
        if j == 0:
            p1_map_name = test_var_name + '_' + Base_File_name_Stats+'_' + 'FWE_corrp_tstat1.nii.gz'
            p1 = image.math_img("(img >=0)*img", img = P_file_name)
            p1_full_path = os.path.join(Stats_saving_DIR, p1_map_name)
            p1.to_filename(p1_full_path)
        elif j ==1:
            p2_map_name = test_var_name + '_' + Base_File_name_Stats+'_' + 'FWE_corrp_tstat2.nii.gz'
            p2 = image.math_img("-1*(img <=0)* img", img = P_file_name)
            p2_full_path = os.path.join(Stats_saving_DIR, p2_map_name)
            p2.to_filename(p2_full_path)
    return p1_full_path, p2_full_path
 
def GLM_on_Sample(Base_working_dir,Sample_info_table_CSV_full_path, Confounders_names, test_variables_name,\
                  merged_image_name,Mask_file_complete, merged_Flag = 1, Image_top_DIR ="", mod_method = "non_linearOnly",\
                  Base_Sample_name ='NKI', n_perm = 100, n_core = 1, Flag_TFCE = 0, SLURM_Que = False, Template_Submision_script_path = ''):
    """
    #%%
    
    So input fixed structure: 
    
    Base_working_dir: this is where my working directory is: /Shahrzad'sPersonalFolder/HCP/Partial_cors/ 
    It assumes that the folder will be from now like this structured: 
        
      /Shahrzad'sPersonalFolder/Main_study/%test_var_name_partial_corr/
                                               /sample_CSV/main_sample_%test_var_name.csv      # Info_table_CSV
                                                           sample_1_Subjects_info_table.csv
                                                           sample_2_Subjects_info_table.csv
     
                                               /4D_images/merged_image_name.nii.gz      # merged 4D images of the subsamples
                                               /GLM_Stats_dir/                          # Stats of GLM which are going to be used to create ROI
                                                             %test_var_name/
                                                                            fsl or nilearn/
                                                                                           %test_var_name_nilearn_stats_FWE_corrp_tstat1/2.nii.gz  # Fixed Outputs saved  
                                                                                           %test_var_name_nilearn_stats_tstat1/2.nii.gz  # Fixed Outputs saved   
                                                                                          Or
                                                                                           %test_var_name_fsl_stats_tfce_corrp_tstat1/2.nii.gz  # Fixed Outputs saved   
                                                                                           %test_var_name_fsl_stats_tstat1/2.nii.gz     # Fixed Outputs saved
    
     1: means positive correlation with test_var and 2 means negative correlations
    
    Outputs of this function are the full path of the following Fixed structure:
        ************************************************************************************************************  
        Voxel-wise t-stats map of possitive association between GMV and %test_var_name:
        t1_full_path:
            nilearn: 
             /Current_Working_dir/GLM_Stats_dir/%test_var_name/nilearn/%test_var_name_nilearn_stats_tstat1.nii.gz
            fsl:
             /Current_Working_dir/GLM_Stats_dir/%test_var_name/fsl/%test_var_name_fsl_stats_tstat1.nii.gz
        ************************************************************************************************************
        Corrected p-values of of possitive association between GMV and %test_var_name:
        p1_full_path:
            nilearn (Cluster_level FWE corrected): 
             /Current_Working_dir/GLM_Stats_dir/%test_var_name/nilearn/%test_var_name_nilearn_stats_FWE_corrp_tstat1.nii.gz
            fsl:
             /Current_Working_dir/GLM_Stats_dir/%test_var_name/fsl/%test_var_name_fsl_stats_tfce_corrp_tstat1.nii.gz
        ************************************************************************************************************    
        Voxel-wise t-stats map of Negative association between GMV and %test_var_name:
        t2_full_path:
            nilearn: 
             /Current_Working_dir/GLM_Stats_dir/%test_var_name/nilearn/%test_var_name_nilearn_stats_tstat1.nii.gz
            fsl:
             /Current_Working_dir/GLM_Stats_dir/%test_var_name/fsl/%test_var_name_fsl_stats_tstat1.nii.gz
        ************************************************************************************************************
        Corrected p-values of of Negative association between GMV and %test_var_name:
        p2_full_path:
            nilearn (Cluster_level FWE corrected): 
             /Current_Working_dir/GLM_Stats_dir/%test_var_name/nilearn/%test_var_name_nilearn_stats_FWE_corrp_tstat2.nii.gz
            fsl (TFCE corrected):
             /Current_Working_dir/GLM_Stats_dir/%test_var_name/fsl/%test_var_name_fsl_stats_tfce_corrp_tstat2.nii.gz
    

    
    """
    ### Initialization:
    # Folders:
    # Base directory of analysis: 
    Base_working_dir = Base_working_dir
    # Where the Csv file is located? (FIXED path) Not needed though
    Info_table_dir = os.path.join(Base_working_dir, 'sample_CSV')
    # Where the 4D image file is located? or should be located(this also should be Fixed, depnding only to the base dir)
    
    merged_sample_dir = os.path.join(Base_working_dir, '4D_images')
    
    try:
            os.makedirs(merged_sample_dir)
    except OSError:
        if not os.path.isdir(merged_sample_dir):
            raise
    # Where should the GLM stats be saved (this also should be Fixed, depnding only to the base dir)
    GLM_Stats_dir = os.path.join(Base_working_dir, 'GLM_Stats_dir')
    try:
            os.makedirs(GLM_Stats_dir)
    except OSError:
        if not os.path.isdir(GLM_Stats_dir):
            raise
    
    # name of Info table of variables of the sample it should be a csv (with Fix column names for sex, age and TIV) 
    
    Sample_info_table = pd.read_csv(Sample_info_table_CSV_full_path, delimiter=',')
    # name of the 4D file of the sample ---- It should also be a fixed name : lets say 4D_image_GLM_sample.nii
    merged_image_name = merged_image_name
    Mask_file_complete = Mask_file_complete
    
    ## initializations of GLM:
        
    # n_permutations
    
    n_perm = n_perm 
    # n_cores/n_jobs: (only works for nilearn GLM)
    n_core = n_core
    # Confounders list
    Confounders_names = Confounders_names
    # test variables list (this should come from the table/csv)
    test_variables_name = test_variables_name
    
    
    Test_var_GLM_results_dir = os.path.join(GLM_Stats_dir, test_variables_name)
    
    try:
            os.makedirs(Test_var_GLM_results_dir)
    except OSError:
        if not os.path.isdir(Test_var_GLM_results_dir):
            raise
#%% End of initializing     
    ###############################################
    # Flag: is 4D file already created?
    
    if merged_Flag == 0 and Flag_TFCE ==0: # means 4D image does not exist
        # privide the folder ofthe original images needed to be merged:
        Image_top_DIR = Image_top_DIR
        # mod_method
        mod_method = mod_method
        ##### The following code needs "Base_Sample_name" as an input to decide how to look for original images
        files_list, nifti_masker, D,gm_maps_masked = load_NIFTI_DATA_FROM_TABLE(Image_top_DIR, Sample_info_table, merged_sample_dir,\
                                                                                merged_image_name, Mask_file_complete, Base_Sample_name, \
                                                                                modulation_method = mod_method, load_nifti_masker_Flag=1)
    elif merged_Flag == 0 and Flag_TFCE ==1: # means 4D image does not exist
        # privide the folder ofthe original images needed to be merged:
        Image_top_DIR = Image_top_DIR
        # mod_method
        mod_method = mod_method
        ##### The following code needs "Base_Sample_name" as an input to decide how to look for original images
        files_list, nifti_masker = load_NIFTI_DATA_FROM_TABLE(Image_top_DIR, Sample_info_table, merged_sample_dir,\
                                                                                merged_image_name, Mask_file_complete, Base_Sample_name, \
                                                                                modulation_method = mod_method, load_nifti_masker_Flag=0)
    
    elif merged_Flag == 1 and Flag_TFCE ==0:
        nifti_masker = NiftiMasker(standardize=False, mask_img=Mask_file_complete)#, memory='nilearn_cache')
        gm_maps_masked = nifti_masker.fit_transform(os.path.join(merged_sample_dir, merged_image_name + '.nii.gz'))
    
    
    #################################
    # What is the type of correction desired? TFCE (FSL), FWE_cluster level (Nilearn) or we look at uncorrected results?
    
    variables_of_interest, confounding_vars = Create_GLM_columns(Sample_info_table, test_variables_name, Confounders_names)
        
    if Flag_TFCE == 1: # means stats with TFCE is desired
        Base_File_name_Stats = "fsl_stats"
        FSL_stats_dir = os.path.join(Test_var_GLM_results_dir , 'fsl')
        try:
            os.makedirs(FSL_stats_dir)
        except OSError:
            if not os.path.isdir(FSL_stats_dir):
                raise
    
        #Create
        
        design_text_full_path = Create_design_text(FSL_stats_dir, variables_of_interest, confounding_vars)
        randomise_design_path = os.path.join(os.path.dirname(design_text_full_path), 'design.mat')
        cmd_d = 'Text2Vest ' + design_text_full_path +  ' ' + randomise_design_path
        subprocess.call(cmd_d, shell=True)
        contrast_text_full_path = Create_contrast_text(FSL_stats_dir, variables_of_interest, confounding_vars)
        randomise_contrast_path = os.path.join(os.path.dirname(contrast_text_full_path), 'design.con')
        cmd_c = 'Text2Vest ' + contrast_text_full_path +  ' ' + randomise_contrast_path
        subprocess.call(cmd_c, shell=True)
        
        # Create design_matrix (for FSL TFCE)  based on Table csv and selected confounders and 
        input_randomise = os.path.join(merged_sample_dir, merged_image_name + '.nii.gz')
        mask_randomise = Mask_file_complete
        Output_randomise = os.path.join(FSL_stats_dir, test_variables_name + '_' + Base_File_name_Stats)
        
        t1_full_path = Output_randomise + '_tstat1.nii.gz'
        t2_full_path = Output_randomise + '_tstat2.nii.gz'
        p1_full_path = Output_randomise + '_tfce_corrp_tstat1.nii.gz'
        p2_full_path = Output_randomise + '_tfce_corrp_tstat2.nii.gz'
    
    
        # this is where we ACTUALLY call randomise
        cmd_randomise = "/bin/bash -c 'source /usr/share/fsl/5.0/etc/fslconf/fsl.sh; randomise -i " + input_randomise + " -m " + mask_randomise + " -o " + Output_randomise + " -d "+ randomise_design_path + " -t " + randomise_contrast_path + " -n " + str(n_perm) + " -T -D'"
        if SLURM_Que == True:
            GLM_Submision_script_full_file = os.path.join(Base_working_dir, "Slurm_submision_script")
            with open(Template_Submision_script_path) as f:
                lines = f.readlines()
                with open(GLM_Submision_script_full_file, "w") as f1:
                    f1.writelines(lines)
            log_dir_full_path = os.path.join(Base_working_dir , 'logs')
            try:
                os.makedirs(log_dir_full_path)
            except OSError:
                if not os.path.isdir(log_dir_full_path):
                    raise
            log_out_line = "#SBATCH --output=" + log_dir_full_path + "/mpi-out.%j"
            log_err_line = "#SBATCH --output=" + log_dir_full_path + "/mpi-err.%j"
            
            with open(GLM_Submision_script_full_file, "a") as f1:
                f1.writelines(log_out_line+"\n")
                f1.writelines(log_err_line+"\n")
                f1.writelines("#SBATCH --partition=large"+"\n")
                f1.writelines(cmd_randomise)  
             
            
            
            
            #Slurm_cmd = "sbatch " + GLM_Submision_script_full_file
            #subprocess.call(Slurm_cmd, shell=True)
        else:
            
            subprocess.call(cmd_randomise, shell=True)
        #print("To be completed")
        
    
    elif Flag_TFCE == 0:# means that I need nilearn stats
        
        # Here I do the nilearn GLM:
        
        Base_File_name_Stats = "nilearn_stats"
        Nilearn_stats_dir = os.path.join(Test_var_GLM_results_dir , 'nilearn')
        try:
            os.makedirs(Nilearn_stats_dir)
        except OSError:
            if not os.path.isdir(Nilearn_stats_dir):
                raise
    ############# So the plan is to have stats as: /base_dir/GLM_stats/Test_var/fsl or nilearn/here I have t1/2 and p1/2 maps (i.e. 4 maps)/
                                                                                                                               
        design_text_full_path = Create_design_text(Nilearn_stats_dir, variables_of_interest, confounding_vars)
        _ , T_file_name, P_file_name = permutation_based_stats_and_saving_p_vals(gm_maps_masked,nifti_masker, variables_of_interest,\
                                                                                 confounding_vars,n_perm, n_core, Nilearn_stats_dir, \
                                                                                 test_variables_name, Base_File_name_Stats)
        t1_full_path, t2_full_path = Separate_nilearn_Tstats_maps_to_sides(T_file_name, Nilearn_stats_dir, Base_File_name_Stats,\
                                                                           test_variables_name)
        p1_full_path, p2_full_path = Separate_nilearn_corrPstats_maps_to_sides(P_file_name, Nilearn_stats_dir, Base_File_name_Stats,\
                                                                               test_variables_name)
        
    
    return t1_full_path, p1_full_path, t2_full_path, p2_full_path, design_text_full_path

#%****Added 16.01.2019***** For cortical thickness glm with palm (at the moment only runs on brainb32a, as octave is only installed there)

def palm_one_var_multiplereg_on_Sample(Base_working_dir,Sample_info_table_CSV_full_path, Confounders_names, test_variables_name,\
                                       List_of_merged_surf_FullPath,List_of_surfaces,n_perm = 100, n_core = 1, Flag_TFCE = 1, TFCE_cmd = '-T -tfce2D -corrmod -logp',\
                                       SLURM_Que = False, Template_Submision_script_path = '', FSavg_SUBJECTS_DIR ='${SUBJECTS_DIR}/fsaverage'):


# I did not add all the options of palm here. So in my case, I am only intersted in TFCE on already merged images/Thicknesses. 
#palm -i lh.thickness.mgz -i rh.thickness.mgz -s lh.white -s rh.white -d design.mat -t design.con -o myresults -n 2000 -demean -T -tfce2D -corrmod -logp 

# List_of_merged_surf_FullPath: a list that is passed to option -i
#   e.g. ['lh.thickness.mgz, 'rh.thickness.mgz']
# List_of_surfaces: for the thickness, for each -i, we need to provide a surface and even area file that gives area at each vertex (for TFCE)
#   eg. ["lh.white lh.avg_area", "rh.white rh.avg_area"]
# Flag_TFCE = '-T -tfce2D -corrmod'

    ### Initialization:
    # Folders:
    # Base directory of analysis: 
    Base_working_dir = Base_working_dir
    # Where the 4D image file is located? or should be located(this also should be Fixed, depnding only to the base dir)
    
#    merged_sample_dir = os.path.join(Base_working_dir, '4D_images')
    
    # Where should the GLM stats be saved (this also should be Fixed, depnding only to the base dir)
    GLM_Stats_dir = os.path.join(Base_working_dir, 'GLM_Stats_dir')
    try:
            os.makedirs(GLM_Stats_dir)
    except OSError:
        if not os.path.isdir(GLM_Stats_dir):
            raise
    
    # name of Info table of variables of the sample it should be a csv (with Fix column names for sex, age and TIV) 

    
    ## initializations of GLM:
        
    # n_permutations
    Sample_info_table = pd.read_csv(Sample_info_table_CSV_full_path, delimiter=',')
    n_perm = n_perm 
    # n_cores/n_jobs: (only works for nilearn GLM)
    n_core = n_core
    # Confounders list
    Confounders_names = Confounders_names
    # test variables list (this should come from the table/csv)
    test_variables_name = test_variables_name
    
    
    Test_var_GLM_results_dir = os.path.join(GLM_Stats_dir, test_variables_name)
    
    try:
            os.makedirs(Test_var_GLM_results_dir)
    except OSError:
        if not os.path.isdir(Test_var_GLM_results_dir):
            raise
#%% End of initializing         
    variables_of_interest, confounding_vars = Create_GLM_columns(Sample_info_table, test_variables_name, Confounders_names)
        
    if Flag_TFCE == 1: # means stats with TFCE is desired
        Base_File_name_Stats = "palm_stats"
        PALM_stats_dir = os.path.join(Test_var_GLM_results_dir , 'palm')
        try:
            os.makedirs(PALM_stats_dir)
        except OSError:
            if not os.path.isdir(PALM_stats_dir):
                raise
    
        #Create
        
        design_text_full_path = Create_design_text(PALM_stats_dir, variables_of_interest, confounding_vars)
        palm_design_path = os.path.join(os.path.dirname(design_text_full_path), 'design.mat')
        cmd_d = 'Text2Vest ' + design_text_full_path +  ' ' + palm_design_path
        subprocess.call(cmd_d, shell=True)
        contrast_text_full_path = Create_contrast_text(PALM_stats_dir, variables_of_interest, confounding_vars)
        palm_contrast_path = os.path.join(os.path.dirname(contrast_text_full_path), 'design.con')
        cmd_c = 'Text2Vest ' + contrast_text_full_path +  ' ' + palm_contrast_path
        subprocess.call(cmd_c, shell=True)
        
        # Create design_matrix (for PALM TFCE)  based on Table csv and selected confounders and 
        input_palm = ''
        for f in List_of_merged_surf_FullPath:
            input_palm = input_palm + '-i ' + f + ' '
            
        surfaces_palm = ''
        
        for f in List_of_surfaces: 
            surfaces_palm = surfaces_palm + '-s ' + FSavg_SUBJECTS_DIR + '/surf/' + f + ' '
        
        Out_put_file_prefix = test_variables_name + '_' + Base_File_name_Stats
        Output_palm = os.path.join(PALM_stats_dir, Out_put_file_prefix)

        #palm -i lh.thickness.mgz -i rh.thickness.mgz -s lh.white -s rh.white -d design.mat -t design.con -o myresults -n 2000 -demean -T -tfce2D -corrmod -logp 

        # this is where we ACTUALLY call palm
        cmd_palm = "/bin/bash -c 'palm " + input_palm + surfaces_palm + " -o " + Output_palm + " -d "+ palm_design_path + " -t " + palm_contrast_path + " -n " + str(n_perm) + ' ' + TFCE_cmd + " -demean'"
        if SLURM_Que == True:
            GLM_Submision_script_full_file = os.path.join(Base_working_dir, "Slurm_submision_script")
            with open(Template_Submision_script_path) as f:
                lines = f.readlines()
                with open(GLM_Submision_script_full_file, "w") as f1:
                    f1.writelines(lines)
            log_dir_full_path = os.path.join(Base_working_dir , 'logs')
            try:
                os.makedirs(log_dir_full_path)
            except OSError:
                if not os.path.isdir(log_dir_full_path):
                    raise
            log_out_line = "#SBATCH --output=" + log_dir_full_path + "/mpi-out.%j"
            log_err_line = "#SBATCH --output=" + log_dir_full_path + "/mpi-err.%j"
            
            with open(GLM_Submision_script_full_file, "a") as f1:
                f1.writelines(log_out_line+"\n")
                f1.writelines(log_err_line+"\n")
                f1.writelines("#SBATCH --partition=large"+"\n")
                f1.writelines(cmd_palm)  
             
            
            
            
            #Slurm_cmd = "sbatch " + GLM_Submision_script_full_file
            #subprocess.call(Slurm_cmd, shell=True)
        else:
            
            subprocess.call(cmd_palm, shell=True)
        #print("To be completed")
        
    
    elif Flag_TFCE == 0:# this should be easy to also implement, yet I would need to read and implement the options in detail 
        
        print('not implemented yet, modify the palm_one_var_multiplereg_on_Sample function')
    P_val_path_dict = {'Pmap_Pos':{}, 'Pmap_Neg':{}}
    for n_mod in np.arange(len(List_of_merged_surf_FullPath)): # numner of modalities
        P_val_path_dict['Pmap_Pos']['m' + str(n_mod +1)] = Output_palm + '_tfce_tstat_mfwep_m' + str(n_mod +1)+'_c1.mgz'
        P_val_path_dict['Pmap_Neg']['m' + str(n_mod +1)] = Output_palm + '_tfce_tstat_mfwep_m' + str(n_mod +1)+'_c2.mgz'    

        
    
    return P_val_path_dict, design_text_full_path

    
#%% Monday 17.07.2018

#***************************************************************
#       Help functions for "Create_binned_ROIS"
#***************************************************************
def Calculate_Cohens_d_effect_size_from_T(DF, T_value):
    d = np.divide(2*T_value,np.sqrt(DF))
    return d
def Calculate_r_effect_size_from_T(DF, T_value):
    r = np.sqrt(np.divide(np.power(T_value, 2),(np.power(T_value,2) + DF)))
    return r


def Calculate_multiple_regression_DF(n_sample = [], n_covariates = [], design_text_full_path = ''):
    '''
    Calculates "Degrees of Freedom (DF)" based on either "n_sample and n_covariates" or Design.txt file
    '''
    
    if n_sample and n_covariates:
        DF = n_sample-n_covariates-1
    else:
        design = np.loadtxt(design_text_full_path)
        n_sample = np.shape(design)[0]
        n_covariates = np.shape(design)[1]
        DF = n_sample-n_covariates-1
    return DF

def Calculate_uncorrected_t_threshold_from_p(DF,voxel_p_thresh = 0.005, test_side = 2):
    
    ''' 
    Calculates t-threshold from degrees of Freedom and based on P-threshold and number of sides tested
    '''
    
    t_threshold = stats.t.ppf(1-voxel_p_thresh/test_side, DF)
    
        
    return t_threshold 
def Create_thresholded_tstats(grot_tfce_corrp_tstat1, grot_tstat1, Cluster_defining_threshold, ROI_saving_DIR):
    ''' This is a wrapper funticon for FSLMaths, making stats inage ready for Cluster command
        ofcourse in case we are intersted in un-corrected stats, we will have same image as grot_tfce_corrp_tstat1 and grot_tstat1
    '''
    
    
    
    os.chdir(ROI_saving_DIR)
    thr_maths = Threshold()
    thr_maths.inputs.in_file = grot_tfce_corrp_tstat1
    thr_maths.inputs.thresh = Cluster_defining_threshold
    thr_maths.inputs.args = '-bin -mul ' + grot_tstat1
    thr_maths.inputs.direction = 'below'
    thr_maths.inputs.out_file = "Thresholded_"+ os.path.basename(grot_tfce_corrp_tstat1)
    
    thr_maths.run()
    return os.path.join(ROI_saving_DIR , thr_maths.inputs.out_file)  
    
def identify_clusters(Thresholded_Stats_file_full_path, ROI_saving_DIR, T_Threshold = 0.000001, min_extent = 100):
    '''
    Inputs:
        Stats_file_full_path: Full Path of T-map or corrected p-map.
        ROI_saving_DIR: Full Path of the directory where ROIs need to be saved in. 
        Threshold: threshold on t-values or p-values for which clusters are going to be created. 
        min_extent: cluster extent threshold (minimum number of voxels in a cluster)
    Outputs:
        Index volume (file full path)
        Path of the Text file of the results' table
    
    '''
    os.chdir(ROI_saving_DIR)
    DD=Cluster()
    DD.inputs.threshold=T_Threshold
    DD.inputs.in_file= Thresholded_Stats_file_full_path
    #DD.inputs.out_localmax_txt_file = os.path.join(ROI_saving_DIR,'stats.txt')
    DD.inputs.out_index_file = os.path.join(ROI_saving_DIR,'index_'+os.path.basename(Thresholded_Stats_file_full_path))
    DD.inputs.num_maxima =1 # This means in each cluster I am interested in only one local maxima, to generate the table.
    DD.inputs.args="--minextent="+ str(min_extent)
    DD.inputs.terminal_output = 'file'   # This will copy the table to the stdout.nipype (so one can also have a look at the cope value for corrected stats)
    DD.run()
    return DD.inputs.out_index_file, os.path.join(ROI_saving_DIR,'stdout.nipype')  

def Separate_clusters_in_order(Stats_map_full_path, index_vol_full_path, new_cluster_list, Order_type, Order_index, cluster_value, min_extent, ROI_save_dir):
    '''
    
    
    '''
    
    BASE_ROI_file_names = os.path.splitext(os.path.splitext(os.path.basename(Stats_map_full_path))[0])[0]
    Masker_obj = NiftiMasker()
    Indexed_image_arr = Masker_obj.fit_transform(index_vol_full_path)
    Cluster_ar = np.empty(Indexed_image_arr.shape)
    Cluster_ar = np.logical_and(Indexed_image_arr >= (cluster_value-0.001) , Indexed_image_arr <= (cluster_value + 0.001))
    where_to_save_bin_ROIs = os.path.dirname(index_vol_full_path)
    new_cluster_list.append(os.path.join(where_to_save_bin_ROIs, BASE_ROI_file_names + 'binned_ROI_' +str(min_extent)+ '_volxels'+ Order_type + '_' + str(Order_index) + '.nii.gz'))
    print(new_cluster_list[Order_index-1])
    Masker_obj.inverse_transform(Cluster_ar).to_filename(new_cluster_list[Order_index-1])       
    return new_cluster_list 
 


#******************************************************************************************************************************
#                                   End of help functions for "Create_binned_ROIS"
#******************************************************************************************************************************


def Create_binned_ROIS(Stats_map_full_path, design_text_full_path, \
                       Stats_type = 'fwe_corrp', thresh_corrp = 0.05, T_Threshold = 0.000001, min_cluster_size_corr = 10, \
                       thresh_uncorrp =  0.005, min_cluster_size_uncorr = 100, test_side = 2, \
                       Order_ROIs = 'extent', Limit_num_ROIS = np.inf, binned_ROIS = True):
    
    """
   
    Stats_type = 'fwe_corrp' or 'tfce_corrp' or 'uncorr'
    Order_ROIs = 'height'or 'extent'. Default: 'extent'
    Limit_num_ROIS = np.inf or a value. 
    
    Input: outputs of GLM_on_Sample Orderd in the following folder structure:
    /Shahrzad'sPersonalFolder/HCP/Partial_cors
                                               /sample_CSV/Subjects_info_table.csv      # Info_table_CSV
                                               /4D_images/merged_image_name.nii.gz      # merged 4D images of the subsamples
                                               /GLM_Stats_dir/                          # Stats of GLM which are going to be used to create ROI
                                                             %test_var_name/
                                                                            
                                                                           fsl/
                                                                               
                                                                               %test_var_name_fsl_stats_tfce_corrp_tstat1/2.nii.gz  # Fixed Outputs saved   
                                                                               %test_var_name_fsl_stats_tstat1/2.nii.gz     # Fixed Outputs saved
                                                                               ROI/
                                                                                   tfce_corrp/
                                                                        ***************************OR*****************
                                                                           nilearn/
                                                                                   %test_var_name_nilearn_stats_FWE_corrp_tstat1/2.nii.gz  # Fixed Outputs saved  
                                                                                   %test_var_name_nilearn_stats_tstat1/2.nii.gz  # Fixed Outputs saved   
                                                                                   ROI/
                                                                                       fwe_corrp/
                                                                                       uncorr/
                                                                                             tstat1/
                                                                                                   index_%test_var_name_nilearn_stats_tstat1.nii.gz
                                                                                                   stats.txt
                                                                                                   list_of_ROIs_based_on_height/or/extent.txt
                                                                                                   binned_ROI_height/or/extent_%d.nii.gz 
                                                                                                   
                                                                                                   
                                                                                             tstat2/
                                                                                       
    
    TO DO: 
        binned_ROIS = True  /can also be False (so the following functions need to be modified:
                                                identify_clusters: To save also maximum/or maybe the thresholded volume as well.
                                                Separate_clusters_in_order: Multiplying the binned cluster_ar by the additional save dvolume 
                                                                            (i.e. thresholded or the maximum voxel volume)
    
    """
    empty_flag = 1
    ROI_base_dir = os.path.dirname(Stats_map_full_path)
    Stats_base_name = os.path.basename(Stats_map_full_path)
    Stats_side = Stats_base_name[-13:-7]    # Discarding .nii.gz and begining of the file, keeping only "tstat1" and "tstat2"
    ROI_dir = os.path.join(ROI_base_dir, 'ROIs' , Stats_type, Stats_side, '_'+str(min_cluster_size_corr)+ '_volxels')
    ROI_list_full_path = os.path.join(ROI_dir, 'list_of_ROIs_based_on_' + Order_ROIs + '.txt')
    try:
        os.makedirs(ROI_dir)
    except OSError:
        if not os.path.isdir(ROI_dir):
            raise
    
            
    ###End of initializing##
    
    if Stats_type == 'tfce_corrp':
        
        Cluster_defining_threshold =1-thresh_corrp
        T_file_name = Stats_base_name.split(Stats_type +'_')[0] + Stats_base_name.split(Stats_type + '_')[1]
        T_map_full_path = os.path.join(os.path.dirname(Stats_map_full_path),T_file_name)
        
        Thresholded_Stats_map_full_path = Create_thresholded_tstats(Stats_map_full_path, T_map_full_path, Cluster_defining_threshold, ROI_dir)
        index_vol_full_path, Local_max_Table_full_path = identify_clusters(Thresholded_Stats_map_full_path, ROI_dir, T_Threshold = T_Threshold,\
                                                                           min_extent = min_cluster_size_corr)
        Results_Table = pd.read_csv(Local_max_Table_full_path, delimiter='\t')
        if Order_ROIs == 'height':
            Results_Table.sort_values(by=["MAX", "Cluster Index"], ascending=False, inplace=True)
            Results_Table.reset_index(inplace=True, drop = True)
        
        if np.isinf(Limit_num_ROIS):
            new_cluster_list = []
            for Order_index in np.arange(Results_Table.shape[0]):
                
                cluster_value = Results_Table.loc[Order_index]["Cluster Index"]
                new_cluster_list = Separate_clusters_in_order(Stats_map_full_path, index_vol_full_path, new_cluster_list, Order_type = Order_ROIs, Order_index = Order_index + 1, \
                                                              cluster_value = cluster_value, min_extent = min_cluster_size_corr, ROI_save_dir= ROI_dir)
                    
        else:
            new_cluster_list = []
            for Order_index in np.arange(min(Results_Table.shape[0], Limit_num_ROIS)):
                
                cluster_value = Results_Table.loc[Order_index]["Cluster Index"]
                new_cluster_list = Separate_clusters_in_order(Stats_map_full_path, index_vol_full_path, new_cluster_list, Order_type = Order_ROIs, Order_index = Order_index + 1, \
                                                              cluster_value = cluster_value,min_extent = min_cluster_size_corr, ROI_save_dir= ROI_dir)
        DF = Calculate_multiple_regression_DF(design_text_full_path = design_text_full_path)
        T_max_df = pd.DataFrame()
        Cohen_d_df = pd.DataFrame()
        r_efect_size_df = pd.DataFrame()
        for count in np.arange(len(new_cluster_list)):
            temp_df = pd.DataFrame(data = Results_Table.loc[count]["MAX"], columns= [os.path.splitext(os.path.splitext(os.path.basename(new_cluster_list[count]))[0])[0]] , index = ['Tmax'])
            T_max_df = pd.concat([T_max_df, temp_df], axis = 1)
            temp_cohen_d = Calculate_Cohens_d_effect_size_from_T(DF, T_value=Results_Table.loc[count]["MAX"])
            temp_cohen_d_df = pd.DataFrame(data = temp_cohen_d, columns= [os.path.splitext(os.path.splitext(os.path.basename(new_cluster_list[count]))[0])[0]] , index = ['cohen_d'])
            Cohen_d_df = pd.concat([Cohen_d_df, temp_cohen_d_df], axis = 1)
            temp_r_effect = Calculate_r_effect_size_from_T(DF,T_value=Results_Table.loc[count]["MAX"])
            temp_r_effect_df = pd.DataFrame(data = temp_r_effect, columns= [os.path.splitext(os.path.splitext(os.path.basename(new_cluster_list[count]))[0])[0]] , index = ['r_effect_size'])
            r_efect_size_df = pd.concat([r_efect_size_df, temp_r_effect_df], axis = 1)
            
            
            
        T_max_df_full_path = os.path.join(ROI_dir, 'T_max_of_binned_ROIS.pkl')
        T_max_df.to_pickle(T_max_df_full_path)
        Cohen_d_df_full_path = os.path.join(ROI_dir, 'Cohen_d_of_binned_ROIS.pkl')
        Cohen_d_df.to_pickle(Cohen_d_df_full_path)
        r_efect_size_df_full_path = os.path.join(ROI_dir, 'r_efect_size_of_binned_ROIS.pkl')
        r_efect_size_df.to_pickle(r_efect_size_df_full_path)
        
        
        if len(new_cluster_list)>0:
             empty_flag = 0
    
                
        np.save(os.path.join(os.path.dirname(Stats_map_full_path), 'empty_flag.npy'),empty_flag)
        f = open(ROI_list_full_path, 'w')
        for item in new_cluster_list:
            f.write("%s\n" % item)
        f.close()
        
        
        
        
    elif Stats_type == 'FWE_corrp':
        
        Cluster_defining_threshold = -np.log10(thresh_corrp)
        T_file_name = Stats_base_name.split(Stats_type +'_')[0] + Stats_base_name.split(Stats_type + '_')[1]
        T_map_full_path = os.path.join(os.path.dirname(Stats_map_full_path),T_file_name)
        
        Thresholded_Stats_map_full_path = Create_thresholded_tstats(Stats_map_full_path, T_map_full_path, Cluster_defining_threshold, ROI_dir)
        index_vol_full_path, Local_max_Table_full_path = identify_clusters(Thresholded_Stats_map_full_path, ROI_dir, T_Threshold = T_Threshold,\
                                                                           min_extent = min_cluster_size_corr)
        Results_Table = pd.read_csv(Local_max_Table_full_path, delimiter='\t')
        
        if Order_ROIs == 'height':
            ''' Here I need to calculate the t-value @ peak voxel and order the results table based on maximum T-value.
            '''
            #T_map = Stats_base_name.split(Stats_type +'_')[0] + Stats_base_name.split(Stats_type + '_')[1]
            #T_map_nb = nb.load(os.path.join(os.path.dirname(Stats_map_full_path),T_map)).get_data()
            #for i in np.arange(len(Results_Table)):
            #    Results_Table.set_value(i, 'MAX T_val',T_map_nb[int(Results_Table.loc[i]['MAX X (vox)']),\
            #                                                    int(Results_Table.loc[i]['MAX Y (vox)']),\
            #                                                    int(Results_Table.loc[i]['MAX Z (vox)'])])
            Results_Table.sort_values(by=["MAX", "Cluster Index"], ascending=False, inplace=True)
            Results_Table.reset_index(inplace=True, drop = True)
        
        if np.isinf(Limit_num_ROIS):
            new_cluster_list = []
            for Order_index in np.arange(Results_Table.shape[0]):
                
                cluster_value = Results_Table.loc[Order_index]["Cluster Index"]
                new_cluster_list = Separate_clusters_in_order(Stats_map_full_path, index_vol_full_path, new_cluster_list, Order_type = Order_ROIs, Order_index = Order_index + 1, \
                                                              cluster_value = cluster_value, min_extent = min_cluster_size_corr, ROI_save_dir= ROI_dir)
                
        else:
            new_cluster_list = []
            for Order_index in np.arange(min(Results_Table.shape[0], Limit_num_ROIS)):
                
                cluster_value = Results_Table.loc[Order_index]["Cluster Index"]
                new_cluster_list = Separate_clusters_in_order(Stats_map_full_path, index_vol_full_path, new_cluster_list, Order_type = Order_ROIs, Order_index = Order_index + 1, \
                                                              cluster_value = cluster_value, min_extent = min_cluster_size_corr, ROI_save_dir= ROI_dir)
        DF = Calculate_multiple_regression_DF(design_text_full_path = design_text_full_path)
        T_max_df = pd.DataFrame()
        Cohen_d_df = pd.DataFrame()
        r_efect_size_df = pd.DataFrame() 
        for count in np.arange(len(new_cluster_list)):
            temp_df = pd.DataFrame(data = Results_Table.loc[count]["MAX"], columns= [os.path.splitext(os.path.splitext(os.path.basename(new_cluster_list[count]))[0])[0]] , index = ['Tmax'])
            T_max_df = pd.concat([T_max_df, temp_df], axis = 1)
            temp_cohen_d = Calculate_Cohens_d_effect_size_from_T(DF, T_value=Results_Table.loc[count]["MAX"])
            temp_cohen_d_df = pd.DataFrame(data = temp_cohen_d, columns= [os.path.splitext(os.path.splitext(os.path.basename(new_cluster_list[count]))[0])[0]] , index = ['cohen_d'])
            Cohen_d_df = pd.concat([Cohen_d_df, temp_cohen_d_df], axis = 1)
            temp_r_effect = Calculate_r_effect_size_from_T(DF,T_value=Results_Table.loc[count]["MAX"])
            temp_r_effect_df = pd.DataFrame(data = temp_r_effect, columns= [os.path.splitext(os.path.splitext(os.path.basename(new_cluster_list[count]))[0])[0]] , index = ['r_effect_size'])
            r_efect_size_df = pd.concat([r_efect_size_df, temp_r_effect_df], axis = 1)
            
        T_max_df_full_path = os.path.join(ROI_dir, 'T_max_of_binned_ROIS.pkl')
        T_max_df.to_pickle(T_max_df_full_path) 
        Cohen_d_df_full_path = os.path.join(ROI_dir, 'Cohen_d_of_binned_ROIS.pkl')
        Cohen_d_df.to_pickle(Cohen_d_df_full_path)
        r_efect_size_df_full_path = os.path.join(ROI_dir, 'r_efect_size_of_binned_ROIS.pkl')
        r_efect_size_df.to_pickle(r_efect_size_df_full_path)
        
        if len(new_cluster_list)>0:
             empty_flag = 0
        
                
        np.save(os.path.join(os.path.dirname(Stats_map_full_path), 'empty_flag.npy'),empty_flag)
        f = open(ROI_list_full_path, 'w')
        for item in new_cluster_list:
            f.write("%s\n" % item)
        f.close()
        
        
        
        
        
    elif  Stats_type == 'uncorr':
        
        DF = Calculate_multiple_regression_DF(design_text_full_path = design_text_full_path)
        uncorr_t_treshold = Calculate_uncorrected_t_threshold_from_p(DF,voxel_p_thresh = thresh_uncorrp, test_side =test_side)
        index_vol_full_path, Local_max_Table_full_path = identify_clusters(Stats_map_full_path, ROI_dir, Cluster_defining_threshold = uncorr_t_treshold,\
                                                                           min_extent = min_cluster_size_uncorr)
        Results_Table = pd.read_csv(Local_max_Table_full_path, delimiter='\t')
        if Order_ROIs == 'height':
            Results_Table.sort_values(by=["MAX", "Cluster Index"], ascending=False, inplace=True)
            Results_Table.reset_index(inplace=True, drop = True)
        
        if np.isinf(Limit_num_ROIS):
            new_cluster_list = []
            for Order_index in np.arange(Results_Table.shape[0]):
                cluster_value = Results_Table.loc[Order_index]["Cluster Index"]
                new_cluster_list = Separate_clusters_in_order(Stats_map_full_path, index_vol_full_path, new_cluster_list,\
                                                              Order_type = Order_ROIs, Order_index = Order_index + 1, \
                                                              cluster_value = cluster_value,min_extent = min_cluster_size_uncorr, ROI_save_dir= ROI_dir)
            
        else:
            new_cluster_list = []
            for Order_index in np.arange(min(Results_Table.shape[0], Limit_num_ROIS)):
                
                cluster_value = Results_Table.loc[Order_index]["Cluster Index"]
                new_cluster_list = Separate_clusters_in_order(Stats_map_full_path, index_vol_full_path, new_cluster_list,\
                                                              Order_type = Order_ROIs, Order_index = Order_index + 1, \
                                                              cluster_value = cluster_value, min_extent = min_cluster_size_uncorr, ROI_save_dir= ROI_dir)
        DF = Calculate_multiple_regression_DF(design_text_full_path = design_text_full_path)
        T_max_df = pd.DataFrame()
        Cohen_d_df = pd.DataFrame()
        r_efect_size_df = pd.DataFrame()  
        for count in np.arange(len(new_cluster_list)):
            temp_df = pd.DataFrame(data = Results_Table.loc[count]["MAX"], columns= [os.path.splitext(os.path.splitext(os.path.basename(new_cluster_list[count]))[0])[0]] , index = ['Tmax'])
            T_max_df = pd.concat([T_max_df, temp_df], axis = 1)
            temp_cohen_d = Calculate_Cohens_d_effect_size_from_T(DF, T_value=Results_Table.loc[count]["MAX"])
            temp_cohen_d_df = pd.DataFrame(data = temp_cohen_d, columns= [os.path.splitext(os.path.splitext(os.path.basename(new_cluster_list[count]))[0])[0]] , index = ['cohen_d'])
            Cohen_d_df = pd.concat([Cohen_d_df, temp_cohen_d_df], axis = 1)
            temp_r_effect = Calculate_r_effect_size_from_T(DF,T_value=Results_Table.loc[count]["MAX"])
            temp_r_effect_df = pd.DataFrame(data = temp_r_effect, columns= [os.path.splitext(os.path.splitext(os.path.basename(new_cluster_list[count]))[0])[0]] , index = ['r_effect_size'])
            r_efect_size_df = pd.concat([r_efect_size_df, temp_r_effect_df], axis = 1)
            
            
        T_max_df_full_path = os.path.join(ROI_dir, 'T_max_of_binned_ROIS.pkl')
        T_max_df.to_pickle(T_max_df_full_path) 
        Cohen_d_df_full_path = os.path.join(ROI_dir, 'Cohen_d_of_binned_ROIS.pkl')
        Cohen_d_df.to_pickle(Cohen_d_df_full_path)
        r_efect_size_df_full_path = os.path.join(ROI_dir, 'r_efect_size_of_binned_ROIS.pkl')
        r_efect_size_df.to_pickle(r_efect_size_df_full_path)
        
        if len(new_cluster_list)>0:
             empty_flag = 0
        
                
        np.save(os.path.join(os.path.dirname(Stats_map_full_path), 'empty_flag.npy'),empty_flag)
        
        f = open(ROI_list_full_path, 'w')
        for item in new_cluster_list:
            f.write("%s\n" % item)
        f.close()
        
            
    return ROI_list_full_path


#%% Create separate clusters from surface data
# A: to generate the overlap for the exploratory findings. mri_binarize --i Results_Left_clustere_tstat_fwep.mgz --min 1 --o p_bin.mgz
# B_: to extract mean CT in each ROI: 
#mri_surfcluster --in Taste_Unadj_palm_stats_tfce_tstat_fwep_m2_c1.mgz --thmin 0.9 --hemi rh --no-adjust --subject fsaverage --sum summaryfile.txt --o ./outputid.mgz --olab ./outlabelbase
    
# This saves each sig. ROI in a separate label file. 
# Now 
#A
    
class FS_stats_masking(object):
    def __init__(self, result_saving_Dir, Stats_map_full_path, hemi):
        self.Saving_dir = result_saving_Dir
        self.INPUT_stats_image = Stats_map_full_path
        self.hemi = hemi        
        
        try:
            os.makedirs(self.Saving_dir)
        except OSError:
            if not os.path.isdir(self.Saving_dir):
                raise

    def mri_binarize(self, Binning_threshold, output_name = ''):
        if len(output_name) == 0:
            output_name = os.path.basename(self.INPUT_stats_image)
            Output_path = os.path.join(self.Saving_dir, self.hemi + '_' +str(Binning_threshold) +\
                                       '_bin_' + os.path.splitext(output_name)[0]+ '.mgz')
            cmd_b = "/bin/bash -c 'source /nfsusr/local/freesurfer6/FreeSurferEnv.sh; mri_binarize --i " + \
            self.INPUT_stats_image + " --min " + str(Binning_threshold) + " --o " + Output_path + "'"
            
        else:
            Output_path = os.path.join(self.Saving_dir, self.hemi + '_' + str(Binning_threshold) + \
                                       '_bin_' + os.path.splitext(output_name)[0]+ '.mgz')
            cmd_b = "/bin/bash -c 'source /nfsusr/local/freesurfer6/FreeSurferEnv.sh; mri_binarize --i " + \
            self.INPUT_stats_image + " --min " + str(Binning_threshold) + " --o " + Output_path + "'"
        subprocess.call(cmd_b, shell=True)
        
        return Output_path
        

    def mrisurfclusterLabel(self, Whichfsaverage = 'fsaverage', Cluster_defining_threshold = 0.01, mask = '', output_base_name = '', additioanl_prefix =''):
        if len(output_base_name) == 0:
            output_base_name = os.path.basename(self.INPUT_stats_image)
            output_base_name_path = os.path.join(self.Saving_dir, self.hemi + '_' + additioanl_prefix +'_' + str(Cluster_defining_threshold) + os.path.splitext(output_base_name)[0])
            summary_file_full_path = os.path.join(self.Saving_dir, self.hemi + '_' + additioanl_prefix +'_' + str(Cluster_defining_threshold) + os.path.splitext(output_base_name)[0] + '_summary.txt')
            
            if len(mask) == 0:
                
                cmd_c = "/bin/bash -c 'source /nfsusr/local/freesurfer6/FreeSurferEnv.sh; mri_surfcluster --in " + self.INPUT_stats_image + \
                " --thmin " + str(Cluster_defining_threshold) + " --hemi " + self.hemi + " --no-adjust --subject " + Whichfsaverage + " --sum " + summary_file_full_path + " --o " +  output_base_name_path + ".mgz" + " --olab " + output_base_name_path + "'"
            else:
                cmd_c = "/bin/bash -c 'source /nfsusr/local/freesurfer6/FreeSurferEnv.sh; mri_surfcluster --in " + self.INPUT_stats_image + \
                " --thmin " + str(Cluster_defining_threshold) + " --hemi " + self.hemi + " --mask " + mask + " --no-adjust --subject " + Whichfsaverage + " --sum " + summary_file_full_path + " --o " +  output_base_name_path + ".mgz" + " --olab " + output_base_name_path + "'"
        else:
            output_base_name_path = os.path.join(self.Saving_dir, self.hemi + '_' + additioanl_prefix +'_' + str(Cluster_defining_threshold) + os.path.splitext(output_base_name)[0])
            summary_file_full_path = os.path.join(self.Saving_dir, self.hemi + '_' + additioanl_prefix +'_' +  str(Cluster_defining_threshold) + os.path.splitext(output_base_name)[0] + '_summary.txt')
            
            if len(mask) == 0:
                cmd_c = "/bin/bash -c 'source /nfsusr/local/freesurfer6/FreeSurferEnv.sh; mri_surfcluster --in " + self.INPUT_stats_image + \
                " --thmin " + str(Cluster_defining_threshold) + " --hemi " + self.hemi + " --no-adjust --subject " + Whichfsaverage + " --sum " + summary_file_full_path + " --o " +  output_base_name_path + ".mgz" + " --olab " + output_base_name_path + "'"
            else:
                cmd_c = "/bin/bash -c 'source /nfsusr/local/freesurfer6/FreeSurferEnv.sh; mri_surfcluster --in " + self.INPUT_stats_image + \
                " --thmin " + str(Cluster_defining_threshold) + " --hemi " + self.hemi + " --mask " + mask + " --no-adjust --subject " + Whichfsaverage + " --sum " + summary_file_full_path + " --o " +  output_base_name_path + ".mgz" + " --olab " + output_base_name_path + "'"

        subprocess.call(cmd_c, shell=True)
         
        Output = {'label_base_name' : os.path.basename(output_base_name_path), 'summary_basename':os.path.basename(summary_file_full_path)}
    
    
        return Output
    
def read_FS_summary_txt2DF(summary_file_path):
    Results_Table = pd.DataFrame(columns=['ClusterNo', 'MAX', 'NVtxs'])

    try :
        with open(summary_file_path,'r') as fh:
         for curline in fh:
             # check if the current line
             # starts with "#"
             if curline.startswith("#"):
                pass
             else:
                 line = curline.split()
                 Results_Table = Results_Table.append({'ClusterNo': int(line[0]), 'MAX': np.double(line[1]),'NVtxs':np.double(line[-1])}, ignore_index=True)
    except:
        pass
    return Results_Table
                 
    
        


def Create_binned_ROIS_on_surface(Stats_map_full_path, hemi, Whichfsaverage ='fsaverage', \
                                  Stats_type = 'tfce_corrp', thresh_corrp = 0.05, T_Threshold = 0.000001, test_side = 2):
    
    # This code is using theordering of the mri_surfcluster, which the largest ROI is at the beggining (lable = 0001) and smallest ROI is the smallest 
    
    empty_flag = 1
    ROI_base_dir = os.path.dirname(Stats_map_full_path)
    Stats_base_name = os.path.basename(Stats_map_full_path)
    Stats_side = Stats_base_name[-6:-4]    # Discarding .nii.gz and begining of the file, keeping only "tstat1" and "tstat2"
    Stats_modality = Stats_base_name[-9:-7] #hemi? 
    ROI_dir = os.path.join(ROI_base_dir, 'ROIs' , Stats_type, Stats_modality, Stats_side)
    ROI_list_full_path = os.path.join(ROI_dir, 'list_of_ROIs_based_on_' + hemi + '_' + Stats_side +'.txt')
    try:
        os.makedirs(ROI_dir)
    except OSError:
        if not os.path.isdir(ROI_dir):
            raise
    ###End of initializing##
    if Stats_type == 'tfce_corrp':
        # In palm I have asked the p-maps to be -log10, So: 
        Cluster_defining_threshold = -1* np.log10(thresh_corrp/test_side)
        
        #A: Create a surfmap of binned sig clusters
        surf_binning = FS_stats_masking(result_saving_Dir= ROI_dir, Stats_map_full_path= Stats_map_full_path,hemi= hemi)
        binned_surf_path = surf_binning.mri_binarize(Binning_threshold=Cluster_defining_threshold)
        
        T_file_name = os.path.basename(Stats_map_full_path).split('tfce_')[0] + 'dpv_tstat_' + Stats_modality + '_' + Stats_side + '.mgz'
        T_map_full_path = os.path.join(os.path.dirname(Stats_map_full_path),T_file_name)
        surf_clustering = FS_stats_masking(result_saving_Dir= ROI_dir, Stats_map_full_path= T_map_full_path,hemi= hemi)
        Clustering_Output = surf_clustering.mrisurfclusterLabel(Whichfsaverage, Cluster_defining_threshold=T_Threshold, mask = binned_surf_path)#Taste_Unadj_palm_stats_dpv_tstat_m1_c1.mgz 
        
        Results_Table = read_FS_summary_txt2DF(os.path.join(ROI_dir, Clustering_Output['summary_basename']))
        if len(Results_Table) == 0:
            empty_flag = 0
            ROI_list_full_path = ''
            
        else:
        
        
            f = open(ROI_list_full_path, 'w')
            for item in np.arange(len(Results_Table)):
                K = str(int(Results_Table['ClusterNo'][item])).zfill(4)
                f.write(os.path.join(ROI_dir, Clustering_Output['label_base_name'] + '-' + K + '.label\n'))
            f.close()
        
        np.save(os.path.join(os.path.dirname(ROI_dir), 'empty_flag.npy'),empty_flag)
        
    
            
        
    return ROI_list_full_path, empty_flag





#%%
    
# Function to create mean THICKNESS of a label file on a 4D surface in fsaverage.
def _label_mean_CT_4D_extraction(list_of_label_files_on_fsaverage, fourD_thickness_on_fsaverage, hemi, Tnew, stats_arg, x):
    ### Be aware that the Label and the fourD_thickness files should be on the same space, i.e. fsaverage
    ROI_label = list_of_label_files_on_fsaverage[x]
    Label_indices = surface.load_surf_data(ROI_label)
    fourDsurf = surface.load_surf_data(fourD_thickness_on_fsaverage)
    if stats_arg == 'Median':
        Tnew = np.median(fourDsurf[Label_indices], axis=0)
    elif stats_arg =='mean':
        Tnew = np.mean(fourDsurf[Label_indices], axis=0)
    return Tnew         



def CT_extraction(Sig_label_list_full_path,fourD_thickness_on_fsaverage, hemi, Sample_info_table_CSV_full_path,where_to_save_ROI_Table,\
                   ROI_table_name_suffix, what_to_extract='Median', n_jobs =20): 
    with open(Sig_label_list_full_path) as f:
        lines = f.read().splitlines()
        
    
    Tnew= [[] for i in np.arange(len(lines))] #### Here maybe a resampling needed to bring ROI to the space of T1 template?!
    Collected_parallel_Tnews = Parallel(n_jobs= n_jobs)(delayed(_label_mean_CT_4D_extraction)(lines, fourD_thickness_on_fsaverage, hemi, Tnew[x], what_to_extract, x) for x in np.arange(len(lines)))
    print(len(Collected_parallel_Tnews))
    print('lines\n')
    print(len(lines))
    print(lines)
    ###Here I want to extarct some part of the list path to name ROI-columns accordingly. 
    
    Base_name_for_ROIs = [os.path.splitext(os.path.basename(i))[0].split('.label')[0] for i in lines]
    ## Or : from nipype.utils.filemanip import split_filename
    print(Base_name_for_ROIs)
    # Here we save these simple ROI names in a file. This file is used to refer to the columns of the table, when one wants to correlate the extrcated GMV with 
    try:
        os.makedirs(where_to_save_ROI_Table)
    except OSError:
        if not os.path.isdir(where_to_save_ROI_Table):
            raise
    ROI_base_names_text_file_full_path = os.path.join(where_to_save_ROI_Table,ROI_table_name_suffix + '.txt')
    ROI_base_names_text_file = open(ROI_base_names_text_file_full_path, 'a')
    
    for item in Base_name_for_ROIs:
        ROI_base_names_text_file.write("%s\n" % item)
    ROI_base_names_text_file.close()
    
    ROI_tables = pd.DataFrame()
    Sample_table_with_CT_info_full_path = os.path.join(where_to_save_ROI_Table,\
                                                        os.path.splitext(os.path.basename(Sample_info_table_CSV_full_path))[0] + \
                                                                        '_' + what_to_extract + '_CT_'+ ROI_table_name_suffix + '.csv')
        
    try:
        Sample_info_table = pd.read_csv(Sample_table_with_CT_info_full_path)
        
    except:
        Sample_info_table = pd.read_csv(Sample_info_table_CSV_full_path, delimiter=',')
        
    if len(lines)>0:
        ROI_tables = pd.DataFrame(np.column_stack(Collected_parallel_Tnews), columns = Base_name_for_ROIs)
        New_Table_with_stats_Test = pd.concat([Sample_info_table, ROI_tables], axis = 1)
        New_Table_with_stats_Test.to_csv(Sample_table_with_CT_info_full_path, index = False)
        
    
    return Sample_table_with_CT_info_full_path, ROI_base_names_text_file_full_path




#%% Wednesday 19.07.2018
#***************************************************************************************************
#           Function to create resampled ROI from a 
#           list of ROIs and save the resampeld ROI
#**************************************************************************************************
    
def geting_arbitrary_ROIS_ready(Original_ROIs_list_full_path, fourD_Full_path, where_to_save_resampled_ROI, MASK_file_FULL_PATH = ''):
    
    try:
            os.makedirs(where_to_save_resampled_ROI)
    except OSError:
        if not os.path.isdir(where_to_save_resampled_ROI):
            raise
    
    with open(Original_ROIs_list_full_path) as f:
        lines = f.read().splitlines()
    ready_ROI_paths = []
    ROI_base_name = []
    for i in np.arange(len(lines)): # each line refere to the location of one ROI:
        
        Need_sampling_flag = Check_if_resampling_is_needed(lines[i], fourD_Full_path)
        if Need_sampling_flag == 0:
            temporary_ROI_path = lines[i]
        else:
            ROI_name = os.path.splitext(os.path.basename(lines[i]))[0].split('.nii')[0]
            resampling_ROI = resampling_and_masking(in_fil_to_be_changed=lines[i], Master_image_FULL_PATH = fourD_Full_path,\
                                                    changed_Out_file_name = ROI_name, where_to_save_changed_file = where_to_save_resampled_ROI)
            #""" This resampled_ROI will be saved in the "where_to_save_changed_file""""
            temporary_ROI_path = resampling_ROI.resampling_Afni()
        
         
        
        if os.path.isfile(MASK_file_FULL_PATH):
            mask_needs_resampling = Check_if_resampling_is_needed(temporary_ROI_path, MASK_file_FULL_PATH)
            if mask_needs_resampling == 0: 
                resampled_ROI_name = os.path.splitext(os.path.basename(temporary_ROI_path))[0].split('.nii')[0]
                #''' This means that the mask and resampled atlas are in the same dimensions'''
                masking_atlas = resampling_and_masking(in_fil_to_be_changed=temporary_ROI_path, MASK_file_FULL_PATH=MASK_file_FULL_PATH, changed_Out_file_name=resampled_ROI_name)
                masked_ROI = masking_atlas.Masking()
            else:
                print("mask was in different dimension than atlas and 4D file, so the mask is not applied to the atlas")
                
                masked_ROI = temporary_ROI_path
        else:
            print("no mask was provided, so the resampled atlas is not masked")
            masked_ROI = temporary_ROI_path
        ROI_base_name.append(os.path.splitext(os.path.basename(masked_ROI))[0].split('.nii')[0])
                
        ready_ROI_paths.append(masked_ROI)
        
    return ready_ROI_paths
    





#***************************************************************
#       Help functions for "vol_extraction"
#***************************************************************
def _ROI_stats_4D_extraction(ROI_resampled_bunch, fourD_GMV, Tnew, stats_arg, x):
    
    
    Stats=ImageStats()
    Stats.inputs.in_file = fourD_GMV
    ROI_resampled = ROI_resampled_bunch[x]
    if stats_arg == 'Median':
        Stats.inputs.op_string = '-k %s -p 50'
    elif stats_arg =='mean':
        Stats.inputs.op_string = '-k %s -m'
    Stats.inputs.split_4d = True
    Stats.inputs.mask_file = ROI_resampled
    #im_meant.inputs.out_file = 'mean_GMV_hipp.txt'
    Stats.inputs.terminal_output= 'stream'
    #K.cmdline
    #im_meant.inputs.mask_file
    K=Stats.run()
    #!ls -lt
    Tnew = K.outputs.out_stat
    return Tnew         
        
#            
#def ROI_stats_4D_extraction(ROI_resampled, fourD_GMV, stats_arg='Median'):
#    Stats=ImageStats()
#    Stats.inputs.in_file = fourD_GMV
#    if stats_arg == 'Median':
#        Stats.inputs.op_string = '-k %s -p 50'
#    elif stats_arg =='mean':
#        Stats.inputs.op_string = '-k %s -m'
#    Stats.inputs.split_4d = True
#    Stats.inputs.mask_file = ROI_resampled
#    #im_meant.inputs.out_file = 'mean_GMV_hipp.txt'
#    Stats.inputs.terminal_output= 'stream'
#    #K.cmdline
#    #im_meant.inputs.mask_file
#    K=Stats.run()
#    #!ls -lt
#    return K.outputs.out_stat
             









######################################################
# End of help functionsfor "vol_extraction"
######################################################



def vol_extraction(ROI_list_full_path,merged_file_full_path,Sample_info_table_CSV_full_path,where_to_save_ROI_Table,\
                   ROI_table_name_suffix, what_to_extract='Median', T_stats_Flag = 1, n_jobs =20): 
    # This can be replaced with NiftiSpheresMasker ??? what happens to the paralleling ?

    
    """ for each inputed merged_file_full_path it will craete a csv file with (all sample info + extracted values per ROI) saved in ROI forlder. 
   *** Base_name_for_ROIs:  1  means use the name of the ROI as named by 
    
    /Shahrzad'sPersonalFolder/HCP/Partial_cors
                                               /sample_CSV/Subjects_info_table.csv      # Info_table_CSV
                                               /4D_images/merged_image_name.nii.gz      # merged 4D images of the subsamples
                                               /GLM_Stats_dir/                          # Stats of GLM which are going to be used to create ROI
                                                             %test_var_name/
                                                                            
                                                                           fsl/
                                                                               
                                                                               %test_var_name_fsl_stats_tfce_corrp_tstat1/2.nii.gz  # Fixed Outputs saved   
                                                                               %test_var_name_fsl_stats_tstat1/2.nii.gz     # Fixed Outputs saved
                                                                               ROI/
                                                                                   tfce_corrp/
                                                                        ***************************OR*****************
                                                                           nilearn/
                                                                                   %test_var_name_nilearn_stats_FWE_corrp_tstat1/2.nii.gz  # Fixed Outputs saved  
                                                                                   %test_var_name_nilearn_stats_tstat1/2.nii.gz  # Fixed Outputs saved   
                                                                                   ROI/
                                                                                       fwe_corrp/
                                                                                       uncorr/
                                                                                             tstat1/
                                                                                                   index_%test_var_name_nilearn_stats_tstat1.nii.gz
                                                                                                   stats.txt
                                                                                                   list_of_ROIs_based_on_height/extent.txt ???
                                                                                                   binned_ROI_height/extent_%d.nii.gz 
                                                                                                   
                                                                                                   
                                                                                             tstat2/
                                                /Secondary_CSVs_for_correlations/
                                                                                Subjects_info_table_%what_to_extract_GMV_%ROI_table_name_suffix.csv
                                                                                
                                                                                
                                                                                
                                            
    
    
    """
                                                                            
    
    with open(ROI_list_full_path) as f:
        lines = f.read().splitlines()
        
    
    Tnew= [[] for i in np.arange(len(lines))] #### Here maybe a resampling needed to bring ROI to the space of T1 template?!
    Collected_parallel_Tnews = Parallel(n_jobs= n_jobs)(delayed(_ROI_stats_4D_extraction)(lines, merged_file_full_path, Tnew[x], what_to_extract, x) for x in np.arange(len(lines)))
    print(len(Collected_parallel_Tnews))
    print('lines\n')
    print(len(lines))
    print(lines)
    ###Here I want to extarct some part of the list path to name ROI-columns accordingly. 
    
    Base_name_for_ROIs = [os.path.splitext(os.path.basename(i))[0].split('.nii')[0] for i in lines]
    ## Or : from nipype.utils.filemanip import split_filename
    print(Base_name_for_ROIs)
    # Here we save these simple ROI names in a file. This file is used to refer to the columns of the table, when one wants to correlate the extrcated GMV with 
    try:
        os.makedirs(where_to_save_ROI_Table)
    except OSError:
        if not os.path.isdir(where_to_save_ROI_Table):
            raise
    ROI_base_names_text_file_full_path = os.path.join(where_to_save_ROI_Table,ROI_table_name_suffix + '.txt')
    ROI_base_names_text_file = open(ROI_base_names_text_file_full_path, 'a')
    
    for item in Base_name_for_ROIs:
        ROI_base_names_text_file.write("%s\n" % item)
    ROI_base_names_text_file.close()
    
    ROI_tables = pd.DataFrame()
    Sample_table_with_GMV_info_full_path = os.path.join(where_to_save_ROI_Table,\
                                                        os.path.splitext(os.path.basename(Sample_info_table_CSV_full_path))[0] + \
                                                                        '_' + what_to_extract + '_GMV_'+ ROI_table_name_suffix + '.csv')
    
    
    
    
    if T_stats_Flag ==1:
        
        
        
        ROIS_T_max_Full_path = os.path.join(where_to_save_ROI_Table,\
                                            'ROI_T_Max_table.pkl')
        ROIS_Cohen_d_Full_path = os.path.join(where_to_save_ROI_Table,\
                                              'ROI_Cohen_d_table.pkl')
        ROIS_r_efect_size_Full_path = os.path.join(where_to_save_ROI_Table,\
                                                   'ROI_r_efect_size_table.pkl')
        
        
        ROI_effects = {'ROIS_T_max_Full_path':ROIS_T_max_Full_path, 'ROIS_Cohen_d_Full_path': ROIS_Cohen_d_Full_path, 'ROIS_r_efect_size_Full_path':ROIS_r_efect_size_Full_path}
        #Cohen_d_df_full_path = os.path.join(ROI_dir, 'Cohen_d_of_binned_ROIS.pkl')
        #    Cohen_d_df.to_pickle(Cohen_d_df_full_path)
        #    r_efect_size_df_full_path = os.path.join(ROI_dir, 'r_efect_size_of_binned_ROIS.pkl')
        try:
            T_max_table = pd.read_pickle(os.path.join(os.path.dirname(lines[0]), 'T_max_of_binned_ROIS.pkl'))
            Cohen_d_table = pd.read_pickle(os.path.join(os.path.dirname(lines[0]), 'Cohen_d_of_binned_ROIS.pkl'))
            r_efect_size_table = pd.read_pickle(os.path.join(os.path.dirname(lines[0]), 'r_efect_size_of_binned_ROIS.pkl'))
        except:
            T_max_table = pd.DataFrame()
            Cohen_d_table = pd.DataFrame()
            r_efect_size_table = pd.DataFrame()
        try:
            Test_concat_T_max = pd.read_pickle(ROIS_T_max_Full_path)
            Test_concat_Cohen_d = pd.read_pickle(ROIS_Cohen_d_Full_path)
            Test_concat_r_efect_size = pd.read_pickle(ROIS_r_efect_size_Full_path)
        except:
            Test_concat_T_max = pd.DataFrame()
            Test_concat_Cohen_d = pd.DataFrame()
            Test_concat_r_efect_size = pd.DataFrame()
            
        
        Test_concat_T_max = pd.concat([Test_concat_T_max, T_max_table], axis = 1)
        Test_concat_Cohen_d = pd.concat([Test_concat_Cohen_d, Cohen_d_table], axis = 1)
        Test_concat_r_efect_size = pd.concat([Test_concat_r_efect_size, r_efect_size_table], axis = 1)
        if len(lines)>0:
            Test_concat_T_max.to_pickle(ROIS_T_max_Full_path)
            Test_concat_Cohen_d.to_pickle(ROIS_Cohen_d_Full_path)
            Test_concat_r_efect_size.to_pickle(ROIS_r_efect_size_Full_path)
    else:
        
        ROI_effects = {}
        
        
        
    try:
        Sample_info_table = pd.read_csv(Sample_table_with_GMV_info_full_path)
        
    except:
        Sample_info_table = pd.read_csv(Sample_info_table_CSV_full_path, delimiter=',')
        
    if len(lines)>0:
        ROI_tables = pd.DataFrame(np.column_stack(Collected_parallel_Tnews), columns = Base_name_for_ROIs)
        New_Table_with_stats_Test = pd.concat([Sample_info_table, ROI_tables], axis = 1)
        New_Table_with_stats_Test.to_csv(Sample_table_with_GMV_info_full_path, index = False)
        
    
    return Sample_table_with_GMV_info_full_path, ROI_base_names_text_file_full_path, ROI_effects

#%%# Help function for "Functional_profiling" function:
    
def tiedrank(X):
    Z = [(x, i) for i, x in enumerate(X)]  
    Z.sort()  
    n = len(Z)  
    Rx = [0]*n   
    start = 0 # starting mark  
    for i in range(1, n):  
        if Z[i][0] != Z[i-1][0]:
            for j in range(start, i):  
                Rx[Z[j][1]] = float(start+1+i)/2.0;
            start = i
    for j in range(start, n):  
        Rx[Z[j][1]] = float(start+1+n)/2.0;

    return Rx

def ranked_input(X):
    
    if len(X.shape) ==1:
        X = X[:,np.newaxis]
    if X.shape[1] >1:
        cc = []
        for i in np.arange(X.shape[1]):
            cc.append(tiedrank(X[:,i]))
        rX = np.array(cc).T
    else:   
        rX = tiedrank(X)
        rX = np.array(rX)
    
    return rX    

def OLS_residuals(X, Y, linear_reg_estimator):
    if len(Y.shape) ==1:
        Y = Y[:,np.newaxis]
    if len(X.shape) ==1:
        X = X[:,np.newaxis]
    linear_reg_estimator.fit(X, Y)
    Y_hat = linear_reg_estimator.predict(X)
    res = Y - Y_hat
    return res

def Functional_correlations(ROI_vol, Cognitive_test_score, Confounders_val = [], correlation_method = 'pearson'):
    ##***** ******
    """
    ROI_base_names_text_file_full_path :is the path to the text file, where column names of ROIs are written.+
    Cog_list_full_path_for_profiling: is the path to the text file where cognitive tests for which we want statistics are written ++++++ This could change into a more dynamic version, for example input of config file of snakemake workflow. 
    Cog_functions_for_profiling: list of all cognitive funtcions that we would like to calculate the correlations of GMV within ROI- 
    ROI_base_names_text_file_fullpath : files name in which "only" name of ROIS (how they are calle in the table) are written. The profilingwill be done for all ROIs of this list
    Sample_table_with_GMV_info__full_path: Table of the specific sample of interest with all variables as well as ROIs' GMV
    correlation_method : 'pearson' or 'spearman' or 'sPartial' or "linear_regression" (which is parametric)
    """
    if correlation_method == 'pearson':
        functional_corr_of_ROI = stats.pearsonr(ROI_vol, Cognitive_test_score)[0]
        P_value = stats.pearsonr(ROI_vol, Cognitive_test_score)[1]
    elif correlation_method == 'spearman':    
        functional_corr_of_ROI = stats.spearmanr(ROI_vol, Cognitive_test_score)[0]
        P_value = stats.spearmanr(ROI_vol, Cognitive_test_score)[1]
        
    else:
        
        if correlation_method == 'sPartial':
            ROI_vol= ranked_input(ROI_vol)
            Confounders_val = ranked_input(Confounders_val)
            Cognitive_test_score = ranked_input(Cognitive_test_score)
        
        res_roi = []
        lm = linear_model.LinearRegression(fit_intercept = True, copy_X = True, n_jobs = 2)
        res_roi = OLS_residuals(Confounders_val, ROI_vol, lm)
        res_cog = []
        lm = linear_model.LinearRegression(fit_intercept = True, copy_X = True, n_jobs = 2)
        res_cog = OLS_residuals(Confounders_val,Cognitive_test_score, lm)
#        if correlation_method == 'sPartial':
#            P_value  = stats.spearmanr(res_roi, res_cog)[1]
#            functional_corr_of_ROI = stats.spearmanr(res_roi, res_cog)[0]
#
#        else:
        functional_corr_of_ROI = stats.pearsonr(res_roi, res_cog)[0]
        P_value  = stats.pearsonr(res_roi, res_cog)[1]

           
    return P_value, [functional_corr_of_ROI]  # This is a dataframe with rows as Cognitive tetsts and columns as ROIs



def _Simple_Bootstrapping(ROI_base_name, Sample_csv_table, Cog_tests_name, Confounders_names, correlation_method, Tnew, n_boot):
    
                
    R = np.array(Sample_csv_table[ROI_base_name])
    T = np.array(Sample_csv_table[Cog_tests_name])
    C = np.array(Sample_csv_table[Confounders_names])
    if len(C) == 0:
        R, T = sk_resample(R, T,random_state=n_boot) # This is from sklearn.utils function
    
        _, Tnew = Functional_correlations(R, T, correlation_method = correlation_method)
        
    else:
        R, T, C = sk_resample(R, T, C, random_state=n_boot) # This is from sklearn.utils function
    
        _, Tnew = Functional_correlations(R, T, C , correlation_method = correlation_method)
        
    return Tnew # This is only one r value
    
def _Simple_Bootstrapping_CI(Collected_parallel_Tnews, alpha):
    """
    Collected_parallel_Tnews: a list of lists of 
    alpha: a value between 0 and 1

    """
    
                    
    flat_r_list = [item for sublist in Collected_parallel_Tnews for item in sublist]
    Corr_bootstrap = pd.DataFrame(flat_r_list, columns = ["boot_r"])
    Corr_bootstrap_stats = Corr_bootstrap.describe(percentiles=[round(alpha, 2), round(1-alpha, 2)])
    ci_low = min(round(100-round(alpha,2)*100),round(alpha*100))
    
    ci_high = max(round(100-round(alpha,2)*100),round(alpha*100))
    stats = tuple(Corr_bootstrap_stats.ix[["mean", str(ci_low)+"%", str(ci_high)+"%"], "boot_r"])
    
    return stats # This is a tuple which first element: mean , second element: low simple CI , thrid element: high simple CI
    
def plot_lib_error_bars(boot_stats_tuple):
    yerr= np.c_[boot_stats_tuple[0]-boot_stats_tuple[1],boot_stats_tuple[2]-boot_stats_tuple[0]].T
    return yerr



def find_percent_of_sig_boostraps(Collected_parallel_Tnews, critical_r, direction = "higher"):
    """ direction: "lower" or "higher" 
    """
    
    
    flat_r_list = [item for sublist in Collected_parallel_Tnews for item in sublist]

    if direction == "lower":
        
        sig_boots = 100*sum(i < critical_r for i in flat_r_list)/len(flat_r_list)
        
    else: 
        sig_boots = 100*sum(i > critical_r for i in flat_r_list)/len(flat_r_list)
    return sig_boots




                
#####End of help functions for "Functional_profiling_simple_boot":
# I will now change this to only deal with one ROI at a time. No Dataframes of multiple ROIS anymore:
## Why? To be able to deal easier with sorting the results. 
def Functional_profiling_simple_boot(ROI_name, Stats_dir, Sample_table_with_GMV_info__full_path, Cog_list_full_path_for_profiling,\
                                     Confounders_list_full_path = '', Group_selection_column_name = '', Group_division = False,\
                                     Group_selection_Label = '', Sort_correlations = False, correlation_method = 'pearson',alpha = 0.05, n_boot = 1000, n_jobs = 3):
    ''' This function could be run in paralle (I mean it can run bootstraps in parallel)'''
    
    
    
    ROI = ROI_name
    
    with open(Cog_list_full_path_for_profiling) as f:
        Cog_tests = f.read().splitlines()
        
    try:
        with open(Confounders_list_full_path) as f:
            Confounders = f.read().splitlines()
        
    except:
        Confounders = []
        correlation_method = 'pearson'
        print("empty confounder list was provided, therefore, bivariate pearson's correlation calculated")
    if len(Group_selection_Label)==0:
        Group_selection_Label = 'all'
    Sample_table = pd.read_csv(Sample_table_with_GMV_info__full_path, delimiter = ',')
    if Group_division == True and (Group_selection_Label != 'all'):
        try:
            Sample_table = Sample_table[Sample_table[Group_selection_column_name] == int(Group_selection_Label)]
        except:
            pass
        
        
    Table_for_corr = Sample_table[list(set(Cog_tests + [ROI] + Confounders))]
    
    means = pd.DataFrame(index = Cog_tests, columns = [ROI])
    P_Value = pd.DataFrame(index = Cog_tests, columns = [ROI])
    masked_mean = pd.DataFrame(index = Cog_tests, columns = [ROI])
    masked_P_Value = pd.DataFrame(index = Cog_tests, columns = [ROI])
    err = pd.DataFrame(index = Cog_tests, columns =  [ROI])
    CIs = pd.DataFrame(index = Cog_tests, columns =  [ROI])
    percent_sig_boots = pd.DataFrame(index = Cog_tests, columns = [ROI])
    available_n= pd.DataFrame(index = Cog_tests, columns = ['n_available'])
    # I think it does not make sense to make ROI, cognitive tests and
    #for j in np.arange(len(ROIS)):
    
    for i in np.arange(len(Cog_tests)):
        print(Cog_tests[i])
        #print(ROIS[j])
        print(ROI)
    
        Tnew= [[] for i in np.arange(n_boot)]
        Sample_csv_table=Table_for_corr[~Table_for_corr[Cog_tests[i]].isnull()]
        Collected_parallel_Tnews = Parallel(n_jobs= n_jobs)(delayed(_Simple_Bootstrapping)(ROI_base_name = ROI,\
                                            Sample_csv_table=Sample_csv_table, Cog_tests_name = Cog_tests[i], Confounders_names = Confounders,\
                                            correlation_method = correlation_method, Tnew=Tnew[x], n_boot = x) for x in np.arange(n_boot))
        
        #print(Collected_parallel_Tnews)
        
        
        boot_stats = _Simple_Bootstrapping_CI(Collected_parallel_Tnews, alpha)
        
        P_value, mean = Functional_correlations(np.array(Sample_csv_table[ROI]), np.array(Sample_csv_table[Cog_tests[i]]),\
                                                         np.array(Sample_csv_table[Confounders]) , correlation_method = correlation_method)
        means.loc[Cog_tests[i],ROI] = mean[0]
        P_Value.loc[Cog_tests[i],ROI] = P_value
        sign_of_CIs = np.sign(boot_stats[1]) *np.sign(boot_stats[2]) 
        
        # Now to identify percent of significant bootstraps( i test this one sided only)
        df = len(Sample_csv_table) - 2 - len(Confounders)      
        critical_t = Calculate_uncorrected_t_threshold_from_p(df,voxel_p_thresh = 0.05, test_side = 1)
        critical_r = Calculate_r_effect_size_from_T(df, critical_t)
        if mean[0] > 0:
            Sig_boots = find_percent_of_sig_boostraps(Collected_parallel_Tnews, critical_r, direction = "higher")
        else:
            Sig_boots = find_percent_of_sig_boostraps(Collected_parallel_Tnews, -1*critical_r, direction = "lower")

        percent_sig_boots.loc[Cog_tests[i],ROI] = Sig_boots
        
        
        
        if sign_of_CIs>0:
            masked_mean.loc[Cog_tests[i],ROI] = mean[0]
            masked_P_Value.loc[Cog_tests[i],ROI] = P_value
        else:
            masked_mean.loc[Cog_tests[i],ROI] = 0
            masked_P_Value.loc[Cog_tests[i],ROI] = 1
            
        yerr = plot_lib_error_bars(boot_stats_tuple = (mean[0], boot_stats[1],boot_stats[2])) 
        err.loc[Cog_tests[i],ROI] = yerr
        CIs.loc[Cog_tests[i],ROI] =[(boot_stats[1],boot_stats[2])]
        available_n.loc[Cog_tests[i],'n_available'] = len(Sample_csv_table)
        
        
    if Sort_correlations:
        means = means.reindex(means[ROI].abs().sort_values(ascending = False).index)
        masked_mean = masked_mean.reindex(means[ROI].abs().sort_values(ascending = False).index)
        P_Value = P_Value.reindex(means[ROI].abs().sort_values(ascending = False).index)
        masked_P_Value = masked_P_Value.reindex(means[ROI].abs().sort_values(ascending = False).index)
        err = err.reindex(means[ROI].abs().sort_values(ascending = False).index)
        CIs = CIs.reindex(means[ROI].abs().sort_values(ascending = False).index)
        available_n = available_n.reindex(means[ROI].abs().sort_values(ascending = False).index)
        percent_sig_boots = percent_sig_boots.reindex(means[ROI].abs().sort_values(ascending = False).index)
    
    
    
                             
    #Saving_Dir = os.path.join(Stats_dir, ROI, correlation_method + '_' + str(alpha)+ '_'+ str(n_boot) +'bootstrap_simple')
    Saving_Dir = os.path.join(Stats_dir, correlation_method + '_' + str(alpha)+ '_'+ str(n_boot) +'bootstrap_simple')
    
    try:
        os.makedirs(Saving_Dir)
    except OSError:
        if not os.path.isdir(Saving_Dir):
            raise    
            
            
    
        
        
        
        
    CIs_full_path = os.path.join(Saving_Dir, 'Sample_' + str(Group_selection_Label) +'_plotlib_CIs.pkl')           
    CIs.to_pickle(CIs_full_path)
    
    error_full_path = os.path.join(Saving_Dir, 'Sample_' + str(Group_selection_Label) +'_plotlib_error.pkl')           
    err.to_pickle(error_full_path)
    percent_sig_boots_Full_path = os.path.join(Saving_Dir, 'Sample_' + str(Group_selection_Label) +'_percent_sig_boots.pkl')           
    percent_sig_boots.to_pickle(percent_sig_boots_Full_path)
    mean_r_full_path = os.path.join(Saving_Dir, 'Sample_' + str(Group_selection_Label) +'_plotlib_means.pkl')
    means.to_pickle(mean_r_full_path)
    available_n_full_path = os.path.join(Saving_Dir, 'Sample_' + str(Group_selection_Label) +'_plotlib_available.pkl')
    available_n.to_pickle(available_n_full_path)
    masked_mean_r_full_path = os.path.join(Saving_Dir, 'Sample_' + str(Group_selection_Label) +'_plotlib_sig_' + str(100 *(1-2*alpha)) + '%CI_masked_means.pkl')
    masked_mean.to_pickle(masked_mean_r_full_path)
    P_Value_full_path = os.path.join(Saving_Dir, 'Sample_' + str(Group_selection_Label) +'_plotlib_P_values.pkl')
    P_Value.to_pickle(P_Value_full_path)
    masked_P_Value_full_path = os.path.join(Saving_Dir, 'Sample_' + str(Group_selection_Label) +'%CI_masked_P_values.pkl')
    masked_P_Value.to_pickle(masked_P_Value_full_path)
        
    
    return mean_r_full_path, error_full_path, available_n_full_path, masked_mean_r_full_path, P_Value_full_path, masked_P_Value_full_path, CIs_full_path, percent_sig_boots_Full_path
#%% Alternatively: This section is for functions that are needed for scikit boostrap function :

            

def Functional_correlations_bi_pearson(ROI_vol, Cognitive_test_score):
    
    
    functional_corr_of_ROI = stats.pearsonr(ROI_vol, Cognitive_test_score)[0]
     
    return functional_corr_of_ROI  # This is a dataframe with rows as Cognitive tetsts and columns as ROIs
    
def Functional_correlations_bi_spearman(ROI_vol, Cognitive_test_score):
    
    
    functional_corr_of_ROI = stats.spearmanr(ROI_vol, Cognitive_test_score)[0]
     
    return functional_corr_of_ROI  # This is a dataframe with rows as Cognitive tetsts and columns as ROIs
    
    
def Functional_correlations_Multiple_reg(ROI_vol, Cognitive_test_score, Confounders_val):
    res_roi = []
    lm = linear_model.LinearRegression(fit_intercept = True, copy_X = True, n_jobs = 2)
    res_roi = OLS_residuals(Confounders_val, ROI_vol, lm)
    res_cog = []
    lm = linear_model.LinearRegression(fit_intercept = True, copy_X = True, n_jobs = 2)
    res_cog = OLS_residuals(Confounders_val,Cognitive_test_score, lm)
    functional_corr_of_ROI = stats.pearsonr(res_roi, res_cog)[0]       
    return functional_corr_of_ROI  # This is a dataframe with rows as Cognitive tetsts and columns as ROIs
## End of help functions for "Functional_profiling_scikit_boot_bca"
## NOTE: Bootstrap_BCA does not create "sPartial" results only linear regression
def Functional_profiling_scikit_boot_bca(ROI_name, Stats_dir, Sample_table_with_GMV_info__full_path, Cog_list_full_path_for_profiling,\
                                         Confounders_list_full_path = '', Group_selection_column_name = '', Group_division = False,\
                                         Group_selection_Label = np.nan,Sort_correlations = False, correlation_method = 'pearson',alpha = 0.05, n_boot = 1000):
    
    ''' This also creates the error bars for each group separately'''
    ROI = ROI_name
        
        
    with open(Cog_list_full_path_for_profiling) as f:
        Cog_tests = f.read().splitlines()
            
    try:
        with open(Confounders_list_full_path) as f:
            Confounders = f.read().splitlines()
            
    except IOError:
        Confounders = []
        correlation_method = 'pearson'
        print("empty confounder list was provided, therefore, bivariate pearson's correlation calculated")
        
    
    Sample_table = pd.read_csv(Sample_table_with_GMV_info__full_path, delimiter = ',')
    if Group_division == True and (~np.isnan(Group_selection_Label)):
        try:
            Sample_table = Sample_table[Sample_table[Group_selection_column_name] == Group_selection_Label]
        except:
            pass
        
        
    Table_for_corr = Sample_table[Cog_tests + ROI + Confounders]
    
    means = pd.DataFrame(index = Cog_tests, columns = [ROI])
    err = pd.DataFrame(index = Cog_tests, columns = [ROI])
    available_n= pd.DataFrame(index = Cog_tests, columns = ['n_available'])
    
    if correlation_method == 'pearson':
        
            
        for i in np.arange(len(Cog_tests)):
            print(Cog_tests[i])
            print(ROI)
                
            T = np.array(Table_for_corr[Cog_tests[i]][~Table_for_corr[Cog_tests[i]].isnull()])
            R = np.array(Table_for_corr[ROI][~Table_for_corr[Cog_tests[i]].isnull()])
            CIS = bootstrap.ci(data=(R,T), statfunction=Functional_correlations_bi_pearson, method='bca',n_samples=n_boot, alpha= alpha)
            mean = Functional_correlations(R,T, correlation_method=correlation_method)
            available_n.loc[Cog_tests[i],'n_available'] = len(T)
            yerr = plot_lib_error_bars(boot_stats_tuple = (mean[0], CIS[0], CIS[1])) 
            err.loc[Cog_tests[i],ROI] = yerr
            means.loc[Cog_tests[i],ROI] = mean[0]
    elif correlation_method == 'spearman':
        
        for i in np.arange(len(Cog_tests)): 
            print(Cog_tests[i])
            print(ROI)    
            T = np.array(Table_for_corr[Cog_tests[i]][~Table_for_corr[Cog_tests[i]].isnull()])
            R = np.array(Table_for_corr[ROI][~Table_for_corr[Cog_tests[i]].isnull()])
            CIS = bootstrap.ci(data=(R,T), statfunction=Functional_correlations_bi_spearman, method='bca',n_samples=n_boot)
            _, mean = Functional_correlations(R,T, correlation_method=correlation_method)
            available_n.loc[Cog_tests[i],'n_available'] = len(T)
            
            yerr = plot_lib_error_bars(boot_stats_tuple = (mean[0], CIS[0], CIS[1])) 
            err.loc[Cog_tests[i],ROI] = yerr
            means.loc[Cog_tests[i],ROI] = mean[0]
    elif correlation_method == 'linear regression':
        
        for i in np.arange(len(Cog_tests)): 
            print(Cog_tests[i])
            print(ROI)    
            T = np.array(Table_for_corr[Cog_tests[i]][~Table_for_corr[Cog_tests[i]].isnull()])
            C = np.array(Table_for_corr[Confounders][~Table_for_corr[Cog_tests[i]].isnull()])
            R = np.array(Table_for_corr[ROI][~Table_for_corr[Cog_tests[i]].isnull()])
        
            CIS = bootstrap.ci(data=(R,T,C), statfunction=Functional_correlations_Multiple_reg, method='bca',n_samples=n_boot)
            mean = Functional_correlations(R,T,C, correlation_method=correlation_method)
            available_n.loc[Cog_tests[i],'n_available'] = len(T)    
            yerr = plot_lib_error_bars(boot_stats_tuple = (mean[0], CIS[0], CIS[1])) 
            err.loc[Cog_tests[i],ROI] = yerr
            means.loc[Cog_tests[i],ROI] = mean[0]
            
    else: 
        print("Error: correlation_method unkonwn for BCA_bootstrap")
        #break
    
    
    
    if Sort_correlations:
        means = means.reindex(means[ROI].abs().sort_values(ascending = False).index)
        err = err.reindex(means[ROI].abs().sort_values(ascending = False).index)
        available_n = available_n.reindex(means[ROI].abs().sort_values(ascending = False).index)
      
    
    if (np.isnan(Group_selection_Label)):
        Group_selection_Label = 'all'
    
                             
    Saving_Dir = os.path.join(Stats_dir, ROI, correlation_method+ '_' + str(alpha)+ '_'+ str(n_boot) +'bootstrap_bca')
    try:
        os.makedirs(Saving_Dir)
    except OSError:
        if not os.path.isdir(Saving_Dir):
            raise    
    error_full_path = os.path.join(Saving_Dir, 'Sample_' + str(Group_selection_Label) +'_plotlib_error.pkl')           
    err.to_pickle(error_full_path)
    mean_r_full_path = os.path.join(Saving_Dir, 'Sample_' + str(Group_selection_Label) +'_plotlib_means.pkl')
    means.to_pickle(mean_r_full_path)
    available_n_full_path = os.path.join(Saving_Dir, 'Sample_' + str(Group_selection_Label) +'_plotlib_available.pkl')
    available_n.to_pickle(available_n_full_path)
        
    
    return mean_r_full_path, error_full_path, available_n_full_path

 









     
    ########## Parameters for tests and defining significant cluetrs:
    # type of analysis: TFCE = FSL palm (needs additionally to create the design matrix , ...) or FWE-clusterlevel (nilearn)
    # I would suggest to create a class called: GLM and it should then have different functions, for example for TFCE should accept the design & ....
    
    ## in the end the output should be a p-value image (at Cluster level) & a t-map.
    
                                                      
                                                      
                                                
        
    ###############End of this program #################  


  
#%% Now functions for creating plots

def Plot_one_row_Functional_profile(ROI_mean_r, ROI_CI_err, plt_lable, Y_low =1, Y_high= -1, ax= '',sample_jitter = 0 , fmt = '.', plot_type = 'errordots'):
    ''' 

    ROI_mean_r: a dataframe with columns as ROIs and rows as  "Cog_Tests_for_profiling" means.iloc[:, i]
    ROI_CI_err: a dataframe with columns as ROIs and ros as  "Cog_Tests_for_profiling" err.iloc[:,i] (i: which ROI???)
    #Row_Plot_title : means.columns.values[i]
    Cog_Tests_for_profiling: name of Rows of ROI_mean_r and ROI_CI_err.  (i.e. ROI_mean_r.index.values)
    ax : ax[Which_cluster, 0] # So in roder to be ble to show all significant clusters of one test in different row of figure. 
    plot_type = 'errordots'Or 'bar'
    
    '''
    label = plt_lable
    #Cog_Tests_for_profiling = ROI_mean_r.index.values
    yerr = np.ndarray(shape = (2, len(ROI_CI_err)))
    
    for i in np.arange(len(ROI_CI_err)):
        yerr[0,i]= ROI_CI_err.iloc[i][0][0]
        yerr[1,i]= ROI_CI_err.iloc[i][0][1]
    
    
    Y_low = Y_low
    Y_high = Y_high
    for i in np.arange(len(ROI_mean_r)):
        temp_l = ROI_mean_r.iloc[i][0] - ROI_CI_err.iloc[i][0][0]
        temp_h = ROI_CI_err.iloc[i][0][1] + ROI_mean_r.iloc[i][0]
        
        Y_low = min(temp_l, Y_low)
        Y_high = max(temp_h,Y_high)
        
    Y_low = min(-0.05, Y_low)
    Y_high = max(0.05, Y_high)
    if plot_type =='errordots':
        ax.locator_params(tight=True ,axis='x', nbins=len(ROI_mean_r))
        ax.errorbar(y=ROI_mean_r.iloc[:,0], x=np.arange(0,2*len(ROI_mean_r),2)+ 1+ sample_jitter, yerr= yerr,fmt = fmt, label = label)
        ax.xaxis.set_ticks(np.arange(0,2*len(ROI_mean_r),2)+ 1)
        ax.axhline(0, color='red', linestyle = 'dashed', alpha= 0.2)
        ax.grid(alpha=0.2, ls ='dashed')
        ax.title.set_fontsize(10)
        #ax.titlepad(6.0)
    
    elif plot_type =='bar':   
        ax.bar(left = np.arange(len(ROI_mean_r))+ sample_jitter, height= ROI_mean_r.iloc[:,0], yerr=yerr)
       
        
         
    #ax.set(title=label)
    ax.set_ylim([Y_low-0.03,Y_high +0.03])
    return Y_low, Y_high

def Chunks(l, n):
    x = []
    for i in range(0, l, n):
        if i+n < l:
            x.append(np.arange(i,i + n))
        else:
            x.append(np.arange(i,l))
    
    return x
    

 #%% Plotting several samples results in one plot (so they share the same COg test name and ROI name , but they have different n_available)

def Plot_multiple_sample_one_page_Functional_profile(mean_r_DF_Full_path,CI_err_DF_Full_path,available_n_full_path,\
                                                     Page_Plot_name, qq, n_rows_one_page = 1, fmt = 'o',ranking_x_axis = True, percent_top_corr_plot = 10, mastre_group_id = 0, plot_type = 'errordots'):
    ''' 
    errorbar Dataframe and Means Dataframe are Sample specific and are from all ROIS of one test.
    So the dimensions are (n_clusters of one test x n_all_cognitive tests_for_profiling.)
    Page_Plot_name: is the name of original Test score,from its GLM we have found the clusters. 
    qq = PdfPages(os.path.join(Saving_figure_dir, 'NAME_of_PDF.pdf'))
    ''' 
    ##
    if percent_top_corr_plot > 1:
        percent_top_corr_plot = np.divide(percent_top_corr_plot, 100)
    
    frames_m = [pd.read_pickle(f) for f in mean_r_DF_Full_path]
    means_df = pd.concat(frames_m, keys=np.arange(len(mean_r_DF_Full_path)))
    frames_e = [pd.read_pickle(f) for f in CI_err_DF_Full_path]
    err_df = pd.concat(frames_e, keys=np.arange(len(CI_err_DF_Full_path)))
     
    
    
##    ## Test:
##    
#    limiting_factor = 5
#    idx = pd.IndexSlice
#    means_df = means_df.loc[idx[:,list(means_df.index.levels[1][0:limiting_factor])], :]
#    err_df = err_df.loc[idx[:,list(err_df.index.levels[1][0:limiting_factor])], :]
##    ####    
#    
    if ranking_x_axis == True:
        # Which sampel is our master sample, defining the order of the variables in X-axis?
        mastre_group_id = mastre_group_id
        n_top_correlations = np.ceil(percent_top_corr_plot*len(means_df.loc[mastre_group_id].index.values))
    
        Cog_Tests_for_profiling = means_df.loc[mastre_group_id].index.values[:int(n_top_correlations)]
        
    else:
        mastre_group_id = 0
        Cog_Tests_for_profiling = means_df.loc[0].index.values
    # Kind of strange with available data here:  So, only if there is one sample to plot, it will use the available_n info
    try:
        if isinstance(available_n_full_path, list):
            available_df = pd.read_pickle(available_n_full_path[mastre_group_id])
        else:
            available_df = pd.read_pickle(available_n_full_path)
        Cog_Tests_for_profiling_plot = Cog_Tests_for_profiling.copy()
        for i in np.arange(len(Cog_Tests_for_profiling)):
            Cog_Tests_for_profiling_plot[i] = Cog_Tests_for_profiling[i] + '\n' + str(available_df.iloc[i,0])
    except:
        Cog_Tests_for_profiling_plot = Cog_Tests_for_profiling.copy()
        pass
    ####### 
    actual_n_rows_one_page = min(n_rows_one_page,means_df.shape[1]) # still holds true for multiple samples. 
    # We do ot want more than X number of rows in one page
    Chunk_ind = Chunks(means_df.shape[1],actual_n_rows_one_page)
    
    for c_ind in np.arange(len(Chunk_ind)):
        # The loop is not anymore needed, as anyhow we will have only one plot per page.
        
        
        fig, axes = plt.subplots(nrows=actual_n_rows_one_page, ncols=1, squeeze=False, sharex=True,figsize=(10,10))
        
        means_ch = means_df
        
        err_ch= err_df
        
        
        i = 0 # this is referring to the number of rows in each page (here we only have one ro per page)    
        Row_Plot_name = means_ch.columns.values[0]
        
        Y_low =1
        Y_high= -1
        for df_ind in np.arange(len(means_ch.index.levels[0])):
            # separation of different samples is happening here
            
            temp = means_ch.loc[(df_ind, slice(None))]
            ROI_mean_r = temp.reindex(Cog_Tests_for_profiling, copy= True)
            temp_err = err_ch.loc[(df_ind, slice(None))]
            ROI_CI_err = temp_err.reindex(Cog_Tests_for_profiling, copy= True)
         
            
            #ROI_mean_r = means_ch.ix[df_ind,i]
            
            
            plt_lable = 'sample'+ str(df_ind)
            sample_jitter = df_ind * 0.21
            Y_low , Y_high = Plot_one_row_Functional_profile(ROI_mean_r, ROI_CI_err, plt_lable, Y_low =Y_low, Y_high= Y_high, ax=axes[i,0], sample_jitter=sample_jitter,fmt = fmt, plot_type = plot_type)
            
            #row_legend.append() 
        axes[i, 0].set_ylabel(Row_Plot_name, fontsize =5)
        axes[i, 0].set_yticks(np.around(np.arange(Y_low , Y_high, 0.2),1))
        axes[i, 0].set_yticklabels(np.around(np.arange(Y_low , Y_high, 0.2), 1), fontsize='xx-small')
        

        axes[i, 0].legend(loc='upper right', fontsize = 'xx-small')
        
        #axes[means_df.shape[1]-1, 0].set_xticks(np.arange(len(ROI_mean_r))+1, Cog_Tests_for_profiling)
        if plot_type == 'bar':
            axes[means_ch.shape[1]-1, 0].set_xticklabels([''] +list(Cog_Tests_for_profiling_plot), rotation=90, fontsize =5)
        
        else:
            #axes[means_ch.shape[1]-1, 0].set_xticklabels([''] +list(Cog_Tests_for_profiling), rotation=45, fontsize =5)
            axes[means_ch.shape[1]-1, 0].set_xticklabels(list(Cog_Tests_for_profiling_plot), rotation=90, fontsize =5)
        
        #ax.set_xticks(np.arange(len(means_df[0,:]))+ 0.2)
        #axes[means_df.shape[1]-1, 0].set_title(Page_Plot_name)
        fig.suptitle(Page_Plot_name)
        #fig.tight_layout()
        
        fig.savefig(qq, format='pdf')    

    return 



def  Create_multiple_sample_Functional_profile_PDFs(Saving_figure_dir, Full_file_with_Saving_dir_of_meansANDerrors ='',\
                                                    Group_selection_Label = 'all', mean_r_DF_Full_path= '', CI_err_DF_Full_path='',\
                                                    available_n_full_path = '', Page_Plot_name='',name_of_pdf= 'NAME_of_PDF',\
                                                    n_rows_one_page = 1, fmt = 'o', mastre_group = '',percent_top_corr_plot = 10, plot_type = 'errordots'):
    ''' if it is not a result of my program (i.e. it is not organized in the folder struture I have propoosed)
        then one can only provide the mean_df_path and CI_path and see the resulted images in "Saving_figure_dir"
    '''
    
    ##
    if len(mastre_group)==0:
        mastre_group_id = 0
    else:
        mastre_group_id = Group_selection_Label.index(str(mastre_group))
    try:
            os.makedirs(Saving_figure_dir)
    except OSError:
        if not os.path.isdir(Saving_figure_dir):
            raise
    qq = PdfPages(os.path.join(Saving_figure_dir, name_of_pdf+'.pdf'))
    
    
    if os.path.isfile(Full_file_with_Saving_dir_of_meansANDerrors):
        with open(Full_file_with_Saving_dir_of_meansANDerrors) as f:
            lines = f.read().splitlines()
            
    else:
        
        lines = []
        
        Plot_multiple_sample_one_page_Functional_profile(mean_r_DF_Full_path = mean_r_DF_Full_path,CI_err_DF_Full_path = CI_err_DF_Full_path,\
                                                         available_n_full_path = available_n_full_path, Page_Plot_name = Page_Plot_name,\
                                                         qq = qq, n_rows_one_page = n_rows_one_page, fmt = fmt, percent_top_corr_plot = percent_top_corr_plot,\
                                                         mastre_group_id = mastre_group_id, plot_type = plot_type)

    if len(lines) > 0:
        
        for main_folder in lines:
        
            if len(Page_Plot_name) == 0:
                Page_Plot_name = os.path.dirname(lines).split('/')[-2] # This is very specific for way of saving of my files for the moment 
            
            mean_r_DF_Full_path = [os.path.join(main_folder, 'Sample_' + str(i) +'_plotlib_means.pkl') for i in Group_selection_Label]
            CI_err_DF_Full_path = [os.path.join(main_folder,'Sample_' + str(i) +'_plotlib_error.pkl')for i in Group_selection_Label]
            available_n_full_path=[os.path.join(main_folder, 'Sample_' + str(i) +'_plotlib_available.pkl')for i in Group_selection_Label]
            Plot_multiple_sample_one_page_Functional_profile(mean_r_DF_Full_path = mean_r_DF_Full_path,CI_err_DF_Full_path = CI_err_DF_Full_path,\
                                                             available_n_full_path = available_n_full_path, Page_Plot_name = Page_Plot_name,\
                                                             qq = qq, n_rows_one_page = n_rows_one_page, fmt = fmt, percent_top_corr_plot = percent_top_corr_plot,\
                                                             mastre_group_id = mastre_group_id, plot_type = plot_type)
    
    qq.close()
    
    return os.path.join(Saving_figure_dir, name_of_pdf+'.pdf')
  
#%% here I create plots for only one cognitive test and all its ROIS. This should be one plot per page and only one page. 
## Help function for "Plot_multiple_sample_one_page_ROIstability"
    
    
# Function that gets a list of ROIs and merges their repsective mean /err / available pickles to create a dataframe.
 
def merging_dataframes_columnwise(DF_Full_path, Full_name_and_path_of_the_new_dataframe):
    frames_m = [pd.read_pickle(f) for f in DF_Full_path]
    df = pd.concat(frames_m, axis = 1, join='inner') # This will merge the dataframes of the 
    df.to_pickle(os.path.splitext(Full_name_and_path_of_the_new_dataframe)[0] + '.pkl')
    return os.path.splitext(Full_name_and_path_of_the_new_dataframe)[0] + '.pkl'
        
        

def Plot_one_row_ROIstability(ROI_mean_r, ROI_CI_err, plt_lable, Y_low =1, Y_high= -1, ax= '',sample_jitter = 0 , fmt = '.', plot_type = 'errordots'):
    ''' 

    ROI_mean_r: a dataframe with columns as ROIs and rows as  "Cog_Tests_for_profiling" means.iloc[:, i]
    ROI_CI_err: a dataframe with columns as ROIs and ros as  "Cog_Tests_for_profiling" err.iloc[:,i] (i: which ROI???)
    #Row_Plot_title : means.columns.values[i]
    Cog_Tests_for_profiling: name of Rows of ROI_mean_r and ROI_CI_err.  (i.e. ROI_mean_r.index.values)
    ax : ax[Which_cluster, 0] # So in roder to be ble to show all significant clusters of one test in different row of figure. 
    plot_type = 'errordots'Or 'bar'
    
    '''
    label = plt_lable
    # We want to have all the ROIs on the rows and one column which is the ROI that we are plotting its stability
    
    ROI_CI_err = ROI_CI_err.transpose()
    
    ROI_mean_r = ROI_mean_r.transpose()
    #Cog_Tests_for_profiling = ROI_mean_r.index.values
    yerr = np.ndarray(shape = (2, len(ROI_CI_err)))
    
    for i in np.arange(len(ROI_CI_err)):
        
        yerr[0,i]= ROI_CI_err.iloc[i][0][0][0]
        yerr[1,i]= ROI_CI_err.iloc[i][0][1][0]
    #yerr
    
    
    Y_low = Y_low
    Y_high = Y_high
    for i in np.arange(len(ROI_mean_r)):
        
        temp_l = ROI_mean_r.iloc[i][0] - ROI_CI_err.iloc[i][0][0][0]
        temp_h = ROI_CI_err.iloc[i][0][1][0] + ROI_mean_r.iloc[i][0]
        Y_low = min(temp_l, Y_low)
        Y_high = max(temp_h,Y_high)
        
    Y_low = min(-0.05, Y_low)
    Y_high = max(0.05, Y_high)
    
    if plot_type =='errordots':
        ax.locator_params(nbins=len(ROI_mean_r))
        
        ax.errorbar(y=ROI_mean_r.iloc[:, 0], x=np.arange(len(ROI_mean_r))+1+ sample_jitter, yerr= yerr,fmt = fmt, label = label)
        ax.xaxis.set_ticks(np.arange(len(ROI_mean_r)+1) +1)
        ax.axhline(0, color='red', linestyle = 'dashed', alpha= 0.2)
        ax.grid(alpha=0.2, ls ='dashed')
    
    elif plot_type =='bar':   
        ax.bar(left = np.arange(len(ROI_mean_r))+ sample_jitter, height= ROI_mean_r, yerr=yerr) 
         
    #ax.set(title=label)
    ax.set_ylim([Y_low-0.03,Y_high +0.03])
    return Y_low, Y_high


####END of Help funtcion for "Plot_multiple_sample_one_page_ROIstability"
def Plot_multiple_sample_one_page_ROIstability(mean_r_DF_Full_path,CI_err_DF_Full_path,available_n_full_path,\
                                               Cog_test, qq, n_rows_one_page = 1, fmt = 'o', plot_type = 'errordots'):
    ''' 
    errorbar Dataframe and Means Dataframe are Sample specific and are from all ROIS of one test.
    So the dimensions are (n_clusters of one test x n_all_cognitive tests_for_profiling.)
    Page_Plot_name: is the name of original Test score,from its GLM we have found the clusters. 
    qq = PdfPages(os.path.join(Saving_figure_dir, 'NAME_of_PDF.pdf'))
    ''' 
    ##
    
    frames_m = [pd.read_pickle(f) for f in mean_r_DF_Full_path]
    means_df = pd.concat(frames_m, keys=np.arange(len(mean_r_DF_Full_path)))
    frames_e = [ pd.read_pickle(f) for f in CI_err_DF_Full_path]
    err_df = pd.concat(frames_e, keys=np.arange(len(CI_err_DF_Full_path)))
    ## SO Which cognitive test we have to look at? 
    # this should be the ROI not cog test 
    ROI_name_for_stability = means_df.columns.values
    print(ROI_name_for_stability)
    # Kind of strange with available data here:  So, only if there is one sample to plot, it will use the available_n info
    try:
        #if isinstance(available_n_full_path, list):
        #    available_df = pd.read_pickle(available_n_full_path[0])
        #else:
        #    available_df = pd.read_pickle(available_n_full_path)
        
        #for i in np.arange(len(ROI_name_for_stability)):
        #    ROI_name_for_stability[i] = ROI_name_for_stability[i] + '\n' + str(available_df.iloc[i,0])
        
        frames_av = [pd.read_pickle(f) for f in available_n_full_path]
        available_df = pd.concat(frames_av, keys=np.arange(len(available_n_full_path)))
    
    except:
        pass
    ####### 
    actual_n_rows_one_page = 1 # still holds true for multiple samples. 
    # We do ot want more than X number of rows in one page
    Chunk_ind = [0]
    for c_ind in np.arange(len(Chunk_ind)):
        fig, axes = plt.subplots(nrows=actual_n_rows_one_page, ncols=1, squeeze=False, sharex=True)
        #gs = gridspec.GridSpec(len(means_df), 1)
        #ax = plt.subplot(gs[:,0])
        means_df.sort_index(inplace=True)
        means_ch = means_df.loc(axis=0)[:, [Cog_test]]  # has all the samples and for each only one row (cog_test) and all comlumns are the ROIs
        err_df.sort_index(inplace=True)
        err_ch= err_df.loc(axis=0)[:, [Cog_test]]
        available_df.sort_index(inplace=True)
        available_ch = available_df.loc(axis=0)[:, [Cog_test]] 
        #for i in np.arange(means_ch.shape[1]): # OLD:over ROIS
        i = 0
        Row_Plot_name = Cog_test
        Y_low =1
        Y_high= -1
        for df_ind in np.arange(len(means_ch.index.levels[0])): # over samples
            print(df_ind)
            ROI_mean_r = means_ch.ix[df_ind,:]
            ROI_CI_err = err_ch.ix[df_ind,:]
            plt_lable = 'sample'+ str(df_ind) + '_# ' + str(available_ch.ix[df_ind,0][0])
            sample_jitter = df_ind * 0.2
            Y_low , Y_high = Plot_one_row_ROIstability(ROI_mean_r, ROI_CI_err, plt_lable, Y_low =Y_low, Y_high= Y_high, ax=axes[i,0], sample_jitter=sample_jitter,fmt = fmt, plot_type = plot_type)
            #row_legend.append() 
        axes[i, 0].set_ylabel(Row_Plot_name, fontsize =5)
        axes[i, 0].set_yticks(np.around(np.arange(Y_low , Y_high, 0.2),1))
        axes[i, 0].set_yticklabels(np.around(np.arange(Y_low , Y_high, 0.2), 1), fontsize='xx-small')

        axes[i, 0].legend(loc='upper right', fontsize = 'xx-small')
        
        #axes[means_df.shape[1]-1, 0].set_xticks(np.arange(len(ROI_mean_r))+1, ROI_name_for_stability)
        if plot_type == 'bar':
            axes[actual_n_rows_one_page-1, 0].set_xticklabels(list(ROI_name_for_stability), rotation=45, fontsize =5)
        
        else:
            axes[actual_n_rows_one_page-1, 0].set_xticklabels(list(ROI_name_for_stability)+ [''] , rotation=45, fontsize =5)
        #ax.set_xticks(np.arange(len(means_df[0,:]))+ 0.2)
        #axes[means_df.shape[1]-1, 0].set_title(Page_Plot_name)
        #fig.suptitle(Row_Plot_name)
        #fig.tight_layout()
        fig.tight_layout()
        fig.savefig(qq, format='pdf')    

    return 

def  Create_multiple_sample_ROIstability_PDFs(Saving_figure_dir, Full_file_with_Saving_dir_of_meansANDerrors ='',\
                                              Group_selection_Label = 'all', mean_r_DF_Full_path= '', CI_err_DF_Full_path='',\
                                              available_n_full_path = '', Cog_test='',name_of_pdf= 'NAME_of_PDF',\
                                              n_rows_one_page = 1, fmt = 'o',  plot_type = 'errordots'):
    ''' if it is not a result of my program (i.e. it is not organized in the folder struture I have propoosed)
        then one can only provide the mean_df_path and CI_path and see the resulted images in "Saving_figure_dir"
    '''
    
    try:
            os.makedirs(Saving_figure_dir)
    except OSError:
        if not os.path.isdir(Saving_figure_dir):
            raise
    qq = PdfPages(os.path.join(Saving_figure_dir, name_of_pdf+'.pdf'))
    
    if os.path.isfile(Full_file_with_Saving_dir_of_meansANDerrors):
        with open(Full_file_with_Saving_dir_of_meansANDerrors) as f:
            lines = f.read().splitlines()
            
    else:
        lines = []
        
        Plot_multiple_sample_one_page_ROIstability(mean_r_DF_Full_path = mean_r_DF_Full_path,CI_err_DF_Full_path = CI_err_DF_Full_path,\
                                                   available_n_full_path = available_n_full_path, Cog_test = Cog_test,\
                                                   qq = qq, n_rows_one_page = n_rows_one_page, fmt = fmt, plot_type = plot_type)

    if len(lines) > 0:
        
        for main_folder in lines:
        
            #Cog_test = os.path.dirname(lines).split('/')[-2] # This is very specific for way of saving of my files for the moment 
            
            mean_r_DF_Full_path = [os.path.join(main_folder, 'Sample_' + str(i) +'_plotlib_means.pkl') for i in Group_selection_Label]
            CI_err_DF_Full_path = [os.path.join(main_folder,'Sample_' + str(i) +'_plotlib_error.pkl')for i in Group_selection_Label]
            available_n_full_path=[os.path.join(main_folder, 'Sample_' + str(i) +'_plotlib_available.pkl')for i in Group_selection_Label]
            Plot_multiple_sample_one_page_ROIstability(mean_r_DF_Full_path = mean_r_DF_Full_path,CI_err_DF_Full_path = CI_err_DF_Full_path,\
                                                       available_n_full_path = available_n_full_path, Cog_test = Cog_test,\
                                                       qq = qq, n_rows_one_page = n_rows_one_page, fmt = fmt, plot_type = plot_type)
    
    qq.close()
    
    return



################ 
#%% Here I create a function that chooses an ROI from Atlas (based on csv file of the atlas)
#### Help Functions for "Nilearn_Atlas_based_Vol_Extraction"
    
class resampling_and_masking(object):
    def __init__(self,in_fil_to_be_changed,Master_image_FULL_PATH = '', MASK_file_FULL_PATH ='', changed_Out_file_name = '', where_to_save_changed_file = ''):
         
        self.in_fil_to_be_changed = in_fil_to_be_changed
        self.Master_image_FULL_PATH = Master_image_FULL_PATH
        self.MASK_file_FULL_PATH = MASK_file_FULL_PATH
        self.changed_Out_file_name = changed_Out_file_name
        if os.path.isfile(self.MASK_file_FULL_PATH): 
            self.master = self.MASK_file_FULL_PATH
        else:
            self.master = self.Master_image_FULL_PATH
        
        if os.path.isdir(where_to_save_changed_file):
            self.Dir = where_to_save_changed_file
        else:
            self.Dir = os.path.dirname(self.master)
            
        
        
        self.voxle_size = abs(nb.load(self.master).affine[0,0]) # this gives info about voxel-dimension of master (in case it is isotropic, it might be helpful for naming)
        
    def resampling_Afni(self):
        from nipype.interfaces import afni as afni
        os.chdir(self.Dir)
        resample = afni.Resample()
        resample.inputs.in_file = self.in_fil_to_be_changed
        resample.inputs.master = self.master
        resample.inputs.out_file = "resampled_" + str(self.voxle_size) +"_"+ self.changed_Out_file_name +".nii.gz"
        resample.inputs.outputtype = "NIFTI_GZ"
        resample.run()
        return os.path.join(self.Dir, resample.inputs.out_file)
    def Masking(self):
        
        from nipype.interfaces.fsl.maths import ApplyMask
        os.chdir(self.Dir)
        masking = ApplyMask()
        masking.inputs.in_file = self.in_fil_to_be_changed
        masking.inputs.mask_file = self.MASK_file_FULL_PATH
        masking.inputs.out_file = "masked_" + self.changed_Out_file_name +".nii.gz" 
        masking.run()
        return os.path.join(self.Dir,masking.inputs.out_file)
    
    
        
        
def Check_if_resampling_is_needed(File_1_full_path, File_2_full_path):
    if isinstance(File_1_full_path, nb.nifti1.Nifti1Image): 
        file_1_image_affine = File_1_full_path.affine
        file_1_shape = File_1_full_path.shape
    else:
        file_1_image_affine = nb.load(File_1_full_path).affine
        file_1_shape = nb.load(File_1_full_path).shape
        
    if isinstance(File_2_full_path, nb.nifti1.Nifti1Image): 
        file_2_image_affine = File_2_full_path.affine
        file_2_shape = File_2_full_path.shape
    else:
        file_2_image_affine = nb.load(File_2_full_path).affine
        file_2_shape = nb.load(File_2_full_path).shape
          
    if np.array_equal(file_1_image_affine, file_2_image_affine) and (file_1_shape == file_2_shape):
        Need_sampling_flag = 0
    else:
        Need_sampling_flag = 1
    return Need_sampling_flag
    #if np.trace(file_1_image_affine) == np.trace(file_2_image_affine)
        
        
### End of Help functions for "Nilearn_Atlas_based_Vol_Extraction"    

def Nilearn_Atlas_based_Vol_Extraction(fourD_Full_path, where_to_save_ROI_Table, MASK_file_FULL_PATH ='', ROI_table_name_suffix ='atlas_based',\
                                       Atlas_name = "cort-maxprob-thr25-2mm", Atlas = "harvard_oxford", ROI_index = [1,2],\
                                       Sample_info_table_CSV_full_path ='', FSL_DIR = '/usr/share/fsl/5.0/'):
    ''' it needs the list of abbreviations used for the ROInames'''
    
    from nilearn.input_data import NiftiLabelsMasker
    from nilearn import datasets
    if 'sub' in Atlas_name:
        symmetric_split = False
    else:
        symmetric_split =True
    dataset = datasets.fetch_atlas_harvard_oxford(Atlas_name, symmetric_split=symmetric_split)
    atlas_filename = dataset.maps
    labels = dataset.labels
    atlas_filename
    
    if len(ROI_index) == 0:
        ''' This means that we are interested in all ROIS'''
        selected_labels =labels[1:]
    else:
        selected_labels =list( labels[i] for i in ROI_index)
        
        
        
#    if Atlas_name == "aal":
#        #Atlas_label_image_full_path = "/data/BnB2/USER/Shahrzad/eNKI_modular/Masks/AAL_ATLAS_TEMPLATE_1_25/masked_aal_1_25_mm.nii.gz"
#        Atlas_label_image_full_path = "/data/BnB2/TOOLS/CAT_versions/cat12/templates_1.50mm/aal.nii"
#        Atlas_CSV_full_path = "/data/BnB2/TOOLS/CAT_versions/cat12/templates_1.50mm/aal.csv"
#        Atlas_table = pd.read_csv(Atlas_CSV_full_path, delimiter=';')
#    
#        selected_labels = []
#        if ROI_names == ['all']:
#            # This means save mean volume of all ROIs of the atlas to the table:
#            ROI_names = list(Atlas_table["ROIabbr"])
#            selected_labels = list(Atlas_table["ROIid"]-1)[1:]
#            
#            
#        for i in ROI_names:
#            # Here I add a "-1" to the id, as when time-series are extracted by nilearn, 
#            # 0 is chosen as background and is not calculated and is not accounted in the columsn of time_series,
#            # so the numbering there starts from 0 for the id = 1 in the atlas.  
#            selected_labels.append( Atlas_table.loc[Atlas_table["ROIabbr"] == i].ROIid.values[0] -1) 
            
    Atlas_needs_resampling = Check_if_resampling_is_needed(fourD_Full_path, atlas_filename)
    if Atlas == "harvard_oxford" and isinstance(atlas_filename, nb.nifti1.Nifti1Image):
        if symmetric_split and ('cort' in Atlas_name): # this means I have to look
            rest_Atlas_name = Atlas_name.split('cort')[1]
            Atlas_label_image_full_path = os.path.join(FSL_DIR, 'data/atlases/HarvardOxford/HarvardOxford-cortl' +rest_Atlas_name + '.nii.gz')
        else: # the subcortical atlas of FSL is already 
            Atlas_label_image_full_path = os.path.join(FSL_DIR, 'data/atlases/HarvardOxford/HarvardOxford-' + Atlas_name + '.nii.gz')
        
    
    if Atlas_needs_resampling == 1:
        ''' this means resampling is needed, So atlas is not n the same dimension as 4d file'''
        
        resampling_atlas = resampling_and_masking(in_fil_to_be_changed=Atlas_label_image_full_path, Master_image_FULL_PATH = fourD_Full_path, changed_Out_file_name = Atlas_name)
        """ This resampled_atlas will be saved in the same folder as master"""
        resampled_Atlas = resampling_atlas.resampling_Afni()
        if os.path.isfile(MASK_file_FULL_PATH): 
            mask_needs_resampling = Check_if_resampling_is_needed(resampled_Atlas, MASK_file_FULL_PATH)
            if mask_needs_resampling == 0: 
                ''' This means that the mask and resampled atlas are in the same dimensions'''
                making_atlas =  resampling_and_masking(in_fil_to_be_changed=resampled_Atlas, MASK_file_FULL_PATH=MASK_file_FULL_PATH, changed_Out_file_name="resampled_" +  Atlas_name)
                masked_atlas = making_atlas.Masking()
            else:
                print("mask was in different dimension than atlas and 4D file, so the mask is not applied to the atlas")
                
                masked_atlas = resampled_Atlas
        else:
            print("no mask was provided, so the resampled atlas is not masked")
            masked_atlas = resampled_Atlas
    else:
        print("Atlas was in the same dimension with the 4D file of subjects")                              
        masked_atlas = Atlas_label_image_full_path


                                        
                                                       
    masker = NiftiLabelsMasker(labels_img=masked_atlas,\
                               standardize=False, memory='nilearn_cache', verbose=5)
    time_series = masker.fit_transform(fourD_Full_path) 
    ids = list( i-1 for i in ROI_index)
    for i in np.arange(len(selected_labels)):
        
        selected_labels[i] = selected_labels[i].replace(',', '_')
        selected_labels[i] = selected_labels[i].replace(' ', '_')
    
    ROI_mean_volumes = pd.DataFrame(time_series[:, ids], columns= selected_labels)
    
    # Here we save these simple ROI names in a file. This file is used to refer to the columns of the table, when one wants to correlate the extrcated GMV with 
    try:
        os.makedirs(where_to_save_ROI_Table)
    except OSError:
        if not os.path.isdir(where_to_save_ROI_Table):
            raise
    ROI_base_names_text_file_full_path = os.path.join(where_to_save_ROI_Table,ROI_table_name_suffix + '.txt')
    ROI_base_names_text_file = open(ROI_base_names_text_file_full_path, 'a')
    
    for item in selected_labels:
        ROI_base_names_text_file.write("%s\n" % item)
    ROI_base_names_text_file.close()
    
    
    Sample_table_with_GMV_info_full_path = os.path.join(where_to_save_ROI_Table,\
                                                        os.path.splitext(os.path.basename(Sample_info_table_CSV_full_path))[0] + \
                                                        '_' + 'mean_GMV_'+ ROI_table_name_suffix + '.csv')
    
    try:
        Sample_info_table = pd.read_csv(Sample_table_with_GMV_info_full_path)
        
    except:
        Sample_info_table = pd.read_csv(Sample_info_table_CSV_full_path, delimiter=',')
        
    
    #ROI_tables = pd.DataFrame(np.column_stack(Collected_parallel_Tnews), columns = Base_name_for_ROIs)
    New_Table_with_stats_Test = pd.concat([Sample_info_table, ROI_mean_volumes], axis = 1)
    New_Table_with_stats_Test.to_csv(Sample_table_with_GMV_info_full_path, index = False)
    
    return Sample_table_with_GMV_info_full_path, ROI_base_names_text_file_full_path

    
    
#%%
    


def Schaefer_Parcelation_based_Vol_Extraction(fourD_Full_path, where_to_save_ROI_Table, MASK_file_FULL_PATH =None, ROI_table_name_suffix ='Schaefer2018',\
                                              n_YEO_ordered_networks = 7, n_parcels = 400,\
                                              Sample_info_table_CSV_full_path ='',\
                                              brain_parcelation_dir = "/data/BnB2/USER/Shahrzad/Schaefer_atlas/CBIG-0.3.2-Schaefer2018_LocalGlobal/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/"):
    from nilearn.input_data import NiftiLabelsMasker
    ''' This function is used to extract All treamed mean GMV within all ROIs of Shcaefer parcelations (100,200, 400, 600, 800)
    The parcelation map 
    
    '''
    try:
        os.makedirs(where_to_save_ROI_Table)
    except OSError:
        if not os.path.isdir(where_to_save_ROI_Table):
            raise
    schaefer_image =  os.path.join(brain_parcelation_dir, "Schaefer2018_"+str(n_parcels)+ "Parcels_" + str(n_YEO_ordered_networks) +"Networks_order_FSLMNI152_1mm.nii.gz")
    masker = NiftiLabelsMasker(labels_img=schaefer_image,mask_img = MASK_file_FULL_PATH,\
                               standardize=False, memory='nilearn_cache', verbose=5)
    time_series = masker.fit_transform(fourD_Full_path)
    schaefer_image_LUT = os.path.join(brain_parcelation_dir, "Schaefer2018_"+str(n_parcels)+ "Parcels_" + str(n_YEO_ordered_networks) +"Networks_order.txt")
    lut_test =pd.read_csv(schaefer_image_LUT, delim_whitespace=True, header=None)
    lut_test.columns = ['label', 'name_netwrok','R', 'G', 'B', 't']
    column_names= lut_test.name_netwrok.astype(str) + '_label_' + lut_test.label.astype(str)
    ROI_table = pd.DataFrame(data= time_series, columns= list(column_names))
    ROI_base_names_text_file_full_path = os.path.join(where_to_save_ROI_Table,ROI_table_name_suffix + '_ROIs.txt')
    ROI_base_names_text_file = open(ROI_base_names_text_file_full_path, 'w+')
    
    for item in list(column_names):
        ROI_base_names_text_file.write("%s\n" % item)
    ROI_base_names_text_file.close()
    
   
    #
    Sample_table_with_GMV_info_full_path = os.path.join(where_to_save_ROI_Table,\
                                                        os.path.splitext(os.path.basename(Sample_info_table_CSV_full_path))[0] + \
                                                                        '_' +ROI_table_name_suffix+'_' +str(n_parcels)+ "Parcels_" + str(n_YEO_ordered_networks) +'Networks_order.csv')
    
    try:
        Sample_info_table = pd.read_csv(Sample_table_with_GMV_info_full_path)
        
    except:
        Sample_info_table = pd.read_csv(Sample_info_table_CSV_full_path, delimiter=',')
        
    New_Table_with_ROI_vols = pd.concat([Sample_info_table, ROI_table], axis = 1)
    New_Table_with_ROI_vols.to_csv(Sample_table_with_GMV_info_full_path, index = False)
    # Now maybe merge the ROI table to sample table. 
    return Sample_table_with_GMV_info_full_path , ROI_base_names_text_file_full_path




def task_Wise_correlation_to_nii(row_element, Label_image,template_image , where_to_save_nii):
    """ schaefer parcellations should have the same number of ROIs as the row element. 
    template_image is the MNI image in the same resolution as the schaefer parcelation
    Label_image= schaefer_image
    """
    masker_label = NiftiLabelsMasker(labels_img=Label_image,\
                                     standardize=False, memory='nilearn_cache', verbose=5)
    
    time_series = masker_label.fit_transform(template_image)

    
    empty_list = []
    for i in np.arange(row_element.shape[0]):
        
        try:
            empty_list.append(row_element.iloc[i][0])
        except: 
            empty_list.append(0)

             
    D= np.array(empty_list)
    D = D[:, np.newaxis]
    data_brain = masker_label.inverse_transform(D.T)
    file_name_full = os.path.join(where_to_save_nii, row_element.name + "stable_correlations" + str(row_element.shape[0]) + "parcels.nii.gz")
    data_brain.to_filename(file_name_full)
    return file_name_full
#%% 
def Create_bin_ROI_image_from_a_indexed_image(index_vol_full_path,label_value, Output_ROI_bin_image_full_path): 


    Masker_obj = NiftiMasker()
    Indexed_image_arr = Masker_obj.fit_transform(index_vol_full_path)
    Cluster_ar = np.empty(Indexed_image_arr.shape)
    Cluster_ar = np.logical_and(Indexed_image_arr >= (label_value-0.001) , Indexed_image_arr <= (label_value + 0.001))
    Masker_obj.inverse_transform(Cluster_ar).to_filename(Output_ROI_bin_image_full_path)

    return Output_ROI_bin_image_full_path 










#def Plotting_dotplot_CIS(Sample_ROI_Specific_mean_corr_full_Path, Sample_ROI_Specific_error_name_full_Path, Cog_list_full_path_for_profiling,Ax_array,low_y_limit, high_y_limit, Group_selection_Label =''):
#'''
#
#Cog_list_full_path_for_profiling: could either be cognitive profiling of one ROI, or it can be different Cluster names (correlation of one )
#'''      
#      ## Here the problem is how to plot all three samples next to each other. 
#
#    means = np.load(Sample_ROI_Specific_mean_corr_full_Path)
#    err = pd.read_pickle(Sample_ROI_Specific_error_name_full_Path)
#    yerr = np.ndarray(shape = (2, len(err)))
#    for i in np.arange(len(err)):
#        yerr[0,i]= err.iloc[i, 0][0]
#        yerr[1,i]= err.iloc[i, 0][1]
#     
#    width = 0.02
#    Ax_array.bar(np.arange(len(means[0,:]))+ width*which_sample_num, means[0,:], yerr=yerr[:,:],width=width, color = color) #, orientation = "horizontal"
#   
#      
#    Ax_array.errorbar(x=means, y=np.arange(len(means[:, 0])) +1, xerr= yerr,fmt = 'o', color = 'k')
#    (x = np.arange(len(means[0,:]))+ width*which_sample_num, means[0,:], yerr=yerr[:,:],width=width, color = color) #, orientation = "horizontal"
#    #plt.errorbar(x=means, y=np.arange(len(means[:, 0])) +1, xerr= yerr,fmt = 'o', color = 'k')
#    #plt.yticks((0, 1, 2, 3, 4), ('', Cog_tests[0],Cog_tests[1], Cog_tests[2],'')) 
#
#  
#    return yerr, low_y_limit, high_y_limit
#    
#      
### Not using this part anymore. The multiple_sample versions have the ability to plot one sample as well.     
def Plot_one_page_Functional_profile(mean_r_DF_Full_path,CI_err_DF_Full_path,available_n_full_path, Page_Plot_name, qq, n_rows_one_page = 5):
    ''' 
    errorbar Dataframe and Means Dataframe are Sample specific and are from all ROIS of one test.
    So the dimensions are (n_clusters of one test x n_all_cognitive tests_for_profiling.)
    Page_Plot_name: is the name of original Test score,from its GLM we have found the clusters. 
    qq = PdfPages(os.path.join(Saving_figure_dir, 'NAME_of_PDF.pdf'))
    ''' 
    err_df = pd.read_pickle(CI_err_DF_Full_path)
    means_df = pd.read_pickle(mean_r_DF_Full_path)
    Cog_Tests_for_profiling = means_df.index.values
    
    try:
        available_df = pd.read_pickle(available_n_full_path)
        for i in np.arange(len(Cog_Tests_for_profiling)):
            Cog_Tests_for_profiling[i] = Cog_Tests_for_profiling[i] + '\n' + str(available_df.iloc[i,0])
    except:
        pass
    actual_n_rows_one_page = min(n_rows_one_page,means_df.shape[1]) 
    # We do ot want more than X number of rows in one page
    Chunk_ind = Chunks(means_df.shape[1],actual_n_rows_one_page)
    
    for c_ind in np.arange(len(Chunk_ind)):
        fig, axes = plt.subplots(nrows=actual_n_rows_one_page, ncols=1, squeeze=False, sharex=True)
        #gs = gridspec.GridSpec(len(means_df), 1)
        #ax = plt.subplot(gs[:,0])
            
        means_ch = means_df.ix[:,list(Chunk_ind[c_ind])]
        err_ch= err_df.ix[:,list(Chunk_ind[c_ind])]
        
        for i in np.arange(means_ch.shape[1]):
            Row_Plot_name = means_ch.columns.values[i]
            ROI_mean_r = means_ch.iloc[:, i]
            ROI_CI_err = err_ch.iloc[:, i]
            Plot_one_row_Functional_profile(ROI_mean_r, ROI_CI_err, Row_Plot_name, ax= axes[i,0], plot_type = 'errordots')
            axes[i, 0].set_ylabel(Row_Plot_name, fontsize =10)
        
        #axes[means_df.shape[1]-1, 0].set_xticks(np.arange(len(ROI_mean_r))+1, Cog_Tests_for_profiling)
        
        axes[means_ch.shape[1]-1, 0].set_xticklabels([' '] +list(Cog_Tests_for_profiling), rotation=30, fontsize =10)
        #ax.set_xticks(np.arange(len(means_df[0,:]))+ 0.2)
        #axes[means_df.shape[1]-1, 0].set_title(Page_Plot_name)
        fig.suptitle(Page_Plot_name)
        fig.tight_layout()
        #fig.tight_layout()
        fig.savefig(qq, format='pdf')    

    return 



def Create_Functional_profile_PDFs(Saving_figure_dir, Full_file_with_Saving_dir_of_meansANDerrors ='', Group_selection_Label = 'all',\
                                   mean_r_DF_Full_path= '', CI_err_DF_Full_path='', available_n_full_path = '', Page_Plot_name='',\
                                   n_rows_one_page = 5):
    ''' if it is not a result of my program (i.e. it is not organized in the folder struture I have propoosed)
        then one can only provide the mean_df_path and CI_path and see the resulted images in "Saving_figure_dir"
    '''
    
    try:
            os.makedirs(Saving_figure_dir)
    except OSError:
        if not os.path.isdir(Saving_figure_dir):
            raise
    qq = PdfPages(os.path.join(Saving_figure_dir, 'NAME_of_PDF.pdf'))
    
    try:
        with open(Full_file_with_Saving_dir_of_meansANDerrors) as f:
            lines = f.read().splitlines()
        
    except:
        lines = []
        
        Plot_one_page_Functional_profile(mean_r_DF_Full_path = mean_r_DF_Full_path,CI_err_DF_Full_path = CI_err_DF_Full_path,\
                                         available_n_full_path = available_n_full_path, Page_Plot_name = Page_Plot_name,\
                                         qq = qq, n_rows_one_page = n_rows_one_page)

    if len(lines) > 0:
        
        for main_folder in lines:
        
            Page_Plot_name = os.path.dirname(lines).split('/')[-2] # This is very specific for way of saving of my files for the moment 
            mean_r_DF_Full_path = os.path.join(main_folder, 'Sample' + str(Group_selection_Label) +'plotlib_means.pkl')
            CI_err_DF_Full_path = os.path.join(main_folder,'Sample' + str(Group_selection_Label) +'plotlib_error.pkl')
            available_n_full_path= os.path.join(main_folder, 'Sample_' + str(Group_selection_Label) +'_plotlib_available.pkl')
            if ~available_n_full_path.exists():
                available_n_full_path = ''
            Plot_one_page_Functional_profile(mean_r_DF_Full_path = mean_r_DF_Full_path,CI_err_DF_Full_path = CI_err_DF_Full_path,\
                                             available_n_full_path = available_n_full_path, Page_Plot_name = Page_Plot_name,\
                                             qq = qq, n_rows_one_page = n_rows_one_page)
    
    qq.close()
    
    return
