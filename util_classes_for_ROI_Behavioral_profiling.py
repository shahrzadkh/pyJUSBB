#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 13:38:28 2018

@author: skharabian
"""

###So how does each step looklike?

import os
import numpy as np 
import pandas as pd
from shutil import copyfile


import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.ioff() #http://matplotlib.org/faq/usage_faq.html (interactive mode)
from GLM_and_ROI_generation import importing_table_from_csv,\
                                    create_test_var_specific_subsamples,\
                                    several_split_train_test_index_generation,\
                                    several_split_ADNI_Site_matched_train_test_index_generation,\
                                    load_NIFTI_DATA_FROM_TABLE,\
                                    vol_extraction,\
                                    geting_arbitrary_ROIS_ready,\
                                    Schaefer_Parcelation_based_Vol_Extraction,\
                                    Create_multiple_sample_Functional_profile_PDFs,\
                                    Functional_profiling_scikit_boot_bca,\
                                    Functional_profiling_simple_boot
                                    





    
class Sample_handling(object):
    def __init__(self,Base_saving_dir,workdir_name ='', specific_name_for_csv =''):
        self.specific_csv_file_name = specific_name_for_csv
        self.Base_dir = Base_saving_dir
        self.Working_dir =  os.path.join(self.Base_dir,workdir_name)
        try:
            os.makedirs(self.Working_dir)
        except OSError:
            if not os.path.isdir(self.Working_dir):
                raise
                
                
    def main_CSV_generation(self, Main_Sample_info_table_CSV_full_path, Important_variables=[],\
                exclusion_criteria = 'loose'):
                                       
        """ 
    

        exclusion_criteria= 'loose', 'moderate' or 'strict'
        So input fixed structure: 
        
        Base_working_dir: this is where my working directory is: /Shahrzad'sPersonalFolder/HCP/test_var_name_partial_corr/ 
        It assumes that the folder will be from now like this structured: 
            
          /Shahrzad'sPersonalFolder/Main_study/working_dir/
                                                   /sample_CSV/main_sample_%test_var_name.csv      # Info_table_CSV                                                    
        
        """
        # Where the Csv file needs to be located? (FIXED path) 
        Info_table_dir = os.path.join(self.Working_dir, 'sample_CSV')
        try:
                os.makedirs(Info_table_dir)
        except OSError:
            if not os.path.isdir(Info_table_dir):
                raise
        # name of Info table of variables of the sample it should be a csv (with Fix column names for sex, age and TIV) 
        Table_main_original = importing_table_from_csv(Main_Sample_info_table_CSV_full_path)
        
        # keep only subjects that are not nan for all the variables in the Important_variables list_
        if len(Important_variables)>0:
            Table_main_filtered = Table_main_original.dropna(subset=Important_variables)
        else:
            Table_main_filtered = Table_main_original
        ### Here I exclude based on the Diagnosis table, subjects not healthy enough:
        Diagnosis_Exclusion = exclusion_criteria + "_exclusion"
        if Table_main_filtered.filter(items=[Diagnosis_Exclusion]).shape[1] >0:
            Table_main_filtered = Table_main_filtered[Table_main_filtered[Diagnosis_Exclusion].isin(["Exclude"]) == False].reset_index(drop = True)
       
        
        ## Now save this as main table of the sample. 
        Table_main_filtered_full_path = os.path.join(Info_table_dir, 'main_sample_'+self.specific_csv_file_name+'.csv') 
        Table_main_filtered.to_csv(Table_main_filtered_full_path, index = False)
    
        return Table_main_filtered_full_path, self.Working_dir
            
    def Split_column_generation(self, subsampling_scripts_base_dir, main_csv_path,\
                                Sex_col_name= 'Sex',Age_col_name= 'Age_current',\
                                SITE_col_name = '',n_bins_age = 2, add_dummy_site_to_confounders = 1,\
                                Confounders_list_full_path ='',Age_step_size = 10,test_sample_size = 0.5,\
                                gender_selection = None):

                                
                                
        
        
        if SITE_col_name=='':
        
            n_split = 1
            # When all finalized maybe take the one function (binning_age" out of Mod_01... and delete the subsampling_sc...)
            Train_index, Test_index = several_split_train_test_index_generation(subsampling_scripts_base_dir,\
                                                                                main_csv_path,\
                                                                                n_split,\
                                                                                Sex_col_name,\
                                                                                Age_col_name,\
                                                                                Age_step_size,\
                                                                                test_sample_size,\
                                                                                gender_selection)
            x = n_split-1
            main_Table = importing_table_from_csv(main_csv_path)
            main_Table.loc[np.array(Train_index['train_' + str(x)]) , 'Which_sample'] = 1
            main_Table.loc[np.array(Test_index['test_'  + str(x)]) , 'Which_sample'] = 2
            if len(self.specific_csv_file_name) > 0 :
                new_samples_base_name = self.specific_csv_file_name + '.csv'
            else:
                new_samples_base_name = os.path.basename(main_csv_path).strip("main_sample_")
            Split_sample_CSV_dir= os.path.join(self.Working_dir,'sample_CSV')
            try:
                os.makedirs(Split_sample_CSV_dir)
            except OSError:
                if not os.path.isdir(Split_sample_CSV_dir):
                    raise
            
            grouped_main_sample_full_path = os.path.join(Split_sample_CSV_dir, 'grouped_main_sample_' + new_samples_base_name)
            main_Table.to_csv(grouped_main_sample_full_path, index = False)
            
            
            new_Confounders_list_full_path = os.path.join(Split_sample_CSV_dir , os.path.basename(Confounders_list_full_path))
            
            copyfile(Confounders_list_full_path, new_Confounders_list_full_path)
            
        else:
            n_split = 1
            # When all finalized maybe take the one function (binning_age" out of Mod_01... and delete the subsampling_sc...)
            Train_index, Test_index = several_split_ADNI_Site_matched_train_test_index_generation(subsampling_scripts_base_dir,\
                                                                                                  main_csv_path,\
                                                                                                  n_split,\
                                                                                                  Sex_col_name,\
                                                                                                  Age_col_name,\
                                                                                                  SITE_col_name,\
                                                                                                  n_bins_age,\
                                                                                                  test_sample_size,\
                                                                                                  gender_selection)
            x = n_split-1
            main_Table = importing_table_from_csv(main_csv_path)
            #main_Table[Sex_col_name+'_num'] = main_Table[Sex_col_name].map({"Female": 0, "Male": 1})
    
            main_Table.loc[np.array(Train_index['train_' + str(x)]) , 'Which_sample'] = 1
            main_Table.loc[np.array(Test_index['test_'  + str(x)]) , 'Which_sample'] = 2
            Split_sample_CSV_dir= os.path.join(self.Working_dir,'sample_CSV')
            try:
                os.makedirs(Split_sample_CSV_dir)
            except OSError:
                if not os.path.isdir(Split_sample_CSV_dir):
                    raise
            if add_dummy_site_to_confounders == 1:
                # Since With Site added for matching, it might be that some subjects are neither part of group 1 nor group2: 
                #14.02.2019 Ahemd and i commented this line: 
                #main_Table = main_Table[pd.notnull(main_Table['Which_sample'])]
        
                # As Now for every site in sample 1 I have a subject in the same site in sample 2, now I can also create the dummies for SITES in the main CSV:
                    ### To add SIte as a covariate
                Site_name_gr1 = []
                unique_sites=[]
                unique_sites= main_Table[SITE_col_name].unique()
                for i in unique_sites:
                    Site_name_gr1.append("SITE_"+str(i))
                
                main_Table = pd.get_dummies(main_Table, columns=[SITE_col_name])
                new_Confounders_list_full_path = os.path.join(Split_sample_CSV_dir , os.path.basename(Confounders_list_full_path))
                
                copyfile(Confounders_list_full_path, new_Confounders_list_full_path)
                with open(new_Confounders_list_full_path) as f:
                    Confounders = f.read().splitlines()
                # Now add the dummy SITE variables' names to the confounder list
                Confounders = Confounders + Site_name_gr1[:-1]
            
                confounds_names_full_file = open(new_Confounders_list_full_path, "w")
                for items in Confounders: 
                    confounds_names_full_file.write("%s\n" % items)
                        
                confounds_names_full_file .close()
                
            else:
                new_Confounders_list_full_path = os.path.join(Split_sample_CSV_dir , os.path.basename(Confounders_list_full_path))
                
                copyfile(Confounders_list_full_path, new_Confounders_list_full_path)
            
            
            if len(self.specific_csv_file_name) > 0 :
                new_samples_base_name = self.specific_csv_file_name + '.csv'
            else:
                new_samples_base_name = os.path.basename(main_csv_path).strip("main_sample_")
            
            grouped_main_sample_full_path = os.path.join(Split_sample_CSV_dir, 'grouped_main_sample_' + new_samples_base_name)
            main_Table.to_csv(grouped_main_sample_full_path, index = False)
        return grouped_main_sample_full_path, new_Confounders_list_full_path
        
    
            


###
class Imaging_data_handling(object):
    def __init__(self,Base_saving_dir,workdir_name ='', Sample_CSV_path ='', MASK_file_FULL_PATH =''):
        self.MASK_file_FULL_PATH = MASK_file_FULL_PATH
        self.Sample_CSV_path = Sample_CSV_path
        self.Base_dir = Base_saving_dir
        self.Working_dir =  os.path.join(self.Base_dir,workdir_name)
        try:
                os.makedirs(self.Working_dir)
        except OSError:
            if not os.path.isdir(self.Working_dir):
                raise
                
    
    def fourD_file_generation_from_table(self,Image_top_DIR,merged_image_name ='',\
                                         Base_Sample_name = 'NKI',modulation_method = 'non_linearOnly',\
                                         Smoothing_kernel_FWHM=0, load_nifti_masker_Flag=0):
        Mask_file_complete = self.MASK_file_FULL_PATH
        Sample_info_table_path = self.Sample_CSV_path
        merged_sample_dir = os.path.join(self.Working_dir,'4D_images')  
        try:
            os.makedirs(merged_sample_dir)
        except OSError:
            if not os.path.isdir(merged_sample_dir):
                raise
        if not merged_image_name:
            merged_image_name = os.path.splitext(os.path.basename(Sample_info_table_path))[0]
        Sample_info_table = pd.read_csv(Sample_info_table_path)
        
        output = load_NIFTI_DATA_FROM_TABLE(Image_top_DIR, Sample_info_table, merged_sample_dir,\
                                            merged_image_name, Mask_file_complete, Base_Sample_name, \
                                            modulation_method, load_nifti_masker_Flag, Smoothing_kernel_FWHM)
        return output["merged_file_path"] # This is a dict
    



            
    def make_ROI_ready(self,predefined_ROI_list_path, merged_file_path = '', where_to_save_resampled_ROI =''):
        if where_to_save_resampled_ROI == '':
            where_to_save_resampled_ROI = os.path.join(self.Working_dir, 'resamled_ROIS')
        Mask_file_complete = self.MASK_file_FULL_PATH
        ready_ROI_paths = geting_arbitrary_ROIS_ready(predefined_ROI_list_path,\
                                                      merged_file_path,\
                                                      where_to_save_resampled_ROI,\
                                                      Mask_file_complete)
        ROI_lists_file = os.path.join(self.Working_dir,'ready_ROI_paths.txt') ## This is a file, in each line has the ROI_list_full_path for different tests, ...

        f = open(ROI_lists_file, 'w+')
        for i in ready_ROI_paths:
            
            f.write(i+'\n')
        f.close()
        
        return ROI_lists_file
    
    
    def CSV_with_ROI_Volume(self, ROI_lists_file,  merged_file_path = '',where_to_save_ROI_Table = '',\
                            ROI_table_name_suffix = '', what_to_extract='mean', n_jobs =1, T_stats_Flag=0):
        if where_to_save_ROI_Table == '':
            
            where_to_save_ROI_Table = os.path.join(self.Working_dir, 'Secondary_CSVs_for_correlations')
    
        Sample_info_table_CSV_full_path = self.Sample_CSV_path
        Sample_table_with_GMV_info_full_path, ROI_base_names_text_file_full_path, __ = vol_extraction(ROI_lists_file,merged_file_path,\
                                                                                                      Sample_info_table_CSV_full_path,\
                                                                                                      where_to_save_ROI_Table, ROI_table_name_suffix,\
                                                                                                      what_to_extract,T_stats_Flag, n_jobs)
        return Sample_table_with_GMV_info_full_path, ROI_base_names_text_file_full_path
        
    def CSV_with_Schaefel_atlas_ROI_Volume(self, merged_file_path = '', brain_parcelation_dir = '',\
                                           where_to_save_ROI_Table = '', ROI_table_name_suffix = '',\
                                           n_parcels='', n_YEO_ordered_networks = ''):
        if where_to_save_ROI_Table == '':
            
            where_to_save_ROI_Table = os.path.join(self.Working_dir, 'Secondary_CSVs_for_correlations')
    
        Sample_info_table_CSV_full_path = self.Sample_CSV_path
        Sample_table_with_GMV_info_full_path, ROI_base_names_text_file_full_path = Schaefer_Parcelation_based_Vol_Extraction(merged_file_path,\
                                                                                                                             where_to_save_ROI_Table,\
                                                                                                                             self.MASK_file_FULL_PATH,\
                                                                                                                             ROI_table_name_suffix,\
                                                                                                                             n_YEO_ordered_networks,\
                                                                                                                             n_parcels,\
                                                                                                                             Sample_info_table_CSV_full_path, brain_parcelation_dir)

        return Sample_table_with_GMV_info_full_path, ROI_base_names_text_file_full_path
        
                      



######
        
class Behavioral_Profiling(object):
    def __init__(self,Base_saving_dir,workdir_name ='',Sample_CSV_path = '', Cog_list_full_path_for_profiling = '',\
                 Confounders_list_full_path = '', Group_selection_column_name = '', Group_division = False,\
                 Sort_correlations = True, correlation_method = '', alpha = 0.05, n_boot = 10):
        
        self.Base_dir = Base_saving_dir
        self.Working_dir =  os.path.join(self.Base_dir,workdir_name)
        try:
            os.makedirs(self.Working_dir)
        except OSError:
            if not os.path.isdir(self.Working_dir):
                raise
        self.Sample_CSV_path = Sample_CSV_path
        self.Cog_list_full_path_for_profiling = Cog_list_full_path_for_profiling
        self.Confounders_list_full_path = Confounders_list_full_path
        self.Group_selection_column_name = Group_selection_column_name
        self.Group_division = Group_division
        #self.Group_selection_Label = Group_selection_Label
        self.Sort_correlations = Sort_correlations
        self.correlation_method = correlation_method
        self.alpha = alpha
        self.n_boot = n_boot
        
    def ROI_Behavioral_profiling(self,ROI_name, stats_dir, Group_selection_Label = 'all', n_jobs=1):
        Sample_table_with_GMV_info__full_path = self.Sample_CSV_path
        
        r_full_path, err_full_path, avail_n_full_path, masked_r_full_path,\
        P_Val_full_path, masked_P_Val_full_path, CIs_full_path,_ = Functional_profiling_simple_boot(ROI_name,stats_dir, Sample_table_with_GMV_info__full_path,\
                                                                                                  self.Cog_list_full_path_for_profiling ,self.Confounders_list_full_path,\
                                                                                                  self.Group_selection_column_name, self.Group_division,\
                                                                                                  Group_selection_Label, self.Sort_correlations, self.correlation_method,\
                                                                                                  self.alpha, self.n_boot, n_jobs)

        return r_full_path, err_full_path, avail_n_full_path, masked_r_full_path, P_Val_full_path, masked_P_Val_full_path, CIs_full_path
        
    
    
#class BLack_Box_4(object):
#    def __init__():
#        
#        self..Base_dir = Base_saving_dir
#        self.Working_dir =  os.path.join(self.Base_dir,workdir_name)
#        try:
#            os.makedirs(self.Working_dir)
#        except OSError:
#            if not os.path.isdir(self.Working_dir):
#                raise
#        
#        
#        
#    def Figure_output(self,):
#        name = 'Functional_profile_'+ ROI_name  + '_' + Functional_profiling_settings["correlation_method"]
#        
#        PDF_File_path = Create_multiple_sample_Functional_profile_PDFs(Output_generation_settings["Saving_output_dir"],\
#                                                                       Group_selection_Label = Functional_profiling_settings["Group_selection_Label"],mean_r_DF_Full_path = groupped_mean,\
#                                                                       CI_err_DF_Full_path = groupped_err, available_n_full_path = groupped_av,\
#                                                                       Page_Plot_name = name,name_of_pdf = name,n_rows_one_page = 1,\
#                                                                       mastre_group = Output_generation_settings["master_group"],\
#                                                                       percent_top_corr_plot = Output_generation_settings["percent_top_corr_plot"],\
#                                                                       plot_type = Output_generation_settings["plot_type"])
#   