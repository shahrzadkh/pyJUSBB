#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 06:55:46 2019

@author: brain
"""


from __future__ import print_function
import json
from argparse import ArgumentParser
from gooey import Gooey, GooeyParser
import os
import numpy as np 
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.ioff() #http://matplotlib.org/faq/usage_faq.html (interactive mode)
from GLM_and_ROI_generation import Create_multiple_sample_Functional_profile_PDFs
from util_classes_for_ROI_Behavioral_profiling import Sample_handling, Imaging_data_handling, Behavioral_Profiling                                    

# Only chnage now should be to delete the fix merged 4D file. And hopefully using only AD subjects, the merging is easilly done in python. 




@Gooey(program_name="ROI Behavioral profiling Tool")
def parse_args():
    """ Use GooeyParser to build up the arguments we will use in our script
    Save the arguments in a default json file so that we can retrieve them
    every time we run the script.
    """
    stored_args = {}
    # get the script name without the extension & use it to build up
    # the json filename
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    args_file = "{}-args.json".format(script_name)
    # Read in the prior arguments as a dictionary
    if os.path.isfile(args_file):
        with open(args_file) as data_file:
            stored_args = json.load(data_file)
    parser = GooeyParser(description='This version is tested for eNKI data & may work for multi-site data as well')
    
    requirered_args=parser.add_argument_group('Required arguments')
    requirered_args.add_argument('--base_analysis_directory',
                                 required=True,
                                 action='store',
                                 default=stored_args.get('base_analysis_directory'),
                                 widget='DirChooser',
                                 help="Where to save the results")
    
    requirered_args.add_argument('--analysis_directory_name',
                                 required=True,
                                 action='store',
                                 default=stored_args.get('analysis_directory_name'),
                                 help="analysis_directory_name")
    
    
    requirered_args.add_argument('--Main_Sample_info_table_CSV_full_path',
                                 required=True,
                                 action='store',
                                 default=stored_args.get('Main_Sample_info_table_CSV_full_path'),
                                 widget='FileChooser',
                                 help='Main_Sample_info_table_CSV_full_path')
    
    
    parser.add_argument('--Other_important_variable',
                        required=False,
                        action='store',
                        default=stored_args.get('Other_important_variable'),
                        help="Other_important_variable")
    
    requirered_args.add_argument('--Confounders_txt_file',
                                 required=True,
                                 action='store',
                                 default=stored_args.get('Confounders_txt_file'),
                                 widget='FileChooser',
                                 help='Confounders_txt_file')
    
#    parser.add_argument('Diagnosis Criteria',
#                        action='store',
#                        default=stored_args.get('Diagnosis_exclusion_criteria'),
#                        help="Diagnosis_exclusion_criteria is a \ncolumn of 1 and missing, with a name: \nloos Or strict Or empty")

    parser.add_argument('--Diagnosis_exclusion_criteria', 
                        required=False,
                        metavar='Diagnosis_exclusion_criteria', help="Name of a column if it exists Otherwise choose 'None' \nChoose one option ",
                        widget="Listbox", nargs='+', choices=['loose', 'strict', 'none'],
                        default=stored_args.get('Diagnosis_exclusion_criteria'))



    requirered_args.add_argument('--subsampling_scripts_base_dir',
                                 required=True,
                                 action='store',
                                 widget='DirChooser',
                                 default=stored_args.get('subsampling_scripts_base_dir'),
                                 help="Directory, where 'Mod_01_binning_shahrzad.py' exists")

    requirered_args.add_argument('--gender_column_name',
                                 required=True,
                                 action='store',
                                 default=stored_args.get('gender_column_name'),
                                 help="gender column name in your main sample table")
    requirered_args.add_argument('--AGE_column_name',
                                 required=True,
                                 action='store',
                                 default=stored_args.get('AGE_column_name'),
                                 help="AGE column name in your main sample table")
    parser.add_argument('--SITE',
                        required=False,
                        action='store',
                        default=stored_args.get('SITE'),
                        help="SITE column name in your main sample table,\nonly if you want site as dummy var")
   
    requirered_args.add_argument('--ROI_full_path',
                                 required=True,
                                 action='store',
                                 default=stored_args.get('ROI_full_path'),
                                 widget='FileChooser',
                                 help='A text file with full path of ROIs')
#    parser.add_argument('what_to_extract',
#                        widget='dropdown',
#                        choices=['mean', 'Median'],
#                        default=stored_args.get('what_to_extract'),
#                        help="what_to_extract: mean or Median")
    requirered_args.add_argument('--what_to_extract', 
                                 required=True,
                                 metavar='what_to_extract', help="Choose one option",
                                 widget="Listbox", nargs='+', choices=['mean', 'Median'],
                                 default=stored_args.get('what_to_extract'))

    requirered_args.add_argument('--num_parallel_jobs',
                                 required=True,
                                 action='store',
                                 default=stored_args.get('num_parallel_jobs'),
                                 help="# jobs (bootstraps/ROIs) to run in parallel")
    


    parser.add_argument('--Base_Sample_name',
                        required=False,
                        action='store',
                        default=stored_args.get('Base_Sample_name'),
                        help="Base_Sample_name: leave empty")

    requirered_args.add_argument('--Image_top_DIR',
                                 required=True,
                                 action='store',
                                 widget='DirChooser',
                                 default=stored_args.get('Image_top_DIR'),
                                 help="Location of CAT processed .nii files")
    
    
    
    
    requirered_args.add_argument('--Smoothing_kernel_FWHM',
                                 required=True,
                                 action='store',
                                 default=stored_args.get('Smoothing_kernel_FWHM'),
                                 help="Smoothing_kernel_FWHM")
    
    
    
    
    requirered_args.add_argument('--Modulation_Method',
                                 required=True,
                                 metavar='Modulation_Method', help="Choose one option",
                                 widget="Listbox", nargs='+', choices=['fully_modulated', 'non_linearOnly'],
                                 default=stored_args.get('Modulation_Method'))
    
    
    requirered_args.add_argument('--Cog_list_for_profiling',
                                 required=True,
                                 action='store',
                                 default=stored_args.get('Cog_list_for_profiling'),
                                 widget='FileChooser',
                                 help='text file with list of Cognitive  \nscores names for Profiling')
    
    
#    parser.add_argument('correlation_method',
#                        action='store',
#                        default=stored_args.get('correlation_method'),
#                        help="correlation_method: 'pearson' \nor 'spearman' or 'sPartial' or 'linear_regression'")
    requirered_args.add_argument('--correlation_method', 
                                 required=True,
                                 metavar='correlation_method', help="Choose one option",
                                 widget="Listbox", nargs='+', choices=['pearson', 'spearman', 'sPartial', 'linear_regression'],
                                 default=stored_args.get('correlation_method'))

    requirered_args.add_argument('--alpha',
                                 required=True,
                                 action='store',
                                 default=stored_args.get('alpha'),
                                 help="alpha corersponding to : \nCI = 100*(1 -alpha*2)")
    
    requirered_args.add_argument('--n_boot',
                                 required=True,
                                 action='store',
                                 default=stored_args.get('n_boot'),
                                 help="number of bootsraps")
#
#    parser.add_argument('OUTPUT_type',
#                        action='store',
#                        default=stored_args.get('OUTPUT_type'),
#                        help="OUTPUT_type: Figure Or Csv")
#    
    requirered_args.add_argument('--OUTPUT_type',
                                 required=True,
                                 metavar='OUTPUT_type', help="Choose one option",
                                 widget="Listbox", nargs='+', choices=['Figure', 'Csv'],
                                 default=stored_args.get('OUTPUT_type'))

    requirered_args.add_argument('--percent_top_corr',
                                 required=True,
                                 action='store',
                                 default=stored_args.get('percent_top_corr'),
                                 help="% of top correlations of show: for all give 100")
   
    
    
    args = parser.parse_args()
    # Store the values of the arguments so we have them next time we run
    with open(args_file, 'w') as data_file:
        # Using vars(args) returns the data as a dictionary
        json.dump(vars(args), data_file)
    return args




if __name__ == '__main__':
    conf = parse_args()
    
    #******************************************************
    #       manual input for initial directory settings
    #******************************************************
    print("Base Directory to perfrom analysis in:")
    original_base_folder = conf.base_analysis_directory
    ###
    print("NAME of the directory")
    run = conf.analysis_directory_name
    specific_name_for_csv = run
    Full_working_directory = os.path.join(original_base_folder, run)
    try:
         os.makedirs(Full_working_directory)
    except OSError:
        if not os.path.isdir(Full_working_directory):
            raise   
    #******************************************************
    #       manual input for CSV generation settings
    #******************************************************
    print("Main_Sample_info_table_CSV_full_path")
    Main_Sample_info_table_CSV_full_path = conf.Main_Sample_info_table_CSV_full_path
    print("Other important variales --a column name from the csv, if 'D' then it is a temproary option set as default, otherwise leave empty")
    if conf.Other_important_variable == None:
        
        Other_important_variables = []
    else:
        Other_important_variables =[str(conf.Other_important_variable)]
    
#    try: 
#        Other_important_variables =[str(conf.Other_important_variable)]
#     
#    except: 
#        Other_important_variables = []
    print("A text file, in each line name of the columes that are going to be the Confounders to control for them")
    Confounders_list_full_path = conf.Confounders_txt_file
    with open(Confounders_list_full_path) as L:
        Confounders_names = L.read().splitlines()    
    Important_variables = Confounders_names + Other_important_variables
    
    print("Diagnosis --a column name from the csv, if do not have, leave empty")
    #Diagnosis_exclusion_criteria = ''
    if conf.Diagnosis_exclusion_criteria == ['none']:
        Diagnosis_exclusion_criteria=''
    else:
        Diagnosis_exclusion_criteria = conf.Diagnosis_exclusion_criteria[0]#'loose'
        

    #    try: 
    #        
    #        Diagnosis_exclusion_criteria =[str(conf.exclusion_criteria)]
    #    except: 
    #        Diagnosis_exclusion_criteria = ''
        ####
    print("Base directory where the subsampling script is: Mod_01_binning_shahrzad.py")
    subsampling_scripts_base_dir = conf.subsampling_scripts_base_dir
    
    print("age, sex column names")
    
    Sex_col_name = conf.gender_column_name
    Age_col_name = conf.AGE_column_name
    if conf.SITE == None:
        SITE_col_name=''
    else:
        SITE_col_name = conf.SITE
    #***************************************************************************
    #       manual input for ROI preparation and GMV extraction settings
    #***************************************************************************
    print('A text file with full path of ROIs')
    predefined_ROI_list_path = conf.ROI_full_path
    print('Where to save resampled ROIs')
    what_to_extract = str(conf.what_to_extract[0])
    n_jobs = int(conf.num_parallel_jobs)
    where_to_save_resampled_ROI = os.path.join(Full_working_directory, 'resampled_ROI')
    where_to_save_ROI_Table = os.path.join(Full_working_directory, 'ROI_table')
    ROI_table_name_suffix = 'ROI_GMV'
    #******************************************************
    #       manual input for NIFTI loading settings
    #******************************************************
    mod_method= conf.Modulation_Method[0]
    if  conf.Base_Sample_name == None:
        Base_Sample_name= '' 
        
    else:
        
        Base_Sample_name = conf.Base_Sample_name
        
    Image_top_DIR = conf.Image_top_DIR #run_masch + 'BnB2/Derivatives/CAT/12.5/ADNI_mixedScanners/'
    #Mask_file_complete = os.path.join(run_masch + 'BnB_USER/Shahrzad/eNKI_modular', 'Masks/binned_FZJ100_all_c1meanT1.nii.gz')
    Smoothing_kernel_FWHM = int(conf.Smoothing_kernel_FWHM)
    Mask_file_complete = ''
    load_nifti_masker_Flag = 0
    merged_image_name = '4D_file'
    #******************************************************
    #       manual input for Behavioral profiling settings
    #******************************************************
    stats_base_dir = os.path.join(Full_working_directory, 'resampled_ROI')
    print('A text file with column names of the csv file, corresponding to cognitive scores')
    Cog_list_full_path_for_profiling = str(conf.Cog_list_for_profiling)
    print(Cog_list_full_path_for_profiling)
    Group_selection_column_name = 'Which_sample'
    Group_division = True
    Group_selection_Label = ['all', '1', '2']
    print(Group_selection_Label)
    correlation_method = conf.correlation_method[0]#'sPartial'
    print(correlation_method)
    Sort_correlations = True
    alpha = float(conf.alpha)# # CIs: 100-alpha*2 
    print(alpha)
    n_boot = int(conf.n_boot)#100
    
    #******************************************************
    #       manual input for output generation settings
    #******************************************************
    Saving_output_dir = os.path.join(Full_working_directory, 'OUtput')
    OUTPUT_type = conf.OUTPUT_type[0]#'Figure' # Or 'Csv'
    master_group = 'all'
    percent_top_corr_plot = float(conf.percent_top_corr)#100
    plot_type = 'errordots'
    #%%  settings
    
    
    print(Smoothing_kernel_FWHM)
    
    
    # Block 1: Parse inputs
    Initial_directory_settings = {'Base_saving_dir' : original_base_folder, 'workdir_name' : run}    
    
    CSV_generation_settings = {'Main_Sample_info_table_CSV_full_path': Main_Sample_info_table_CSV_full_path, 'Important_variables' : Important_variables, \
                               'exclusion_criteria':Diagnosis_exclusion_criteria, 'specific_name_for_csv':specific_name_for_csv}
    
    NIFTI_loading_Setting = {'Image_top_DIR': Image_top_DIR, 'Mask_file_complete': Mask_file_complete, 'merged_image_name' : merged_image_name, \
                             'Base_Sample_name':Base_Sample_name,'mod_method' : mod_method,'Smoothing_kernel_FWHM':Smoothing_kernel_FWHM, 'load_nifti_masker_Flag': load_nifti_masker_Flag}
    
    #Split_settings = {'subsampling_scripts_base_dir': subsampling_scripts_base_dir, 'test_sample_size' : 0.5, 'Age_step_size': 10, 'gender_selection' : None,\
    #                  'n_split' : 1, 'Sex_col_name': 'sex_mr', 'Age_col_name' : 'Age_current'}
    if SITE_col_name=='':
        
        Split_settings = {'subsampling_scripts_base_dir': subsampling_scripts_base_dir, 'test_sample_size' : 0.5, 'Age_step_size': 10, 'gender_selection' : None, \
                          'n_split' : 1, 'Sex_col_name': Sex_col_name, 'Age_col_name' : Age_col_name, 'Confounders_list_full_path': Confounders_list_full_path}
    else:
        Split_settings = {'subsampling_scripts_base_dir': subsampling_scripts_base_dir, 'test_sample_size' : 0.5, 'n_bins_age': 2, 'gender_selection' : None, \
                  'n_split' : 1, 'Sex_col_name': Sex_col_name, 'Age_col_name' : Age_col_name, 'SITE_col_name': SITE_col_name, \
                  'add_dummy_site_to_confounders' : 1, 'Confounders_list_full_path': Confounders_list_full_path}
    

                
    
    ROI_preparation_settings = {'predefined_ROI_list_path': predefined_ROI_list_path, 'where_to_save_resampled_ROI' : where_to_save_resampled_ROI,\
                                'where_to_save_ROI_Table':where_to_save_ROI_Table, 'ROI_table_name_suffix':ROI_table_name_suffix, 'what_to_extract':what_to_extract, 'n_jobs':n_jobs}
    
    
    Behavioral_profiling_settings = {'stats_base_dir' : stats_base_dir, 'Cog_list_full_path_for_profiling':Cog_list_full_path_for_profiling, 'Confounders_list_full_path':Confounders_list_full_path,\
                                     'Group_selection_column_name':Group_selection_column_name, 'Group_division' : Group_division, 'Group_selection_Label':Group_selection_Label,\
                                     'Sort_correlations':Sort_correlations,'correlation_method':correlation_method, 'alpha':alpha, 'n_boot':n_boot, 'n_jobs':n_jobs}
    
    
    Output_generation_settings = {'Saving_output_dir':Saving_output_dir, 'OUTPUT_type': OUTPUT_type, 'master_group':master_group,\
                                  'percent_top_corr_plot' : percent_top_corr_plot, 'plot_type':plot_type}
    
    

    print(Initial_directory_settings['Base_saving_dir'])
    print(Initial_directory_settings['workdir_name'])
    #%%
    
    
    Input_preparation_class = Sample_handling(Initial_directory_settings['Base_saving_dir'],Initial_directory_settings['workdir_name'], CSV_generation_settings['specific_name_for_csv'])
    
    
    CSV_main_Path, _ = Input_preparation_class.main_CSV_generation(CSV_generation_settings['Main_Sample_info_table_CSV_full_path'],\
                                                                   CSV_generation_settings['Important_variables'], CSV_generation_settings['exclusion_criteria'])
    print(SITE_col_name)
    if SITE_col_name=='':
        
        CSV_grouped_Path, new_Confounders_list_full_path = Input_preparation_class.Split_column_generation(subsampling_scripts_base_dir = Split_settings['subsampling_scripts_base_dir'], main_csv_path= CSV_main_Path,\
                                                                                                           Sex_col_name=Split_settings['Sex_col_name'],Age_col_name= Split_settings['Age_col_name'],\
                                                                                                           Confounders_list_full_path = Split_settings['Confounders_list_full_path'], Age_step_size = Split_settings['Age_step_size'],\
                                                                                                           test_sample_size = Split_settings['test_sample_size'],gender_selection = Split_settings['gender_selection'])
    else:
        CSV_grouped_Path, new_Confounders_list_full_path = Input_preparation_class.Split_column_generation(subsampling_scripts_base_dir = Split_settings['subsampling_scripts_base_dir'], main_csv_path= CSV_main_Path,\
                                                                                                           Sex_col_name=Split_settings['Sex_col_name'],Age_col_name= Split_settings['Age_col_name'],\
                                                                                                           SITE_col_name = Split_settings['SITE_col_name'],n_bins_age = Split_settings['n_bins_age'],\
                                                                                                           add_dummy_site_to_confounders = Split_settings['add_dummy_site_to_confounders'],Confounders_list_full_path =Split_settings['Confounders_list_full_path'],\
                                                                                                           test_sample_size = Split_settings['test_sample_size'],gender_selection = Split_settings['gender_selection'])

    #%%
    Imaging_Input_preparation_class = Imaging_data_handling(Initial_directory_settings['Base_saving_dir'],Initial_directory_settings['workdir_name'],\
                                                  CSV_grouped_Path, NIFTI_loading_Setting['Mask_file_complete'])
    
    
    merged_file_path = Imaging_Input_preparation_class.fourD_file_generation_from_table(NIFTI_loading_Setting['Image_top_DIR'], NIFTI_loading_Setting['merged_image_name'],\
                                                                                        NIFTI_loading_Setting['Base_Sample_name'], NIFTI_loading_Setting['mod_method'],NIFTI_loading_Setting['Smoothing_kernel_FWHM'],\
                                                                                        NIFTI_loading_Setting['load_nifti_masker_Flag'])
    #merged_file_path = '/data/BnB_USER/Shahrzad/eNKI_modular/ADNI_ROI_profiling/test_GUI5/4D_images/4D_file.nii.gz'
    nii_ROI_file = Imaging_Input_preparation_class.make_ROI_ready(ROI_preparation_settings['predefined_ROI_list_path'], merged_file_path, ROI_preparation_settings['where_to_save_resampled_ROI']) 
    
    
    GMV_CSV_path, ROI_base_names_path = Imaging_Input_preparation_class.CSV_with_ROI_Volume(nii_ROI_file,  merged_file_path ,ROI_preparation_settings['where_to_save_ROI_Table'],\
                                                                                            ROI_preparation_settings['ROI_table_name_suffix'] , ROI_preparation_settings['what_to_extract'], ROI_preparation_settings['n_jobs'])
    
    
    #%%
    Behavioral_profiling_settings = {'stats_base_dir' : stats_base_dir, 'Cog_list_full_path_for_profiling':Cog_list_full_path_for_profiling, 'Confounders_list_full_path':new_Confounders_list_full_path,\
                                     'Group_selection_column_name':Group_selection_column_name, 'Group_division' : Group_division, 'Group_selection_Label':Group_selection_Label,\
                                     'Sort_correlations':Sort_correlations,'correlation_method':correlation_method, 'alpha':alpha, 'n_boot':n_boot, 'n_jobs':n_jobs}
    
    Behavioral_profiling_class = Behavioral_Profiling(Initial_directory_settings['Base_saving_dir'],Initial_directory_settings['workdir_name'],\
                                                      GMV_CSV_path, Behavioral_profiling_settings['Cog_list_full_path_for_profiling'],\
                                                      Behavioral_profiling_settings['Confounders_list_full_path'], Behavioral_profiling_settings['Group_selection_column_name'],\
                                                      Behavioral_profiling_settings['Group_division'],\
                                                      Behavioral_profiling_settings['Sort_correlations'], Behavioral_profiling_settings['correlation_method'],\
                                                      Behavioral_profiling_settings['alpha'], Behavioral_profiling_settings['n_boot'])
    
    with open(ROI_base_names_path) as f:
        ROIS = f.read().splitlines()
    for roi_id in np.arange(len(ROIS)):
        ROI_name = ROIS[roi_id]
        if Behavioral_profiling_settings['stats_base_dir'] == '':
            
            stats_dir = os.path.join(os.path.dirname(ROI_base_names_path), ROI_name)
        else:
            stats_dir = os.path.join(Behavioral_profiling_settings['stats_base_dir'], ROI_name)
            
        try:
            os.makedirs(stats_dir)
        except OSError:
            if not os.path.isdir(stats_dir):
                raise
        groupped_mean = []
        groupped_err = []
        groupped_av = []
        groupped_P_val = []
        groupped_CIs = []
        for g in Behavioral_profiling_settings['Group_selection_Label']:
           
            # Here you can create an if statement to change the 
            mean, err, av, _, P_val, _, CIs = Behavioral_profiling_class.ROI_Behavioral_profiling(ROI_name, stats_dir, g, Behavioral_profiling_settings['n_jobs'])
            ###
            groupped_mean.append(mean)
            groupped_err.append(err)
            groupped_av.append(av)
            groupped_P_val.append(P_val)
            groupped_CIs.append(CIs)
            
        
        if Output_generation_settings['OUTPUT_type'] == 'Figure':
            
            name = 'Behavioral_profile_'+ ROI_name  + '_' + Behavioral_profiling_settings['correlation_method']
            
            PDF_File_path = Create_multiple_sample_Functional_profile_PDFs(Output_generation_settings['Saving_output_dir'],\
                                                                           Group_selection_Label = Behavioral_profiling_settings['Group_selection_Label'],mean_r_DF_Full_path = groupped_mean,\
                                                                           CI_err_DF_Full_path = groupped_err, available_n_full_path = groupped_av,\
                                                                           Page_Plot_name = name,name_of_pdf = name,n_rows_one_page = 1,\
                                                                           mastre_group = Output_generation_settings['master_group'],\
                                                                           percent_top_corr_plot = Output_generation_settings['percent_top_corr_plot'],\
                                                                           plot_type = Output_generation_settings['plot_type'])
       
        else:
            try:
                os.makedirs(Output_generation_settings['Saving_output_dir'])
            except OSError:
                if not os.path.isdir(Output_generation_settings['Saving_output_dir']):
                    raise
                    
                    
            frames_m = [pd.read_pickle(f) for f in groupped_mean]
            means_df = pd.concat(frames_m, keys=np.arange(len(groupped_mean)))
            means_df.columns = [str(col) + '_effectsize' for col in means_df.columns]
            frames_p = [pd.read_pickle(f) for f in groupped_P_val]
            P_val_df = pd.concat(frames_p, keys=np.arange(len(groupped_P_val)))
            P_val_df.columns = [str(col) + '_Pvalue' for col in P_val_df.columns]
            frames_CI = [pd.read_pickle(f) for f in groupped_CIs]
            CIs_df = pd.concat(frames_CI, keys=np.arange(len(groupped_CIs)))
            CIs_df.columns = [str(col) + '_CI' for col in CIs_df.columns]
            frames_av = [pd.read_pickle(f) for f in groupped_av]
            av_df = pd.concat(frames_av, keys=np.arange(len(groupped_av)))
            av_df.columns = [str(col) + '_n_available' for col in av_df.columns]
            merged_Behavioral_profile_info_file = os.path.join(Output_generation_settings['Saving_output_dir'], ROI_name +'_merged_profile_info.csv')
            output = pd.concat([means_df,P_val_df, CIs_df,av_df], axis=1)
            output.to_csv(merged_Behavioral_profile_info_file)
    
            
    print("Done")
         
          
        
        
        
        


##### 











#
#        frames_m = [pd.read_pickle(f) for f in mean_r_DF_Full_path]
#        means_df = pd.concat(frames_m, keys=np.arange(len(mean_r_DF_Full_path)))
#        frames_e = [pd.read_pickle(f) for f in CI_err_DF_Full_path]
#        err_df = pd.concat(frames_e, keys=np.arange(len(CI_err_DF_Full_path)))
#        if ranking_x_axis == True:
#            # Which sampel is our master sample, defining the order of the variables in X-axis?
#            if percent_top_corr_plot > 1:
#                percent_top_corr_plot = np.divide(percent_top_corr_plot, 100)
#    
#            mastre_group_id = mastre_group_id
#            n_top_correlations = np.ceil(percent_top_corr_plot*len(means_df.loc[mastre_group_id].index.values))
#        
#            Cog_Tests_for_profiling = means_df.loc[mastre_group_id].index.values[:int(n_top_correlations)]
#        else:
#            mastre_group_id = 0
#            Cog_Tests_for_profiling = means_df.loc[0].index.values
#                
#


























        
       










                                





















