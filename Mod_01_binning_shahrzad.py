import os
import pandas as pd
import numpy as np
#from matplotlib.pylab import hist
#%matplotlib inline

# reading data in:
def importing_table_from_excel(Table_DIR, File_nam_with_extension, Sheetname, header):
    os.chdir(Table_DIR) #os.chdir('/Users/shahrzad/sciebo/Jülich_analysis_F1000/Fz100/')
    Original_data_table=pd.read_excel(File_nam_with_extension, sheetname=Sheetname, header=header) #All_data_Fz100=pd.read_excel('FZJ100.xlsx')
    Original_Table=Original_data_table.reset_index()
    return Original_Table


# creating bins of age:

def binning_age(Original_data_table, min_age, max_age,  Age, step_size = 0, n_bins_age = 0):
    
    if n_bins_age == 0:
        
        age_bin = np.arange(min_age,max_age +1,step_size).astype(int).tolist()
        catrgory_num=np.arange(len(age_bin)-1)+1 # We want categories to be one less than the length of age_bin list, but we add 1 to start from "1" for the youngest category.
        #Original_data_table['age_group_num'] = pd.cut(Original_data_table[Age], age_bin, labels=catrgory_num)
        Original_data_table['age_group_num'] = pd.cut(Original_data_table[Age], len(catrgory_num), labels=catrgory_num)
    else: 
        Original_data_table['age_group_num'] = pd.cut(Original_data_table[Age], bins = n_bins_age, labels=False)
        catrgory_num = np.arange(n_bins_age)
        
    
    return Original_data_table, catrgory_num


# creating bins of VarX:

def binning_VarX(Original_data_table, VarX, n_bins_VarX = 0):
    
    new_varX = VarX + '_group_num'
    
    Original_data_table[new_varX] = pd.cut(Original_data_table[VarX], bins = n_bins_VarX, labels=False)
    catrgory_num_varX = np.arange(n_bins_VarX)
        
    return Original_data_table, catrgory_num_varX

#def age_sex_VarX_matched_splitting(Original_data_table, catrgory_num_age, Age, sex, VarX, catrgory_num_VarX):
##    from sklearn.model_selection import train_test_split
##    X = Original_data_table.index.values.tolist()
##    Y = np.array(Original_data_table[VarX + '_group_num'])
##    X_train, X_test, y_train, y_test = train_test_split(X, Y,
##                                                        stratify=Y, 
##                                                        test_size=0.5)

#    for i in catrgory_num_VarX

# Now we sort the table, based on age,
# then split it into women and men and
# then craete age matched goups for females (2 groups) and males (2 groups)
def age_matched_splitting(Original_data_table, catrgory_num, Age, sex):
    Original_data_table=Original_data_table.sort_values(by=[Age], ascending=[True])
    female = Original_data_table[Original_data_table[sex]==-1].copy()
    male = Original_data_table[Original_data_table[sex]==1].copy()
    gr_1_male=pd.DataFrame()
    gr_1_female=pd.DataFrame()
    gr_2_male=pd.DataFrame()
    gr_2_female=pd.DataFrame()
    which_bin =0


    count=0
    for i in np.arange(len(catrgory_num)).tolist():
        which_bin=i+1
        categ_male=pd.DataFrame()
        categ_female=pd.DataFrame()
        categ_male=male[male.age_group_num==which_bin].copy()
        categ_female=female[female.age_group_num==which_bin].copy()

        print("%d female" % len(categ_female))
        print("%d male" % len(categ_male))
        #print(i)

        if (len(categ_male)/2>0)and(len(categ_female)/2>0):
            n=0

            n_from_end=min(len(categ_male)/2, len(categ_female)/2)
            n=int(n_from_end)
            random_categ_female=categ_female.sample(2*n).copy()
            random_categ_male=categ_male.sample(2*n).copy()
            gr_1_female= gr_1_female.append(random_categ_female.iloc[0:n])
            #categ_female.iloc[0:n]
            gr_1_male=gr_1_male.append(random_categ_male.iloc[0:n])

            gr_2_female= gr_2_female.append(random_categ_female.iloc[-int(n):])
            gr_2_male= gr_2_male.append(random_categ_male.iloc[-int(n):])
            count=count+n
            #return gr_1_female, gr_2_female, gr_1_male, gr_2_male
        elif (len(categ_male)/2>0) and (len(categ_female)/2<1):
            n=0
            n_from_end=len(categ_male)/2
            n=int(n_from_end)
            #random_categ_female=categ_female.sample(2*n).copy()
            random_categ_male=categ_male.sample(2*n).copy()
            #gr_1_female= []
            #categ_female.iloc[0:n]
            gr_1_male=gr_1_male.append(random_categ_male.iloc[0:n])

            #gr_2_female= []
            gr_2_male= gr_2_male.append(random_categ_male.iloc[-int(n):])
            count=count+n
        elif (len(categ_male)/2<1) and (len(categ_female)/2>0):
            n=0

            n_from_end=len(categ_female)/2
            n=int(n_from_end)
            random_categ_female=categ_female.sample(2*n).copy()
            #random_categ_male=categ_male.sample(2*n).copy()
            gr_1_female= gr_1_female.append(random_categ_female.iloc[0:n])
            #categ_female.iloc[0:n]
            #gr_1_male=[]
            #gr_1_male=gr_1_male.append(random_categ_male.iloc[0:n])

            gr_2_female= gr_2_female.append(random_categ_female.iloc[-int(n):])
            #gr_2_male=[]
            #gr_2_male= gr_2_male.append(random_categ_male.iloc[-int(n):])
            count=count+n     
    return gr_1_female, gr_2_female, gr_1_male, gr_2_male

def create_ageAndsex_matched_sugroups_from_DF(DF,Age, sex,step_size=5):

    Original_data_table = DF
    min_age = Original_data_table[Age].min()
    max_age = Original_data_table[Age].max()

    Original_data_table, catrgory_num = binning_age(Original_data_table, min_age, max_age,  Age, step_size)
    gr_1_female, gr_2_female, gr_1_male, gr_2_male = age_matched_splitting(Original_data_table, catrgory_num, Age, sex)
    First_group=gr_1_female.append(gr_1_male)
    Second_group=gr_2_female.append(gr_2_male)
    return First_group , Second_group


def create_ageAndsex_matched_sugroups_from_excel(DIR, File_nam_with_extension, Sheetname, header, Age, sex, step_size=5):

    Original_data_table = importing_table_from_excel(DIR, File_nam_with_extension, Sheetname, header)
    min_age = Original_data_table[Age].min()
    max_age = Original_data_table[Age].max()

    Original_data_table, catrgory_num = binning_age(Original_data_table, min_age, max_age, Age, step_size)
    gr_1_female, gr_2_female, gr_1_male, gr_2_male = age_matched_splitting(Original_data_table, catrgory_num, Age, sex)
    First_group=gr_1_female.append(gr_1_male)
    Second_group=gr_2_female.append(gr_2_male)
    return First_group , Second_group



def Create_ageAndsexAndVarx_matched_sugroups_from_DF(DF,Age, sex, VarX = '', step_size_age=0, n_bins_age=0, n_bins_VarX=0):
    
    Original_data_table = DF
    if step_size_age != 0 and n_bins_VarX == 0:
        
        min_age = Original_data_table[Age].min()
        max_age = Original_data_table[Age].max()
    
        Original_data_table, catrgory_num = binning_age(Original_data_table, min_age, max_age,  Age, step_size_age)
        gr_1_female, gr_2_female, gr_1_male, gr_2_male = age_matched_splitting(Original_data_table, catrgory_num, Age, sex)
        First_group=gr_1_female.append(gr_1_male)
        Second_group=gr_2_female.append(gr_2_male)
        
    elif n_bins_age !=0 and n_bins_VarX==0:
        
        min_age = Original_data_table[Age].min()
        max_age = Original_data_table[Age].max()
    
        Original_data_table, catrgory_num = binning_age(Original_data_table, min_age, max_age,  Age, n_bins_age)
        gr_1_female, gr_2_female, gr_1_male, gr_2_male = age_matched_splitting(Original_data_table, catrgory_num, Age, sex)
        First_group=gr_1_female.append(gr_1_male)
        Second_group=gr_2_female.append(gr_2_male)
    elif step_size_age != 0 and n_bins_VarX != 0:
        print('HO')
        min_age = Original_data_table[Age].min()
        max_age = Original_data_table[Age].max()
    
        Original_data_table, catrgory_num_age = binning_age(Original_data_table, min_age, max_age,  Age, step_size_age)
        Original_data_table, catrgory_num_VarX = binning_VarX(Original_data_table, VarX, n_bins_VarX)
        First_group = pd.DataFrame()
        Second_group = pd.DataFrame()
        
        for i in catrgory_num_VarX:
            
            Original_data_table_cat_VarX = Original_data_table[Original_data_table[VarX + '_group_num'] == i]
            gr_1_female, gr_2_female, gr_1_male, gr_2_male = age_matched_splitting(Original_data_table_cat_VarX, catrgory_num_age, Age, sex)
            minor_First_group=gr_1_female.append(gr_1_male)
            minor_Second_group=gr_2_female.append(gr_2_male)
            First_group = First_group.append(minor_First_group)
            Second_group = Second_group.append(minor_Second_group)
    elif n_bins_age !=0 and  n_bins_VarX != 0:
        print('Hi')
        min_age = Original_data_table[Age].min()
        max_age = Original_data_table[Age].max()
    
        Original_data_table, catrgory_num_age = binning_age(Original_data_table, min_age, max_age,  Age, n_bins_age)
        Original_data_table, catrgory_num_VarX = binning_VarX(Original_data_table, VarX, n_bins_VarX)
        First_group = pd.DataFrame()
        Second_group = pd.DataFrame()
        
        for i in catrgory_num_VarX:
            Original_data_table_cat_VarX = Original_data_table[Original_data_table[VarX + '_group_num'] == i]
            
            gr_1_female, gr_2_female, gr_1_male, gr_2_male = age_matched_splitting(Original_data_table_cat_VarX, catrgory_num_age, Age, sex)
            minor_First_group=gr_1_female.append(gr_1_male)
            minor_Second_group=gr_2_female.append(gr_2_male)
            First_group = First_group.append(minor_First_group)
            Second_group = Second_group.append(minor_Second_group)
    
        
        
    return First_group , Second_group









if __name__ == "__main__":
    step_size =5
    Directory = '/media/sf_shahrzad_mac/sciebo/Jülich_analysis_F1000/Fz100/data_test_fz100_list_CSVs/'
    File_name = "FZJ100.xlsx"
    header = 0
    Sheetname = None
    Age = 'Age'
    sex ='sex'
    First_group , Second_group = create_ageAndsex_matched_sugroups_from_excel(Directory, File_name, Sheetname , header ,Age, sex, step_size)
    os.chdir("/Users/shahrzad/sciebo/Jülich_analysis_F1000/Fz100/data_test_fz100_list_CSVs/")
    First_group.to_csv('firs_group.csv', sep=',')
    Second_group.to_csv('second_group.csv', sep=',')
    np.save('firs_group.npy', First_group)
    np.save('second_group.npy', Second_group)
