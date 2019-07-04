# pyJUSBB
This is the repository of the GUI-based tool for structure-brain-behavior analysis. 


# Requirements:
python 3.5
simplejson==3.11.1
Gooey==1.0.2
numpy==1.13.1
pandas==0.20.3
matplotlib==2.1.0
nibabel==2.1.0
nilearn==0.4.1
nipype==0.13.1
scikit-learn==0.19.0
scikits.bootstrap==0.3.3
scipy==1.1.0
seaborn==0.8.1
joblib==0.11

(Alternatively you can install the requirements.txt file. This however might install some additional packegs that are not used within the scope of the SBB project). 

* In addition, the program is using fsl and AFNI to create merged 4D images and resample the regions_of_interests. So, it is required to have these neuroimaging packages also in the path. 

