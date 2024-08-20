# DLBL_Project

Generate the thumbnails of your images using the file wsi_to_thumbnail.py to have an easy visualization of images

Given below is the process to be followed to get the desired results :
1. First put all the slides in a folder and go inside the CLAM folder
2. Run the file dataset_csv_create.py and give the path of the dataset in the file
3. Now update the labels of the slides in the created csv file
4. Refer the readme in CLAM folder to understand the below commands in detail
5. Run the file create_patches_fp.py and give the required path for slides and specify a directory where you want to store the results (Refer
6. Run the file read_h5.py to create the csv files of the coordinates from the extracted patches, specify the path of patches folder obtained from the previous code along with the desired folder you want to save results in.
7. Now run the white_patch_filter_mean_std.py file to perform the white patch filtering to remove the unncessary patches which are completely white or completely dark to go into the model as it can confuse the model and effect it's predictions. Specify the paths in the file as mentioned in comments of that file. We have used 220 and 10 as thresholds along with a standard deviation of 10, this can be changed as per requirement and can be treated as hyperparameter.
8. Run the file create_clam_h5.py to create the h5 files again, but this time we will get the updated coordinates after white patch filtering, specify the paths carefully in this file.
9. Now run the extract_features_fp.py file, see the readme file in CLAM folder to understand better about this command and how to use it.
10. Create a Data Root Directory folder as mentioned in the readme file in CLAM folder
11. Now create the splits for train, val and test. Use the create_splits_seq.py for this purpose, add or change the task in the final according to your requirement and the number of classes.
12. Now you can run the main.py and eval.py, look at the readme file of CLAM folder to understand in detail about the hyperparameters and the use of these files
13. To verify that the model is focussing on the exact region of interest (tumor), one can observe the heatmaps for any WSI image by running the create_heatmaps.py
14. Modify the path to the pt files folder and h5 files folder in this file
15. Now create a .yaml file with the required specifications and paths (refer the clam_8.yaml and modify it as per your requirement)
16. Now create a process list which contains the list of slides along with their labels for which you want to generate the heatmap
17. Also create a csv preset file and modify the parameters as per your model (refer task_3_cancer_test_preset.csv)

Further steps are for doing the morphological analysis of the heatmaps and the high attention regions :

18. Run the "extarct_patches_from_h5.py" after giving the path of heatmap output to generate the csv file which has the coordinates along with the attention values for each patch
19. Now run the "create_high_attention_patches_from_csv.py" and give the required paths in this file, now you can specify the number of high attention patches that you want for further analysis
20. All further analysis are done in a notebook file "Patch_Analysis.ipynb", the notebook is self sufficient in explaining

Thereby you can conclude all the results of this project.
