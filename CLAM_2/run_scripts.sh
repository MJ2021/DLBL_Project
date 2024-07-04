scripts=("white_patch_filter_mean_std.py" "create_clam_h5.py" 
"extract_features_fp.py --data_h5_dir /home/Drivessd2tb/Mohit_Combined/h5_files_with_white_filter_from_patches_combined_200 --data_slide_dir /home/Drivessd2tb/dlbl_combined --csv_path /home/ravi/Mohit/CLAM/csv_files/dlbl_data_combined.csv --feat_dir /home/Drivessd2tb/Mohit_Combined/Features_200 --batch_size 512 --slide_ext .tiff")

for script in "${scripts[@]}"
do
    echo "Running $script..."
    python "$script"
    if [ $? -ne 0 ]; then
        echo "Error: $script failed to execute."
        exit 1
    fi
done

echo "All scripts executed successfully."