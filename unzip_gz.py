import os
import gzip
import shutil

def unzip_gz_files(dir_path):
    """
    Unzips all `.gz` files in a specified directory.
    
    Parameters:
        dir_path (str): The path to the directory containing `.gz` files.
    
    Returns:
        None
    """
    # Get a list of all .gz files in the specified directory
    gz_files = [f for f in os.listdir(dir_path) if f.endswith('.gz')]
    
    # Iterate over each .gz file in the directory
    for gz_file in gz_files:
        gz_file_path = os.path.join(dir_path, gz_file)
        output_file_path = os.path.join(dir_path, os.path.splitext(gz_file)[0])
        
        # Unzip the .gz file
        with gzip.open(gz_file_path, 'rb') as f_in:
            with open(output_file_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Optional: Delete the original .gz file
        # os.remove(gz_file_path)


root_dir = 'data'
subject_dirs = [os.path.join(root_dir, subj) for subj in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, subj))]
for subject_dir in subject_dirs:
    subject_id = os.path.basename(subject_dir)
    print(f"Processing subject: {subject_id}")
    unzip_gz_files(subject_dir)