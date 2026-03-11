import os
import numpy as np
import pandas as pd

# Function to compute the average across streamlines for each point (12 points per streamline)
def compute_avg_per_point(df, metric):
        if metric not in df.columns:
            print(f"Warning: {metric} not found in dataframe.")
            return [np.nan] * 12  # Return NaNs if metric is missing

        grouped = df.groupby(["Bundle", "Point"])[metric].mean().reset_index()
        return grouped

# Load the lateralization and deaf side data
df = pd.read_csv("T3_LQ_hemi.csv")
deaf_side_df = pd.read_csv("deaf_side.csv")

# Define the root directory containing subject data
root_dir = 'data' 
subject_dirs = [os.path.join(root_dir, subj) for subj in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, subj))]

values = [] # Store computed values

# Loop through each subject directory
for subject_dir in subject_dirs:
    subject_id = os.path.basename(subject_dir)
    print(f"Processing subject: {subject_id}")
   
    # Define file paths for left and right hemisphere diffusion metrics
    left_metrics_path = os.path.join(subject_dir, f"{subject_id}_diffusion_metrics_all_points_left.csv")
    right_metrics_path = os.path.join(subject_dir, f"{subject_id}_diffusion_metrics_all_points_right.csv")
   
    # Load diffusion metrics
    left_metrics_df = pd.read_csv(left_metrics_path)
    right_metrics_df = pd.read_csv(right_metrics_path)
    
    # Extract data
    T3_value = df.loc[df['Subject'] == subject_id, 'T3_LQ_hemi']
    deaf_side = deaf_side_df.loc[deaf_side_df['subject'] == subject_id, 'deaf_side']
    age = deaf_side_df.loc[deaf_side_df['subject'] == subject_id, 'age']

    # Determine lateralization category
    if not T3_value.empty:
       T3_value = T3_value.iloc[0]  # Extract the single value
       if T3_value > 40:
        lateralization_value = "left" # Left lateralization
        #print("Lateralization: Left")
       elif T3_value < -40:
        lateralization_value = "right" # Right lateralization
        #print("Lateralization: Right")
       else:
        lateralization_value = "both" # Both hemispheres
        #print("Lateralization: Both")
    else:
      lateralization_value = -1 # Subject ID not found
      print(f"Warning: Subject ID {subject_id} not found in lateralization data.")

    # Extract deaf side and age if available
    if not deaf_side.empty:
        deaf_side = deaf_side.iloc[0]
    else:
        deaf_side = None
    
    if not age.empty:
        age = age.iloc[0]
    else:
        age = None
    
    # Compute average diffusion metrics for left hemisphere
    left_ad = compute_avg_per_point(left_metrics_df, "AD")
    left_fa = compute_avg_per_point(left_metrics_df, "FA")
    left_md = compute_avg_per_point(left_metrics_df, "MD")
    left_rd = compute_avg_per_point(left_metrics_df, "RD")
    
    # Compute average diffusion metrics for right hemisphere
    right_ad = compute_avg_per_point(right_metrics_df, "AD")
    right_fa = compute_avg_per_point(right_metrics_df, "FA")
    right_md = compute_avg_per_point(right_metrics_df, "MD")
    right_rd = compute_avg_per_point(right_metrics_df, "RD")
    
    # Merge all metrics into a single DataFrame for left and right hemispheres
    left_data = left_ad.merge(left_fa, on=["Bundle", "Point"]).merge(left_md, on=["Bundle", "Point"]).merge(left_rd, on=["Bundle", "Point"])
    left_data.columns = ["Bundle", "Point", "Left_AD", "Left_FA", "Left_MD", "Left_RD"]

    right_data = right_ad.merge(right_fa, on=["Bundle", "Point"]).merge(right_md, on=["Bundle", "Point"]).merge(right_rd, on=["Bundle", "Point"])
    right_data.columns = ["Bundle", "Point", "Right_AD", "Right_FA", "Right_MD", "Right_RD"]

    # Merge left and right hemisphere data on (Bundle, Point)
    merged_data = left_data.merge(right_data, on=["Bundle", "Point"], how="outer")

    # Store results for each (bundle, point)
    for _, row in merged_data.iterrows():
        values.append([
            subject_id, row["Bundle"], row["Point"], 
            row["Left_AD"], row["Left_FA"], row["Left_MD"], row["Left_RD"],  
            row["Right_AD"], row["Right_FA"], row["Right_MD"], row["Right_RD"],
            lateralization_value, deaf_side, age
        ])
'''
# Convert collected values into a DataFrame
values_df = pd.DataFrame(values, columns=["Subject","Bundle","Point", "Left_AD", "Left_FA", "Left_MD", "Left_RD", "Right_AD", "Right_FA", "Right_MD", "Right_RD", "Lateralization", "Deaf_Side", "Age"])
#print("Right Data Sample:\n", values_df.head())

# Save results to CSV file
csv_path =  f"final.csv"
values_df.to_csv(csv_path, index=False, float_format='%.6f')
print(f"Saved  for all points to {csv_path}")
'''