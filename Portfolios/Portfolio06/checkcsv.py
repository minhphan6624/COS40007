import pandas as pd

# Load the generated IoU CSV
iou_csv_path = 'outputCSV\iter-no.csv'  # Path to your CSV
df = pd.read_csv(iou_csv_path)

# Group by image name and get the maximum IoU for each image
grouped_df = df.groupby('image_name').agg({'IoU_value': 'max'}).reset_index()

# Filter images with IoU ≥ 0.9
high_iou_images = grouped_df[grouped_df['IoU_value'] >= 0.9]

# Calculate the percentage of images with IoU ≥ 0.9
total_images = len(grouped_df)
high_iou_count = len(high_iou_images)
percentage_high_iou = (high_iou_count / total_images) * 100

# Check if 80% or more images have IoU ≥ 0.9
if percentage_high_iou >= 80:
    print(f"Success! {percentage_high_iou:.2f}% of images have IoU ≥ 0.9. No need to retrain.")
else:
    print(f"Only {percentage_high_iou:.2f}% of images have IoU ≥ 0.9. You should retrain the model.")
