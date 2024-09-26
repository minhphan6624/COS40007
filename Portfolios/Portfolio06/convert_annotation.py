import pandas as pd
import os

# Load the CSV file with annotations
annotations_df = pd.read_csv('data/bounding_boxes/train_labels.csv')

# Function to convert annotations to YOLO format


def convert_to_yolo_format(row):
    x_center = (row['xmin'] + row['xmax']) / 2 / row['width']
    y_center = (row['ymin'] + row['ymax']) / 2 / row['height']
    box_width = (row['xmax'] - row['xmin']) / row['width']
    box_height = (row['ymax'] - row['ymin']) / row['height']

    # Class label (0 for graffiti, adjust for other classes if needed)
    class_id = 0
    return [class_id, x_center, y_center, box_width, box_height]


# Apply the conversion to each row
yolo_annotations = annotations_df.apply(convert_to_yolo_format, axis=1)

# Create DataFrame for YOLO format
yolo_df = pd.DataFrame(yolo_annotations.tolist(), columns=[
                       'class_id', 'x_center', 'y_center', 'box_width', 'box_height'])
yolo_df['filename'] = annotations_df['filename']

# Create output directory for YOLO annotations
output_dir = 'data/yolo_annotations_txt'
os.makedirs(output_dir, exist_ok=True)

# Group by filename and save each image's annotations as a separate .txt file
grouped = yolo_df.groupby('filename')
for filename, group in grouped:
    # Handle both .jpg and .JPG extensions by normalizing to lowercase
    txt_filename = os.path.join(
        output_dir, filename.lower().replace('.jpg', '.txt'))
    group[['class_id', 'x_center', 'y_center', 'box_width', 'box_height']].to_csv(
        txt_filename, sep=' ', header=False, index=False)


# import pandas as pd

# file_path = 'data/bounding_boxes/train_labels.csv'

# # Load the data
# data = pd.read_csv(file_path)

# def convert_to_yolo(data):
#     yolo_annotations = []

#     for _, row in data.iterrows():
#         class_id = 0 if row['class'] == 'Graffiti' else 1 # 0 for graffiti, 1 for not graffiti
#         x_center = (row['xmin'] + row['xmax']) / 2 / row['width']
#         y_center = (row['ymin'] + row['ymax']) / 2 / row['height']
#         box_width = (row['xmax'] - row['xmin']) / row['width']
#         box_height = (row['ymax'] - row['ymin']) / row['height']

#         # Append the annotation to the list
#         yolo_annotations.append([row['filename'], class_id, x_center, y_center, box_width, box_height])

#     return pd.DataFrame(yolo_annotations, columns=['filename', 'class_id', 'x_center', 'y_center', 'box_width', 'box_height'])

# if __name__ == '__main__':
#     yolo_df = convert_to_yolo(data)
#     print(yolo_df)

#     unique_classes = yolo_df['class_id'].unique()
#     num_unique_classes = yolo_df['class_id'].nunique()

#     # Display the results
#     print(f"Unique class labels: {unique_classes}")
#     print(f"Number of unique classes: {num_unique_classes}")

#     yolo_df.to_csv('data/train_yolo_annotations.csv', index=False)
