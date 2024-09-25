import pandas as pd

file_path = 'data/bounding_boxes/train_labels.csv'

# Load the data
data = pd.read_csv(file_path)

def convert_to_yolo(data):
    yolo_annotations = []

    for _, row in data.iterrows():
        class_id = 0 if row['class'] == 'Graffiti' else 1 # 0 for graffiti, 1 for not graffiti
        x_center = (row['xmin'] + row['xmax']) / 2 / row['width']
        y_center = (row['ymin'] + row['ymax']) / 2 / row['height']
        box_width = (row['xmax'] - row['xmin']) / row['width']
        box_height = (row['ymax'] - row['ymin']) / row['height']

        # Append the annotation to the list
        yolo_annotations.append([row['filename'], class_id, x_center, y_center, box_width, box_height])
    
    return pd.DataFrame(yolo_annotations, columns=['filename', 'class_id', 'x_center', 'y_center', 'box_width', 'box_height'])

if __name__ == '__main__':
    yolo_df = convert_to_yolo(data)
    print(yolo_df)

    unique_classes = yolo_df['class_id'].unique()
    num_unique_classes = yolo_df['class_id'].nunique()

    # Display the results
    print(f"Unique class labels: {unique_classes}")
    print(f"Number of unique classes: {num_unique_classes}")

    yolo_df.to_csv('data/train_yolo_annotations.csv', index=False)