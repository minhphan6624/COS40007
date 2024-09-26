##Validate with IOU

import torch
import os
import csv
import cv2

model = torch.hub.load('ultralytics/yolov5', 'custom', path='/content/yolov5/runs/train/exp/weights/best.pt',force_reload=True)

import scipy.optimize
def bbox_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = xB - xA + 1
    interH = yB - yA + 1

    if interW <= 0 or interH <= 0:
        return 0.0

    interArea = interW * interH
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def match_bboxes(bbox_gt, bbox_pred, IOU_THRESH=0.5):
    n_true = bbox_gt.shape[0]
    n_pred = bbox_pred.shape[0]
    MIN_IOU = 0.0

    iou_matrix = np.zeros((n_true, n_pred))
    for i in range(n_true):
        for j in range(n_pred):
            iou_matrix[i, j] = bbox_iou(bbox_gt[i,:], bbox_pred[j,:])

    if n_pred > n_true:
        diff = n_pred - n_true
        iou_matrix = np.concatenate((iou_matrix, np.full((diff, n_pred), MIN_IOU)), axis=0)

    if n_true > n_pred:
        diff = n_true - n_pred
        iou_matrix = np.concatenate((iou_matrix, np.full((n_true, diff), MIN_IOU)), axis=1)

    idxs_true, idxs_pred = scipy.optimize.linear_sum_assignment(1 - iou_matrix)

    ious = iou_matrix[idxs_true, idxs_pred] if idxs_true.size and idxs_pred.size else np.array([])

    sel_pred = idxs_pred < n_pred
    idx_pred_actual = idxs_pred[sel_pred]
    idx_gt_actual = idxs_true[sel_pred]
    ious_actual = iou_matrix[idx_gt_actual, idx_pred_actual]
    sel_valid = (ious_actual > IOU_THRESH)
    return idx_gt_actual[sel_valid], idx_pred_actual[sel_valid], ious_actual[sel_valid], sel_valid.astype(int)

import numpy as np
results_folder = "/content/results"
os.makedirs(results_folder, exist_ok=True)


def compute_and_write_csv(model, output_csv, test_images_folder, test_labels_folder):
    """
    Function to compute the IoU for each image and write the results to a CSV file.

    Parameters:
    - model: YOLO model for inference.
    - output_csv: Path to the CSV file where results will be written.
    - test_images_folder: Path to the folder containing test images.
    - test_labels_folder: Path to the folder containing ground truth labels.
    """

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['image_name', 'confidence', 'IoU'])

        for image_name in os.listdir(test_images_folder):
            image_path = os.path.join(test_images_folder, image_name)

            try:
                results = model(image_path)
            except Exception as e:
                print(f"Skipping {image_name} due to an error: {e}")
                continue

            predictions = results.xyxy[0].cpu().numpy()

            label_name = f"{os.path.splitext(image_name)[0]}.txt"
            label_path = os.path.join(test_labels_folder, label_name)

            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    ground_truth = []
                    for row in f:
                        class_id, x_center, y_center, width, height = map(float, row.split())
                        img = cv2.imread(image_path)
                        img_h, img_w = img.shape[:2]
                        x_min = int((x_center - width / 2) * img_w)
                        y_min = int((y_center - height / 2) * img_h)
                        x_max = int((x_center + width / 2) * img_w)
                        y_max = int((y_center + height / 2) * img_h)
                        ground_truth.append([x_min, y_min, x_max, y_max])

                ground_truth = np.array(ground_truth)
                pred_boxes = predictions[:, :4]  # Get predicted bounding boxes
                confs = predictions[:, 4]  # Get confidence scores

                idx_gt, idx_pred, ious, valid = match_bboxes(ground_truth, pred_boxes)

                for gt_idx, pred_idx, iou_val in zip(idx_gt, idx_pred, ious):
                    confidence = confs[pred_idx]
                    csvwriter.writerow([image_name, confidence, iou_val])

            else:
                csvwriter.writerow([image_name, 0, 0])

    print(f"Results saved to {output_csv}")



output_csv_path = "/content/results/iteration_1.csv"
test_images_folder_path = "/content/test_dataset/images"
test_labels_folder_path = "/content/test_dataset/labels"

model_path = "/content/yolov5/runs/train/exp/weights/best.pt"
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path,force_reload=True)

compute_and_write_csv(model, output_csv_path, test_images_folder_path, test_labels_folder_path)

import csv

def check_iou_threshold_from_csv(csv_file, iou_threshold=0.9, required_percentage=0.8):
    total_images = 0
    high_iou_images = 0

    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)

        for row in reader:
            total_images += 1
            iou_value = float(row['IoU'])

            if iou_value >= iou_threshold:
                high_iou_images += 1

    if total_images > 0:
        percentage = high_iou_images / total_images
        if percentage >= required_percentage:
            print(f"Success: {100 * percentage:.2f}% of images have IoU over {iou_threshold * 100}%.")
        else:
            print(f"Failure: Only {100 * percentage:.2f}% of images have IoU over {iou_threshold * 100}%.")
    else:
        print("No images found in the CSV file.")


csv_file_path = "/content/results/iteration_1.csv"  # Path to your CSV file
check_iou_threshold_from_csv(csv_file_path, iou_threshold=0.9)
