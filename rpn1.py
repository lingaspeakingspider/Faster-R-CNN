import os
import logging
import keras.backend as K
from keras.models import Sequential
from keras.layers import *
from pathlib import Path
import cv2 as cv
import tensorflow as tf
import numpy as np

class RPN:
    def __init__(self, image_ds:Path, annotation_ds:Path):
        """
        image_ds: Folder of images
        annotation_ds: Folder of annotations
        """
        self.image_ds = image_ds
        self.annotation_ds = annotation_ds

    def convolutional_layers(self, image:np.ndarray) -> tf.Tensor:
        x = Conv2D(16, kernel_size = (3,3), activation="relu")(image)
        x = Conv2D(32, kernel_size = (3,3), activation="relu")(x)
        x = Conv2D(64, kernel_size = (3,3), activation="relu")(x)
        x = Conv2D(128, kernel_size = (3,3), activation="relu")(x)
        return x

    def anchor_boxes(self,feature_map:np.ndarray, aspect_ratios:list, scales:list, grid_size: int):
        """
        Args:
        Aspect Ratios: Aspect ratio refers to the ratio of the width to the height of an object or an image.
        Example for aspect ratios: [1:1, 1:2, 2:1]
        Scales: scales represent the sizes of the boxes relative to the size of the input image. 
        Example for scales: [32,64,128]
        Grid Size: Number of grid cells on an edge.

        Returns:
        A list which contents anchor box coordinates of image
        """
        #Seperate grid cells
        grid_coordinates = list()
        anchor_centers = list()

        for i in range(grid_size[0]): #Rows
            w = feature_map.shape[0] // grid_size[0]
            x = i * (w)
            
            for j in range(grid_size[1]): #Columns
                h = feature_map.shape[1] // grid_size[1]
                y = j * (h)

                grid_coordinates.append([x,y,w,h])
                middle = feature_map[w-x][h-y]
                for value in range(len(middle)):
                    middle[value] = 255

                x_center = (i + 0.5) * w
                y_center = (j + 0.5) * h
                anchor_centers.append((x_center, y_center))

        #Generate anchor boxes
        anchor_boxes_list = list()
        for grid in grid_coordinates:  

            grid_anchor_index = 0
            for scale in scales:
                for ratio in aspect_ratios:
                    width = scale * np.sqrt(ratio)
                    height = scale / np.sqrt(ratio)
                    
                    x_center = anchor_centers[grid_anchor_index][0]
                    y_center = anchor_centers[grid_anchor_index][1]

                    # Calculate the coordinates of the anchor box
                    x1 = max(0, x_center - width / 2)
                    y1 = max(0, y_center - height / 2)
                    x2 = min(feature_map.shape[0], x_center + width / 2)
                    y2 = min(feature_map.shape[1], y_center + height / 2)

                    x_top_left = x1
                    y_top_left = y1

                    # x1 /= height_roi
                    # y1 /= width_roi
                    # x2 /= height_roi
                    # y2 /= width_roi

                    anchor_boxes_list.append([x1, y1, x2, y2])
                    grid_anchor_index += 1

        return anchor_boxes_list
    
    def calculate_iou(self, anchor_boxes:list, ground_truth_boxes:list, iou_threshold: float = 0.7):
        """
        Args:
        Anchor boxes: Predicted boxes
        Ground truth boxes: True boxes, which annotated boxes in the dataset
        IoU (Intersection of Union): Area Overlap / Area Union

        Returns:
        A list of positive labeled anchor boxes
        A list of iou values
        """

        self.positive_labeled_anchors = list()
        iou_list = list()
        true_boxes = list()

        for true_box in ground_truth_boxes:
            [x_true, y_true, w_true, h_true] = true_box

            for anchor_box in anchor_boxes:
                [x1,y1,x2,y2] = anchor_box
                w_anchor = x2 - x1
                h_anchor = y2 - y1

                x_left = max(x1, x_true)
                y_top = max(y1, y_true)
                x_right = min(x1 + w_anchor, x_true + w_true)
                y_bottom = min(y1 + h_anchor, y_true + h_true)

                # Calculate intersection rectangle
                intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)
    
                area_anchor = w_anchor * h_anchor
                area_true = w_true * h_true

                # Calculate union area
                union_area = area_anchor + area_true - intersection_area

                #Calculate IoU
                iou = intersection_area / union_area
                iou_list.append(iou)
                
                if iou >= iou_threshold:
                    self.positive_labeled_anchors.append(anchor_box)
                    true_boxes.append(true_box)

                return [self.positive_labeled_anchors, iou_list]
    
    def train(self, positive_labeled_anchors: list, epochs: int = 10, batch_size : int = 32):

        num_anchors = len(positive_labeled_anchors)
        model_classifiers = Sequential()
        model_classifiers.add(Dense(num_anchors * 2, activation="softmax")) # Classifier
        model_classifiers.add(Dense(num_anchors * 4, activation="linear")) # Regressor
        
        model_classifiers.compile(optimizer='adam',
              loss={'rpn_cls': 'binary_crossentropy', 'rpn_reg': 'mse'},
              loss_weights={'rpn_cls': 1.0, 'rpn_reg': 1.0})
        
        for epoch in range(epochs):
            logging.basicConfig(logging.debug)
            logging.info(f"Epoch {epoch + 1}/{epochs}: \n")
            batch_imgs = list()
            batch_anchors = list()
            batch_annotations = list()
            
            image_paths = os.listdir(self.image_ds)
            annotation_list = os.listdir(self.annotation_ds)

            # Images
            for i in image_paths:
                img = cv.imread(i)
                feature_map = K.eval(self.convolutional_layers(img))
                batch_imgs.append(feature_map)
            
            # Anchor Boxes
            for i in range(batch_size):
                for image in batch_imgs:
                    batch_anchors.extend(self.anchor_boxes(image, [0.5, 1, 2], [32,64,128], grid_size=14))
            
            # Annotations
            for annotation_file in annotation_list:
                with open(annotation_file, "r") as f:
                    coordinates = f.readline()
                    coordinates = coordinates.split(" ")
                    batch_annotations.extend(coordinates)

            batch_imgs = np.array(batch_imgs)
            batch_anchors = np.array(batch_anchors)
            batch_annotations = np.array(batch_annotations)
            model_classifiers.train_on_batch(batch_imgs, {'rpn_cls': batch_anchors, 'rpn_reg': batch_annotations})
                    

rpn = RPN("/home/oguz/Desktop/output.jpg")
rpn.anchor_boxes([(1,1),(1,2),(2,1)], [0.5, 1.0, 2.0], (2,2))
