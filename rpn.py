import tensorflow as tf
import os
import numpy as np
from keras.layers import *
from pathlib import Path
import cv2 as cv
import logging 

class RPN:
    def __init__(self, image_dataset:Path, annotation_dataset:Path):
        self.image_ds = image_dataset
        self.annotation_ds = annotation_dataset

    def apply(self):
        """
        Applies RPN and returns saved model
        """
        dataset_list = os.listdir(self.image_ds)
        annotation_list = os.listdir(self.annotation_ds)
        for image in dataset_list:
            image_name = image.split(".")
            image_name = image_name.pop(1)
            ground_truth_box_string = self.annotation_ds(dataset_list.index(image))
            ground_truth_box = list(ground_truth_box_string.split(",").pop(0))

            img = cv.imread(image)
            grids, anchors_center, image = self.generate_grids_and_anchor_centers(img, grid_size=30)   
            anchor_boxes, images = self.generate_anchor_boxes([0.5, 1, 2], [32,64,128], image, grids, anchors_center)
            [x_true, y_true, w_true, h_true] = ground_truth_box

            positives = list()
            negatives = list()
            positive_anchor_center = self.extract_grids(grids, ground_truth_box, anchors_center)
            for anchor in anchor_boxes:
                boolean = self.calculate_iou(ground_truth_box, anchor, positive_anchor_center_list=positive_anchor_center,
                                            anchor_center_list=anchors_center, anchor_box_index=anchor_boxes.index(anchor))
                
                if boolean == 1:
                    positives.append(anchor)

                elif boolean == 0:
                    negatives.append(anchor)

            prediction_ds = self.store_predictions(anchor_boxes, image_name, 
                                    "/home/biocally/Desktop/anchor_predictions",
                                   positives, negatives)
            
            rpn_model = self.rpn_model(len(anchor_boxes))
            self.training(self.image_ds, self.annotation_ds, prediction_ds, rpn_model, 
                          model_save_path = "/home/biocally/Desktop")

    def generate_grids_and_anchor_centers(self, image:np.ndarray, grid_size : int = 14):
        """
        Args:
        image -> Target image
        grid_size -> Number of grid cells on one edge

        Returns:
        A list of grid coordinates and anchor centers coordinates
        """

        anchor_centers = list()
        grid_coordinates = list()
        cell_width = image.shape[0] // grid_size
        cell_height = image.shape[1] // grid_size
        self.matrix = list()

        for i in range(grid_size): #Rows
            x = i * (cell_width)
            
            for j in range(grid_size): #Columns
                y = j * (cell_height)

                cv.circle(image, (x + cell_width // 2, y + cell_height // 2), radius = 2, color=(0,255,255), thickness=1)
                cv.rectangle(image, (x,y), (x + cell_width, y + cell_height), color = (255,0,0), thickness=1)
                grid_coordinates.append([x, y, cell_width, cell_height])
                anchor_centers.append((x + cell_width // 2, y + cell_height // 2))

        self.grid_size = grid_size
        return grid_coordinates, anchor_centers, image

    def generate_anchor_boxes(self, aspect_ratios:list, scales:list, 
                              image:np.ndarray, grid_coordinates:list,
                              anchor_centers:list):
        """
        Generates anchor boxes according to aspect ratios and scales
        Anchor box number = Length of aspect ratios list * Length of scales list
        """
        anchor_boxes = list()
        for grid in grid_coordinates:
            index = grid_coordinates.index(grid)

            for scale in scales:
                for ratio in aspect_ratios:
                    width = scale * np.sqrt(ratio)
                    height = scale / np.sqrt(ratio)

                    x_center = anchor_centers[index][0]
                    y_center = anchor_centers[index][1]

                    x1 = max(0, x_center - width / 2)
                    y1 = max(0, y_center - height / 2)
                    x2 = min(image.shape[0], x_center + width / 2)
                    y2 = min(image.shape[1], y_center + height / 2)

                    anchor_boxes.append([x1,y1,x2,y2])
                    
        return anchor_boxes, image

    def calculate_iou(self, ground_truth_box:list, anchor_box:list, positive_anchor_center_list:list, 
                      anchor_box_index:int, anchor_center_list:list,
                      iou_threshold_value: float = 0.7):
        
        """
        Calculates IoU between anchor box and ground_truth box
        """

        [x_true, y_true, w_true, h_true] = ground_truth_box         
        [x1,y1,x2,y2] = anchor_box 

        if (anchor_box_index / 9) % 9 == 0:
            index = int(anchor_box_index / 9) - 1

        else:
            index = int(anchor_box_index / 9) 

        if positive_anchor_center_list.count(anchor_center_list[index]) != 0:

            w_anchor = abs(x2 - x1)
            h_anchor = abs(y2 - y1)
            self.anchor_width_height = [w_anchor, h_anchor]

            x_left = max(x1, x_true)
            y_top = max(y1, y_true)
            x_right = min(x2, x_true + w_true)
            y_bottom = min(y2, y_true + h_true)

            intersection_area = abs(x_right - x_left) * abs(y_top - y_bottom)

            area_anchor = w_anchor * h_anchor
            area_true = w_true * h_true

            # Calculate union area
            union_area = area_anchor + area_true - intersection_area

            # Calculate IoU
            iou = float(intersection_area / union_area)

            print("Overlap: ", intersection_area)
            print("Union: ", union_area)
            print(iou)

            self.coordinates = [x_left, x_right, y_top, y_bottom]

            if iou >= iou_threshold_value:
                return 1
                
            elif iou <= 0.3:
                return 0

    def draw_rectangle(self, image:np.ndarray, boxes:list = None, true_box : list = None):

        if boxes != None:    
            for box in boxes:
                [x1,y1,x2,y2] = box
                [x1,y1,x2,y2] = list(map(int, [x1,y1,x2,y2]))
                cv.rectangle(image, (x1,y1), (x2, y2), color = (255,255,255), thickness=1)

        if true_box is not None:
            [x1,y1,x2,y2] = true_box
            [x1,y1,x2,y2] = list(map(int, [x1,y1,x2,y2]))
            cv.rectangle(image, (x1,y1), (x2 + x1, y2 + y1), color = (0,0,255), thickness=1)

        cv.imshow("Frame", image)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def extract_grids(self, grid_coordinates:list, ground_truth_box, anchor_center_list:list):
        """
        Extracts grid if anchor box has a overlap area with ground truth box
        """

        [x_true, y_true, w_true, h_true] = ground_truth_box   
        positive_grids = list()

        for grid in grid_coordinates:
            start_x = grid[0]
            finish_x = grid[0] + grid[2]
            start_y = grid[1]
            finish_y = grid[1] + grid[3]

            if finish_x > x_true and start_x < x_true + w_true and finish_y > y_true and start_y < y_true + h_true:
                index = grid_coordinates.index(grid)
                anchor_center = anchor_center_list[index]
                positive_grids.append(anchor_center)

        return positive_grids
    
    def balancing_anchor_boxes(self, positive_anchors:list,
                               negative_anchors: list,
                               ignore_anchors:list, 
                               mini_batch_size:int = 256, 
                               negative_ratio:float = 0.5):
        """
        Balances anchor boxes according to mini batch size (default 256)
        """
        n_foreground = int((1-negative_ratio) * mini_batch_size)
        n_background = int(negative_ratio * mini_batch_size)

        # check if we have excessive positive anchors
        if len(positive_anchors) > n_foreground:
            ignore_index = positive_anchors[n_foreground:]
            for i in ignore_index:
                index = positive_anchors.index(i)
                positive_anchors.pop(index)
                ignore_anchors.append(i)

        # sample background examples if we don't have any enough positive examples to match the anchor box size
        if len(positive_anchors) < n_foreground:
            diff = n_foreground - len(positive_anchors)
            n_background += diff

        # check if we have excessive positive anchors
        if len(negative_anchors) > n_background:
            ignore_index = negative_anchors[n_background:]
            for i in ignore_index:
                index = negative_anchors.index(i)
                positive_anchors.pop(index)
                ignore_anchors.append(i)

    def store_predictions(self, anchor_boxes:list, image_name:str, main_dir:Path, 
                          positive_anchors:list,
                          negative_anchors:list):
        
        """
        Creates files which includes anchor box classes and coordinates
        Returns: Path of prediction dataset
        """
        os.chdir(main_dir)

        if image_name[-3:] != "jpg" or image_name[-3:] != "png" or image_name[-4:] != "jpeg":
            raise ValueError("Expected '.jpg', '.png' or '.jpeg'")
        
        if image_name[-4:] == "jpeg":
            file_name = image_name.replace(image_name[-4: ], "")
            file_name = file_name + "txt"

        if image_name[-4:] != "jpeg":
            file_name = image_name.replace(image_name[-3: ], "")
            file_name = file_name + "txt"

        for coordinate in anchor_boxes:
            [x1,y1,x2,y2] = coordinate
            w = abs(x2-x1)
            h = abs(y2-y1)

            if positive_anchors.count(coordinate) != 0:
                class_index = 1

            elif negative_anchors.count(coordinate) != 0:
                class_index = 0

            else:
                class_index = -1

            with open(file_name, "a") as f:
                f.write(class_index + " " + f"{x1} {y1} {w} {h}" + "\n")

            return main_dir

    def binary_cross_entropy_loss(predictions:np.ndarray, targets):
        epsilon = 1e-7  # Small constant to avoid division by zero
        loss = -targets * np.log(predictions + epsilon) - (1 - targets) * np.log(1 - predictions + epsilon)
        return loss
    
    def smooth_l1_loss(predictions:np.ndarray, targets):
        diff = predictions - targets
        abs_diff = np.abs(diff)
        loss = np.where(abs_diff < 1, 0.5 * diff**2, abs_diff - 0.5)
        return loss

    def rpn_model(self, nb_anchors:int, image_size : int = 450):
        inputs = Input(shape=(image_size, image_size, 3))

        #Backbone network 
        backbone = tf.keras.applications.ResNet50(include_top=False, weights='imagenet')
        backbone.trainable = False
        backbone_output = backbone(inputs)

        rpn_conv = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(backbone_output)
        obj_score = tf.keras.layers.Conv2D(nb_anchors, (1, 1), activation='sigmoid')(rpn_conv)
        bbox_deltas = tf.keras.layers.Conv2D(nb_anchors * 4, (1, 1), activation='linear')(rpn_conv)

        model = tf.keras.Model(inputs=inputs, outputs=[obj_score, bbox_deltas])
        return model

    def training(self, image_dataset:Path, 
                 annotation_dataset:Path, 
                 predictions_dataset:Path,
                 model : tf.keras.Model,
                 optimizer: tf.keras.optimizers = tf.keras.optimizers.SGD,
                 model_save_path:Path = os.getcwd(),
                 batch_size: int = 32, 
                 epochs:int = 10):

        image_list = os.listdir(image_dataset)
        annotation_list = os.listdir(annotation_dataset)
        predictions_list = os.listdir(predictions_dataset)

        images = np.array(image_list)
        annotations = np.array(annotation_list)
        predictions_array = np.array(predictions_list)

        dataset = tf.data.Dataset.from_tensor_slices((images, annotations))
        dataset = dataset.shuffle(buffer_size=1000).batch(batch_size)

        for epoch in range(epochs):
            total_loss = 0.0
            
            for batch_images, batch_annotations in dataset:
                with tf.GradientTape() as tape:
                    timer = 0
                    pred_coordinates = np.array([float(predictions_array[1]), float(predictions_array[2]),
                                                float(predictions_array[3]), float(predictions_array[4])])
                    
                    annotation_coordinates = np.array([float(annotations[1]), float(annotations[2]),
                                                float(annotations[3]), float(annotations[4])])
                    
                    cls_loss = self.binary_cross_entropy_loss(predictions_array[timer][0], annotations[timer][0])
                    delta_loss = self.smooth_l1_loss(pred_coordinates, annotation_coordinates)

                    loss = cls_loss + delta_loss
                    
                gradients = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                total_loss += loss
                average_loss = total_loss / batch_size

            logging.basicConfig(logging.DEBUG)
            logging.info(f"Epoch {epoch + 1}/{epochs}, Loss: {average_loss.numpy():.4f}")

        model.save(os.path.join(model_save_path, "rpn.h5"))

rpn = RPN("/home/biocally/Desktop/train_ds", "/home/biocally/Desktop/annotation_ds")
rpn.apply()



