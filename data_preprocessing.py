from keras.applications import *
import shutil
import os
from pathlib import Path

class DataProcessing:
    def __init__(self, image_dataset_dir: Path, annotation_dir:Path, nb_classes: int):
        self.image_dataset = image_dataset_dir
        self.annotations = annotation_dir
        self.nb_classes = nb_classes

    def yolobbox2bbox(x,y,w,h):
        x1, y1 = x-w/2, y-h/2
        x2, y2 = x+w/2, y+h/2
        return x1, y1, x2, y2
        
    def split_train_test(self, validation_data : bool = False, test_size : float = 0.25):
        image_ds_list = os.listdir(self.image_dataset)
        os.mkdir("train_dataset")
        os.mkdir("test_dataset")
        if validation_data is True:
            os.mkdir("validation_dataset")

        timer = 0
        timer_val = 0
        
        for i in image_ds_list:
            if timer <= len(image_ds_list) * test_size:
                shutil.move(os.path.join(image_ds_list, i), "test_dataset")
                timer += 1

            elif validation_data is True:
                if timer_val <= len(image_ds_list) * 0.3:
                    shutil.move(os.path.join(image_ds_list, i), "validation_dataset")
                    timer_val += 1

            else:
                shutil.move(os.path.join(image_ds_list, i), "train_dataset")

    def backbone_model(self):
        model = VGG16(include_top= False, classes = self.nb_classes)
        model.layers.pop(-1)
        model.layers.pop(-1)
        return model

    