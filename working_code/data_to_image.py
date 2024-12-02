import numpy as np

class Image:

    '''A way of presenting basic information about an individual image from an images dataset
    
    enitre_dataset: a numpy array of the entire data set with. 
        each individual data point is an array of grayscale pixels, with the last point in the array being the category
        
    choice_of_image_index: the index of a chosen image'''

    def __init__(self, entire_dataset: np.array, choice_of_image_index: int):
        self.image = entire_dataset[choice_of_image_index,:-1]
        self.category = entire_dataset[choice_of_image_index,-1]

    def _tell_class(self):
        class_here = 'Error'
        if self.category == 0:
            class_here = '0 T-shirt/top'
        elif self.category == 1:
            class_here = '1 Trouser'
        elif self.category == 2:
            class_here = '2 Pullover'
        elif self.category == 3:
            class_here = '3 Dress'
        elif self.category == 4:
            class_here = '4 Shirt'

        return(f"Class: {class_here}")
