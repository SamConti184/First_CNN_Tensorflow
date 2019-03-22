from PIL import Image
from typing import List
import os
import random

#Using Pyhton package pillow, we check some basic propoerties of random images
def explore_training_files(folder: str, files: List[str]):
    for file in files:
        image = Image.open("./cell-images-for-detecting-malaria/cell_images/" + folder + "/" + file)
        width, height = image.size
        channels = image.getbands()
        print("The image {} has shape {} x {} and channels {}".format(file, width,
                                                            height, channels))

#Checking randomly size and channels of some images in the training set,
#in order to see what kind of preprocessing is needed.
#For example we may see that images are both RGB and grayscale, or,
#more importantly, that they not have a square dimension in order to be the CNN input.
def get_random_training_files(folder: str) -> List[str]:
    print("Some random images from the " + folder + " dataset: \n")
    files_list = os.listdir("./cell-images-for-detecting-malaria/cell_images/" + folder)
    total_files = len(files_list)
    print("Total of {} files under the class {}\n".format(total_files, folder))
    selected_files = []
    for _ in range(10):
        num_file = random.randint(0, total_files-1)
        file_name = files_list[num_file]
        selected_files.append(file_name)
    return selected_files

def explore_data():
    files = get_random_training_files("Parasitized")
    explore_training_files("Parasitized", files)
    files = get_random_training_files("Uninfected")
    explore_training_files("Uninfected", files)

#By exploring some random data from both the classes, we see that most images are
#RGB and of different dimensions.
if __name__ == "__main__":
    explore_data()
