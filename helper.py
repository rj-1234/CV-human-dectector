import matplotlib.pyplot as plt
import cv2
import math
import numpy as np
import os
import pathlib
from os import listdir

class Helper:
    def __init__(self):
        """ 
        Specify the Image directory(should only contain the images 
        and one other directory named 'output' to save output images) 
        """
        self.train_pos_images = ".\\Human\\Train_Positive\\" 
        self.train_neg_images = ".\\Human\\Train_Negative\\" 
        self.test_pos_images = ".\\Human\\Test_Positive\\" 
        self.test_neg_images = ".\\Human\\Test_Neg\\" 
        self.hog_vectors_file = ".\\Human\\"
        # self.image_dir = ".\\Human\\Train_Positive\\" 

        # HOG Stuff
        self.cell_size = 8              # 8 x 8 pixels
        self.block_size = 16            # 16 x 16 pixels
        self.bin_size = 9               # Number of bins per cell
        self.angle_unit = 180 // self.bin_size  # to find the appropriate index in the cell histogram

    def read_images(self, source_path):
        '''
        read the image in a specific directory
        :param source_path: path to the directory contains images
        :return: a list of image objects
        '''
        imgs = []
        files = []
        for file in listdir(source_path):
            if file.split("_")[0] == "outputs":
                pass
            else:
                img = cv2.imread(source_path + file)
                imgs.append(img)
                files.append(file)
        return imgs, files

    def HOG(self, gradient_magnitude, gradient_angle):
        height, width = gradient_magnitude.shape

        # create a cell gradient vector and initialize with zeros
        cell_gradient_vector = np.zeros((height // self.cell_size, width // self.cell_size, self.bin_size))

        # calculate the cell histogram for each cell in the cell gradient vector
        for i in range(cell_gradient_vector.shape[0]):
            for j in range(cell_gradient_vector.shape[1]):
                cell_magnitude = gradient_magnitude[i * self.cell_size:(i + 1) * self.cell_size,
                                 j * self.cell_size:(j + 1) * self.cell_size]
                cell_angle = gradient_angle[i * self.cell_size:(i + 1) * self.cell_size,
                             j * self.cell_size:(j + 1) * self.cell_size]
                cell_gradient_vector[i][j] = self.cell_gradient(cell_magnitude, cell_angle)

        hog_image = self.plot_gradient(np.zeros([height, width]), cell_gradient_vector)
        hog_vector = []
        for i in range(cell_gradient_vector.shape[0] - 1):
            for j in range(cell_gradient_vector.shape[1] - 1):
                # creating a block vector
                block_vector = []
                block_vector.extend(cell_gradient_vector[i][j])
                block_vector.extend(cell_gradient_vector[i][j + 1])
                block_vector.extend(cell_gradient_vector[i + 1][j])
                block_vector.extend(cell_gradient_vector[i + 1][j + 1])
                # normaling the block vector
                mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
                magnitude = mag(block_vector)
                if magnitude != 0:
                    normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
                    block_vector = normalize(block_vector, magnitude)
                hog_vector.append(block_vector)
        return hog_vector, hog_image

    def cell_gradient(self, cell_magnitude, cell_angle):
        orientation_centers = [0] * self.bin_size
        for i in range(cell_magnitude.shape[0]):
            for j in range(cell_magnitude.shape[1]):
                gradient_strength = cell_magnitude[i][j]
                gradient_angle = cell_angle[i][j]
                min_angle, max_angle, mod = self.get_closest_bins(gradient_angle)
                orientation_centers[min_angle] += (gradient_strength * (1 - (mod / self.angle_unit)))
                orientation_centers[max_angle] += (gradient_strength * (mod / self.angle_unit))
        return orientation_centers
    
    def get_closest_bins(self, gradient_angle):
        """
        calculates the ratio by which the magnitude will be divided in the two bins
        returns bin 1 and bin 2 along with the ratio for distribution
        """
        if gradient_angle > 170:
            gradient_angle = (gradient_angle - 180)
        idx = int(gradient_angle / self.angle_unit)
        mod = gradient_angle % self.angle_unit
        if idx == self.bin_size:
            return idx - 1, (idx) % self.bin_size, mod
        return idx, (idx + 1) % self.bin_size, mod

    def plot_gradient(self, image, cell_gradient):
        cell_width = self.cell_size / 2
        max_mag = np.array(cell_gradient).max()
        for x in range(cell_gradient.shape[0]):
            for y in range(cell_gradient.shape[1]):
                cell_grad = cell_gradient[x][y]
                cell_grad /= max_mag
                angle = 0
                angle_gap = self.angle_unit
                for magnitude in cell_grad:
                    angle_radian = math.radians(angle)
                    x1 = int(x * self.cell_size + magnitude * cell_width * math.cos(angle_radian))
                    y1 = int(y * self.cell_size + magnitude * cell_width * math.sin(angle_radian))
                    x2 = int(x * self.cell_size - magnitude * cell_width * math.cos(angle_radian))
                    y2 = int(y * self.cell_size - magnitude * cell_width * math.sin(angle_radian))
                    cv2.line(image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
                    angle += angle_gap
        return image

    def convolution(self, image, kernel):
        image_height = image.shape[0]
        image_width  = image.shape[1]

        kernel_height = kernel.shape[0]
        kernel_width = kernel.shape[1]

        height = (kernel_height - 1) // 2
        width = (kernel_width - 1) // 2

        output = np.zeros((image_height, image_width))

        for i in np.arange(height, image_height - height):
            for j in np.arange(width, image_width - width):
                sum = 0
                for k in np.arange(-height, height+1):
                    for l in np.arange(-width, width+1):
                        a = image[i+k, j+l]
                        p = kernel[height+k, width+l]
                        sum += (p * a)
                output[i, j] = sum
        
        return output


    def unison_shuffled_copies(self, a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    def gaussian_filter(self, img):
        """
        Returns a smoothed image with a 7 x 7 Gaussian filter for the input image
        """
        gaussian_kernel = (1.0/140) * np.array(
                            [[1 ,1 ,2 ,2 ,2 ,1 ,1],
                            [1 ,2 ,2 ,4 ,2 ,2 ,1],
                            [2 ,2 ,4 ,8 ,4 ,2 ,2],
                            [2 ,4 ,8 ,16,8 ,4 ,2],
                            [2 ,2 ,4 ,8 ,4 ,2 ,2],
                            [1 ,2 ,2 ,4 ,2 ,2 ,1],
                            [1 ,1 ,2 ,2 ,2 ,1 ,1]])

        s = sum(sum(gaussian_kernel))
        print(s)
        print("Running Smoothing operation using a 7 x 7 Gaussian Kernel")

        after7x7Smoothning = np.copy(img)

        for i in range(3,len(img)-3):
            for j in range(3,len(img[i])-3):
                after7x7Smoothning[i][j] = (sum(map(sum, (gaussian_kernel * after7x7Smoothning[i-3:i+4,j-3:j+4]))))
        
        print("DONE")
        return after7x7Smoothning

    def greyscale_Image(self, rgb):
        """
        Returns a gray scale version of the input image
        """
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

        return gray

    def gradient(self, img):
        """
        Returns the horizonal and vertical gradient for the input image
        """
        prewitt_hor = np.array([[-1 ,0 ,1],
                                [-1 ,0 ,1],
                                [-1 ,0 ,1]])

        prewitt_ver = np.array([[-1 ,-1 ,-1],
                                [ 0 , 0 , 0],
                                [ 1 , 1 , 1]])

        hor_gradient = np.copy(img)
        # print("Running horizonal gradient operation using prewitt_hor Kernel")
        hor_gradient = self.convolution(hor_gradient, prewitt_hor)
        # print("DONE")

        ver_gradient = np.copy(img)
        # print("Running vertical gradient operation using prewitt_ver Kernel")
        ver_gradient = self.convolution(ver_gradient, prewitt_ver)
        # print("DONE")

        return hor_gradient, ver_gradient

    def NMsuppression(self, img, grad_h, grad_v, grad_m):

        height = img.shape[0]
        width = img.shape[1]

        angle = np.arctan2(grad_v, grad_h)

        quantized_angle = (np.round(angle * (5.0 / np.pi) + 5 )) % 5 
        supressed_grad_m = grad_m.copy()

        print("Running Non Maxima Suppression operation on gradient magnitude")
        for i in range(height):
            for j in range(width):
                if i == 0 or i == height - 1 or j == 0 or j == width - 1:
                    supressed_grad_m[i, j] = 0

                qa = quantized_angle[i, j] % 4

                if qa == 0: # 0 -> E-W (horizontal)
                    if supressed_grad_m[i, j] <= supressed_grad_m[i, j-1] or supressed_grad_m[i, j] <= supressed_grad_m[i, j+1]:
                        supressed_grad_m[i, j] = 0
                
                if qa == 1: # 1 -> NE SW
                    if supressed_grad_m[i, j] <= supressed_grad_m[i-1, j+1] or supressed_grad_m[i, j] <= supressed_grad_m[i+1, j-1]:
                        supressed_grad_m[i, j] = 0
                
                if qa == 2: # 2 -> N-S (vertical)
                    if supressed_grad_m[i, j] <= supressed_grad_m[i-1, j] or supressed_grad_m[i, j] <= supressed_grad_m[i+1, j]:
                        supressed_grad_m[i, j] = 0
                
                if qa == 3: # 3 -> NW SE
                    if supressed_grad_m[i, j] <= supressed_grad_m[i-1, j-1] or supressed_grad_m[i, j] <= supressed_grad_m[i+1, j+1]:
                        supressed_grad_m[i, j] = 0
        print("DONE")
        return supressed_grad_m
                
    def thresholding(self, img, thresholdValue):
        img_copy = np.copy(img)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img_copy[i, j] < thresholdValue: img_copy[i, j] = 0
        return img_copy

    def ptile(self, img, p):
        height = img.shape[0]
        width = img.shape[1]

        gray_val = []

        for i in range(height):
            for j in range(width):
                if img[i, j] > 0.0:
                    gray_val.append(img[i, j])
        print ("Length of Gray value list : "+str(len(gray_val)))
        gray_val.sort(reverse=True)
        # print(gray_val[:100])
        idx = int((p/float(100)) * len(gray_val))
        print ("Chosen index from gray value list : "+str(idx))
        print ("Total no. of edges detected for p = "+str(p)+" : "+str(len(gray_val[:idx+1])))
        print ("Chosen gray level value for this p value : "+str(gray_val[idx]))
        return gray_val[idx]

    def save(self, images, names, image_dir):
        """
        Takes 2 lists
            images : list of images as numpy arrays
            names  : list of names as a string
        """
        for i in range(len(names)):
            # print("saving : "+names[i])
            # print(names[i])
            path = image_dir+"outputs_"+names[i].split(" ")[-1]+"\\"
            pathlib.Path(path).mkdir(parents=True, exist_ok=True) 
            cv2.imwrite(os.path.join(path , str(names[i])+'_image.png'), images[i])

    def show(self, images, names):
        """
        Takes 2 lists
            images : list of images as numpy arrays
            names  : list of names as a string
        """
        for i in range(len(names)):
            if str(images[i]) == 'gray_img':
                plt.imshow(images[i], cmap = plt.get_cmap('gray'))
            else:
                plt.imshow(images[i])
            print("showing : "+str(names[i]))        
            plt.show()
