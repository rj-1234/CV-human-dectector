from helper import *
import numpy as np
import matplotlib.pyplot as plt
from neural_net import *
import pathlib
import os

def main():
    helper = Helper()

    def preprocess(images, fileame, path, data = "", data_input = [], data_output = []):
        for c in range(len(images)): 
            img = images[c]
            img = np.array(img, dtype=float)
            name = fileame[c]
            print("\n################### "+name+" ###################\n")
            """ Conver to grayscale """
            gray_img = helper.greyscale_Image(img)

            """ Computing horizontal and vertical gradients for the gray image """
            horizontal_gradient_img, vertical_gradient_img = helper.gradient(gray_img)
            # print(horizontal_gradient_img[22])
            # print(vertical_gradient_img[22])
            
            """ Compute the gradient magnitude """
            # print("Computing the Gradient Magnitude")
            gradient_magnitude_img = np.sqrt(np.power(horizontal_gradient_img, 2) + np.power(vertical_gradient_img, 2))
            
            """ Normalizing for the range -> 0 - 255 """
            # print("Normalizing the Gradient Magnitude")
            normalized_gradient_magnitude_img =(gradient_magnitude_img / np.max(gradient_magnitude_img)) * 255
            
            # print("Computing the Gradient angles")
            gradient_angle = np.arctan(vertical_gradient_img/horizontal_gradient_img )* 180 / np.pi

            # print("Replacing the NAN values in the gradient angles with 0")
            gradient_angle[np.isnan(gradient_angle)] = 0
            for i in range(len(gradient_angle)):
                for j in range(len(gradient_angle[i])):
                    if gradient_angle[i][j] < 0:
                        gradient_angle[i][j] += 360

            # print("Computing the HOG Vector for "+name+" image")
            HOG_vector, HOG_image = helper.HOG(normalized_gradient_magnitude_img, gradient_angle)             

            HOG_vector = np.array(HOG_vector)

            # print("Flattening the HOG vector to form a large 7524 x 1 vector")
            flat_HOG_vector = HOG_vector.reshape(HOG_vector.shape[0]*HOG_vector.shape[1])
            
            # Save HOG vector to the output folder
            path = helper.hog_vectors_file+"hog_vectors\\"
            print(path)
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)
            with open(path+"hog_vector_"+name+'.txt', 'a') as the_file:
                for i in flat_HOG_vector:
                    the_file.write(str(i)+"\n")

            # print("Apeending the Final HOG Vector to the Train input Array")
            data_input.append(flat_HOG_vector)
            if data == "pos":
                data_output.append([1])
            else:
                data_output.append([0])

            # helper.show([HOG_image], ["HOG Image"])
            # helper.save([gray_img, gradient_magnitude_img, normalized_gradient_magnitude_img, HOG_image],
            #             ["Gray Image "+name, "Gradient Magnitude "+name, "Normalized Gradient Magnitude "+name, "Hog Gradient image "+name],
            #              path)

        return data_input, data_output

    """ Read the Image """
    train_images_pos, filename_pos = helper.read_images(helper.train_pos_images)
    train_images_neg, filename_neg = helper.read_images(helper.train_neg_images)

    test_images_pos, test_filename_pos = helper.read_images(helper.test_pos_images)
    test_images_neg, test_filename_neg = helper.read_images(helper.test_neg_images)

    # """ Build the Train Input Array with dimensions -> 20 x 7524 """
    pos_train_input, pos_train_output = preprocess(train_images_pos, filename_pos, helper.train_pos_images, data="pos", data_input=[], data_output=[])
    final_train_input, final_train_output = preprocess(train_images_neg, filename_neg, helper.train_neg_images, data="neg", data_input=pos_train_input, data_output=pos_train_output)
    final_train_input = np.array(final_train_input)
    final_train_output = np.array(final_train_output)
    
    final_train_input, final_train_output = helper.unison_shuffled_copies(final_train_input, final_train_output)
    
    post_test_input, pos_test_output = preprocess(test_images_pos, test_filename_pos, helper.test_pos_images, data="pos", data_input=[], data_output=[])
    final_test_input, final_test_output = preprocess(test_images_neg, test_filename_neg, helper.test_neg_images, data="neg", data_input=post_test_input, data_output=pos_test_output)
    final_test_input = np.array(final_test_input)
    final_test_output = np.array(final_test_output)

    # Write the Train and Test data to file
    np.save("train_input.npy", final_train_input)
    np.save("train_output.npy", final_train_output)
    np.save("test_input.npy", final_test_input)
    np.save("test_output.npy", final_test_output)

if __name__ == "__main__":
    main()