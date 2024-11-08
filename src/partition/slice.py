import cv2
import os


slice_row = 2
slice_col = 2

def slice_image(image_path, output_dir, overlap=100):

    image = cv2.imread(image_path)
    image_height, image_width, _ = image.shape
    
    slice_width = image_width // slice_row
    slice_height = image_height // slice_row

    file_name, ext = os.path.splitext(os.path.basename(image_path))
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    count = 1
    for i in range(slice_row):
        for j in range(slice_col):
            left = j * slice_width
            upper = i * slice_height
            right = min(left + slice_width + overlap, image_width)
            lower = min(upper + slice_height + overlap, image_height)

            slice_img = image[upper:lower, left:right]
            cv2.imwrite(os.path.join(output_dir, f"{file_name}_{count}{ext}"), slice_img)
            count += 1

def slice_images_in_dir(input_dir, output_dir, overlap=100):
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".jpg"):
            image_path = os.path.join(input_dir, file_name)
            slice_image(image_path, output_dir, overlap)

if __name__ == "__main__":
    image_path = "/home/edge/work/datasets/PANDA_dataset/images/val"
    output_dir = "/home/edge/work/datasets/PANDA_dataset/images/sliced_val"

    slice_images_in_dir(image_path, output_dir)
