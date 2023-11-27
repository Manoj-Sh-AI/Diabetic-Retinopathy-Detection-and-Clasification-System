from PIL import Image
import os

def resize_images(input_folder, output_folder, target_size=(512, 340)):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.endswith('.tif')]

    for image_file in image_files:
        # Open the image
        input_path = os.path.join(input_folder, image_file)
        img = Image.open(input_path)

        # Resize the image using LANCZOS filter
        img_resized = img.resize(target_size, Image.LANCZOS)

        # Save the resized image to the output folder
        output_path = os.path.join(output_folder, image_file)
        img_resized.save(output_path)

if __name__ == "__main__":
    # Set the input and output folders
    input_folder = "sample input/"
    output_folder = "sample output/"

    # Set the target size for resizing
    target_size = (512, 340)

    # Resize images in the input folder and save them to the output folder
    resize_images(input_folder, output_folder, target_size)
