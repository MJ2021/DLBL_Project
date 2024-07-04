import cv2
import numpy as np
import csv
import openslide as op
import glob
from PIL import Image, ImageDraw, ImageOps
import pandas as pd

wsi= op.OpenSlide('/home/Drivessd2tb/dlbl_combined/013708CN__20231221_104125_(3).tiff')
dim=wsi.dimensions
lvl=wsi.level_dimensions
print('dim=',dim)
print('lvl=',lvl)

# Read coordinates from CSV file
def read_coordinates_from_csv(csv_file):
    coordinates = []
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row if present
        for row in reader:
            try:
                x, y = map(int, row)  # Convert values to integers
                coordinates.append((x, y))
            except ValueError:
                print("Skipping invalid row:", row)
    return coordinates

# Create a mask image from coordinates
def create_mask_image(df, image_size):
    # mask = Image.new('L', image_size, 0)  # Create blank mask image
    # draw = ImageDraw.Draw(mask)
    mask = np.zeros((image_size[0], image_size[1]), dtype = np.uint8)
    print(mask)
    print(mask.shape)
    # draw.polygon(coordinates, fill=255)  # Draw filled polygon using coordinates
    for index, row in df.iterrows():
        x, y = int(row['x']/512), int(row['y']/512)
        if 0 <= x < image_size[0] and 0 <= y < image_size[1]:  # Ensure coordinates are within mask bounds
            mask[x, y] = 1
    return mask

# Read coordinates from CSV file
df = pd.read_csv('/home/Drivessd2tb/Mohit_Combined/csv_files_with_white_filter_from_patches_combined/013708CN__20231221_104125_(3).csv')
coordinates = read_coordinates_from_csv('/home/Drivessd2tb/Mohit_Combined/csv_files_with_white_filter_from_patches_combined/013708CN__20231221_104125_(3).csv')  # Change 'coordinates.csv' to your CSV file path

# Create a mask image
size = (int(dim[0]/512), int(dim[1]/512))
# mask = create_mask_image(coordinates, size)
mask = create_mask_image(df, size)
mask = np.rot90(mask, 3)
print(mask)
mask_image = Image.fromarray(mask * 255)  # Multiply by 255 to convert binary mask to grayscale image
mask_image = ImageOps.mirror(mask_image)

mask = mask_image.save('/home/ravi/Mohit/CLAM/013708CN__20231221_104125_(3).png')