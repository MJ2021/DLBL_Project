import sys
import pyvips

# for filename in sys.argv[2:]:
image = pyvips.Image.new_from_file('/home/Drivessd2tb/dlbl_data/Mohit/TEST_IMAGE/003128CN__20240214_131534.tiff')
image.tiffsave('/home/Drivessd2tb/dlbl_data/Mohit/TEST_IMAGE/003128CN__20240214_131534_pyramid.tiff', 
        compression="jpeg", 
        Q=50, 
        tile=True, 
        tile_width=256, 
        tile_height=256, 
        pyramid=True)