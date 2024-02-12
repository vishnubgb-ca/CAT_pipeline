from PIL import Image
import numpy as np
import random
import os 
import requests
import zipfile
from pathlib import Path
import numpy as np
import nibabel as nib
from skimage import io
import SimpleITK as sitk
from matplotlib.pyplot import plot 
import matplotlib.pyplot as plt
from data_extract import extract_data
from io import BytesIO

def open_random_images(path):
    # Get a list of all files in the folder
    all_files = os.listdir(path)
    random.shuffle(all_files)
    image_names = all_files[:7]
    image_paths = []
    for i in range(7):
        image_path = os.path.join(path, image_names[i])
        image_paths.append(image_path)
    return image_paths




# def to_uint8(data):
#     data -= data.min()
#     data /= data.max()
#     data *= 255
#     return data.astype(np.uint8)


# def nii_to_jpgs(input_path, output_dir, rgb=False):
#     output_dir = Path(output_dir)
#     data = nib.load(input_path).get_fdata()
#     *_, num_slices, num_channels = data.shape
#     for channel in range(num_channels):
#         volume = data[..., channel]
#         volume = to_uint8(volume)
#         channel_dir = output_dir / f'channel_{channel}'
#         channel_dir.mkdir(exist_ok=True, parents=True)
#         for slice in range(num_slices):
#             slice_data = volume[..., slice]
#             if rgb:
#                 slice_data = np.stack(3 * [slice_data], axis=2)
#             output_path = channel_dir / f'channel_{channel}_slice_{slice}.jpg'
#             io.imsave(output_path, slice_data)


def openimage():
    try:
        url = extract_data()
        url_response = requests.get(url)
        url_response.raise_for_status()  # Raise an error for bad responses
        with zipfile.ZipFile(BytesIO(url_response.content)) as z:
            for member in z.namelist():
                # Extract file to current directory
                z.extract(member, path='.')
                # Check if the extracted file already exists
                if os.path.exists(member):
                    print(f"File '{member}' already exists. Skipping extraction.")
                else:
                    print(f"Extracted '{member}'.")
        print("Data extraction successful.")
    except requests.exceptions.RequestException as e:
        print("Error downloading data:", e)
    except zipfile.BadZipFile:
        print("The downloaded file is not a valid zip file.")
    except Exception as e:
        print("An unexpected error occurred:", e)
    
    path = os.path.join(os.getcwd(),"datasample/datasample/images")
    x = open_random_images(path)
    count = 0
    for file in x:
        filearr = file.split("/")[-1]
        filename = filearr.split(".")[0]
        # Read the .nii image containing the volume with SimpleITK
        # sitk_t1 = sitk.ReadImage(file)
        # # and access the numpy array:
        # t1 = sitk.GetArrayFromImage(sitk_t1)
        # print(t1.shape)
        # for y in range(len(t1[0])):
        #     im = Image.fromarray((t1[y,:,:] ).astype(np.uint8))
        #     print(im)
        
        # Load the .nii.gz file
        img = nib.load(file)
        # Get the image data
        img_data = img.get_fdata()
        # Display one of the image slices (you can change the slice index)
        slice_index = img_data.shape[2] // 2  # Choose a slice index in the z-axis
        
        slice_image = img_data[:, :, slice_index]

        # Normalize the image data to the range [0, 255]
        slice_image = (slice_image - slice_image.min()) / (slice_image.max() - slice_image.min()) * 255

        # Convert the image data to an unsigned 8-bit integer array
        slice_image = slice_image.astype('uint8')
        img_png = Image.fromarray(slice_image)
        img_png.save(f'{filename}.png')
        count=count+1
        plt.imshow(img_data[:, :, slice_index], cmap='gray')
        plt.axis('off')  # Turn off axis
        plt.show()


openimage()
