import os
import numpy as np
import cv2
import argparse
from pma_python import core


def tile_wise_registration(reference_image, target_image, n, tile_size=(256, 256)):
    # Convert images to grayscale
    reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=n)

    # Find keypoints and descriptors with ORB
    keypoints1, descriptors1 = orb.detectAndCompute(reference_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(target_gray, None)

    # Create BFMatcher (Brute Force Matcher) object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(descriptors1, descriptors2)

    # Sort them in ascending order of distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract matched points
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Estimate affine transformation
    transform_matrix = cv2.estimateAffinePartial2D(src_pts, dst_pts)[0]

    # Extract translation and rotation parameters
    tx = transform_matrix[0, 2]
    ty = transform_matrix[1, 2]
    rotation_rad = np.arctan2(transform_matrix[1, 0], transform_matrix[0, 0])

    # Convert rotation from radians to degrees
    rotation_deg = np.degrees(rotation_rad)

    # Apply the transformation to the target image
    registered_image = cv2.warpAffine(reference_image, transform_matrix, (target_image.shape[1], target_image.shape[0]))

    return registered_image, transform_matrix, tx, ty, rotation_deg


def main(ihc_path, he_path, X, Y,patch_size):
    print("'ihc' (string):", ihc_path)
    print("'he' (string):", he_path)
    print("'X' (number):", X)
    print("'Y' (number):", Y)


    sessionID = core.connect('http://10.100.76.62/core/', 'tmccomputpath', 'Z5XPLQ2I')

    folder_name_prefix = os.path.splitext(os.path.basename(he_path))[0]
    folder_path = f"{folder_name_prefix}_region_{X}_{Y}"
    os.makedirs(folder_path, exist_ok=True)
    print('folder_created_for_patches_with_name',folder_path)

    ihc_8192 = core.get_region(ihc_path, x=X, y=Y, width=8192, height=8192, scale=1, rotation=0, sessionID=sessionID)
    output_path = os.path.join(folder_path, "ihc.tiff")
    ihc_8192.save(output_path)
    he_8192 = core.get_region(he_path, x=X, y=Y, width=8192, height=8192, scale=1, rotation=0, sessionID=sessionID)
    output_path = os.path.join(folder_path, "he.tiff")
    he_8192.save(output_path)

    he = os.path.join(folder_path, 'he.tiff')
    ihc = os.path.join(folder_path, 'ihc.tiff')

    target_image = cv2.imread(ihc)
    reference_image = cv2.imread(he)

    registered_image, transformation_matrix, tx, ty, rotation_deg = tile_wise_registration(reference_image,
                                                                                             target_image, 50000)
    registered_image_path = os.path.join(folder_path, "registered_X.tiff")
    cv2.imwrite(registered_image_path, registered_image)
    print('registered_shifts:',transformation_matrix,'x:', tx,'y:', ty,'rotation:', rotation_deg)
    print("shifting coordinates and accounting rotation")

    xnew = X + tx
    ynew = Y + ty
    r = rotation_deg
    print('xnew:', xnew, 'ynew:', ynew, 'rotation', r)

    im = core.get_region(ihc_path, x=xnew, y=ynew, width=8192, height=8192, scale=1, rotation=r, sessionID=sessionID)

    shifted_output_path = os.path.join(folder_path, "shifted_target.tiff")
    im.save(shifted_output_path)
    print('saving patches.....')

    image = cv2.imread(shifted_output_path, cv2.IMREAD_UNCHANGED)
    height, width = image.shape[:2]
    patch_size = patch_size
    save_dir = os.path.join(folder_path, 'patches_ihc')
    os.makedirs(save_dir, exist_ok=True)
    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            patch = image[y:y + patch_size, x:x + patch_size]
            patch_filename = os.path.join(save_dir, f'patch_{x}_{y}.png')
            cv2.imwrite(patch_filename, patch)

    img = cv2.imread(he, cv2.IMREAD_UNCHANGED)
    height, width = img.shape[:2]
    patch_size = patch_size
    save_dir = os.path.join(folder_path, 'patches_he')
    os.makedirs(save_dir, exist_ok=True)
    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            patch = img[y:y + patch_size, x:x + patch_size]
            patch_filename = os.path.join(save_dir, f'patch_{x+X}_{y+Y}.png')
            cv2.imwrite(patch_filename, patch)

    print('registered patches saved successfully')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process inputs.")
    parser.add_argument("--ihc_file_path", type=str, help="Path to the IHC file")
    parser.add_argument("--he_file_path", type=str, help="Path to the HE file")
    parser.add_argument("--X", type=int, help="Coordinate for 'X'")
    parser.add_argument("--Y", type=int, help="Coordinate for 'Y'")
    parser.add_argument("--patch_size", type = int ,help = "patch_size")
    args = parser.parse_args()

    main(args.ihc_file_path, args.he_file_path, args.X, args.Y,args.patch_size)
