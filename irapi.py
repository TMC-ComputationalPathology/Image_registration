import os
import cv2
import shutil
from fastapi import FastAPI, HTTPException
from starlette.responses import StreamingResponse
from io import BytesIO
from zipfile import ZipFile
from pma_python import core
from urllib.parse import unquote
import numpy as np
import uvicorn






app = FastAPI(
    title="KS 17/COMPUTATIONAL PATHOLOGY & AI LAB ACTREC",
    version="0.0.0",
    openapi_url="/ACTREC",
    docs_url="/registration",
    redoc_url="/redoc",
)
app.secret_key = 'HSTRInF234'


def tile_wise_registration(reference_image, target_image,n, tile_size=(256, 256)):
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

    return registered_image, transform_matrix, tx, ty,rotation_deg

@app.post("/registration/")
async def download_files(ihc_file_path: str, he_file_path: str, X: int, Y: int, patch_size: int):
    ihc_file_path = unquote(ihc_file_path)
    he_file_path = unquote(he_file_path)

    sessionID = core.connect('http://10.100.76.62/core/', 'tmccomputpath', 'Z5XPLQ2I')

    folder_name_prefix = str(he_file_path.split('/')[-2])

    temp_dir = folder_name_prefix
    os.makedirs(temp_dir, exist_ok=True)

    ihc_output_path = os.path.join(temp_dir, "ihc.tiff")
    core.get_region(ihc_file_path, x=X, y=Y, width=8192, height=8192, scale=1, rotation=0, sessionID=sessionID).save(ihc_output_path)

    he_output_path = os.path.join(temp_dir, "he.tiff")
    core.get_region(he_file_path, x=X, y=Y, width=8192, height=8192, scale=1, rotation=0, sessionID=sessionID).save(he_output_path)

    reference_image = cv2.imread(he_output_path)
    target_image = cv2.imread(ihc_output_path)

    registered_image, transformation_matrix, tx, ty, rotation_deg = tile_wise_registration(reference_image, target_image, 50000)

    registered_image_path = os.path.join(temp_dir, "registered_X.tiff")
    cv2.imwrite(registered_image_path, registered_image)

    xnew = X + tx
    ynew = Y + ty
    r = rotation_deg
    im = core.get_region(ihc_file_path, x=xnew, y=ynew, width=8192, height=8192, scale=1, rotation=r, sessionID=sessionID)

    shifted_output_path = os.path.join(temp_dir, "shifted_target.tiff")
    im.save(shifted_output_path)

    image = cv2.imread(shifted_output_path, cv2.IMREAD_UNCHANGED)
    height, width = image.shape[:2]
    save_dir_ihc = os.path.join(temp_dir, 'patches_ihc')
    save_dir_he = os.path.join(temp_dir, 'patches_he')
    os.makedirs(save_dir_ihc, exist_ok=True)
    os.makedirs(save_dir_he, exist_ok=True)
    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            patch_ihc = image[y:y + patch_size, x:x + patch_size]
            patch_filename_ihc = os.path.join(save_dir_ihc, f'patch_{x}_{y}.png')
            cv2.imwrite(patch_filename_ihc, patch_ihc)

            patch_he = reference_image[y:y + patch_size, x:x + patch_size]
            patch_filename_he = os.path.join(save_dir_he, f'patch_{x+X}_{y+Y}.png')
            cv2.imwrite(patch_filename_he, patch_he)

    zip_filename = f"{temp_dir}.zip"
    with ZipFile(zip_filename, 'w') as zip:
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                zip.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), temp_dir))

    confirmation_message = "Processing completed successfully!"

    with open(zip_filename, "rb") as file:
        zip_file = BytesIO(file.read())

    response = StreamingResponse(zip_file, media_type="application/octet-stream")
    response.headers["Content-Disposition"] = f"attachment; filename={os.path.basename(zip_filename)}"
    response.headers["X-Confirmation-Message"] = confirmation_message  # Add custom header for confirmation message

    os.remove(zip_filename) 

    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
