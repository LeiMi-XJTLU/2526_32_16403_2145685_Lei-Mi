import os
import glob
import numpy as np
import nibabel as nib
from skimage.transform import resize

def main():
    # Root directory of the original TCIA dataset
    data_dir = "tcia_data"
    
    # 1. Load all segmentation files and extract patient IDs
    seg_paths = sorted(glob.glob(os.path.join(data_dir, "*_seg.nii.gz")))
    seg_ids = [os.path.basename(p).split("_")[0] for p in seg_paths]
    print(f"Found {len(seg_paths)} labeled patients")

    # 2. Match T1w images for patients who have corresponding labels
    img_paths = []
    for patient_id in seg_ids:
        pattern = os.path.join(data_dir, f"{patient_id}*T1w.nii.gz")
        patient_images = glob.glob(pattern)
        if patient_images:
            img_paths.append(patient_images[0])

    # Sort to ensure one-to-one correspondence
    img_paths = sorted(img_paths)
    seg_paths = sorted(seg_paths)
    print(f"Matched {len(img_paths)} image-label pairs")

    images = []
    labels = []

    # Process each paired image and label
    for img_path, seg_path in zip(img_paths, seg_paths):
        print(f"Processing: {os.path.basename(img_path)} <-> {os.path.basename(seg_path)}")
        
        # Load 3D NIfTI data
        img = nib.load(img_path).get_fdata()
        seg = nib.load(seg_path).get_fdata()

        # Extract the middle slice to convert 3D volume to 2D image
        z_mid = img.shape[-1] // 2
        img = img[..., z_mid]
        seg = seg[..., z_mid]

        # Resize image and label to 224x224
        img = resize(img, (224, 224), anti_aliasing=True, preserve_range=True)
        seg = resize(seg, (224, 224), order=0, preserve_range=True)

        # Normalize image intensity to [0, 1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        # Convert to 3 channels for TransUNet input
        img = np.stack([img, img, img], axis=-1)

        # Binarize segmentation label
        seg = (seg > 0).astype(np.float32)

        # Append processed data to lists
        images.append(img)
        labels.append(seg)

    # Create output directory if it does not exist
    os.makedirs("processed_data", exist_ok=True)

    # Save processed data as .npy files for training
    np.save("processed_data/images.npy", np.array(images))
    np.save("processed_data/labels.npy", np.array(labels))
    
    print(" Preprocessing finished!")
    print(f"Final data shape: images={np.array(images).shape}, labels={np.array(labels).shape}")

if __name__ == "__main__":
    main()
