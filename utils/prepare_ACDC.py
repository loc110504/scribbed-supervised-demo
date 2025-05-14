import os
import h5py
import nibabel as nib
import numpy as np

def main():
    slices_dir   = "../datasets/ACDC/ACDC_training_slices"
    scribble_dir = "..acdc_scribbles"
    out_dir      = "../datasets/ACDC/ACDC_training_scribble_slices"
    os.makedirs(out_dir, exist_ok=True)

    for fname in sorted(os.listdir(slices_dir)):
        if not fname.endswith(".h5"):
            continue

        slice_path = os.path.join(slices_dir, fname)
        scribble_name = fname.replace(".h5", "_scribble.nii")
        scribble_path = os.path.join(scribble_dir, scribble_name)
        if not os.path.exists(scribble_path):
            print(f"[WARN] Không tìm thấy scribble cho {fname}, bỏ qua.")
            continue

        # 1) Load image & label
        with h5py.File(slice_path, "r") as f:
            image = f["image"][:]
            label = f["label"][:]

        # 2) Load scribble từ NIfTI
        nii    = nib.load(scribble_path)
        scrib  = nii.get_fdata().astype(np.uint8)

        # 3) Ghi HDF5 mới
        out_path = os.path.join(out_dir, fname)
        with h5py.File(out_path, "w") as f:
            f.create_dataset("image",    data=image, compression="gzip")
            f.create_dataset("label",    data=label, compression="gzip")
            f.create_dataset("scribble", data=scrib, compression="gzip")

        print(f"[INFO] Đã tạo {out_path}")

    print(">> Hoàn tất chuẩn bị thư mục ACDC_training_scribble_slices.")

if __name__ == "__main__":
    main()
