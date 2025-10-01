import os
import traceback
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import torchio as tio
from datasets import load_dataset
from src import CACHE_DIR, CT_RATE_DIR, HF_TOKEN, REPO_CT_RATE_ID, logger
from src.utils.downloading import download_file
from src.utils.preprocess import is_leaf_dir, TrainingPreprocessor, InferencePreprocessor
from torch.utils.data import Dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    ScaleIntensityRanged,
    Orientationd,
    Spacingd,
    CastToTyped,
    EnsureChannelFirstd,
)
from src.utils.exceptions import MissingRequiredTagsException


class ExaminationType(Enum):
    DICOM = "DICOM_FOLDER"
    NIFTI = "NIFTI"


def resize_array(array, current_spacing, target_spacing):
    # Calculate new dimensions
    original_shape = array.shape[2:]
    scaling_factors = [
        current_spacing[i] / target_spacing[i] for i in range(len(original_shape))
    ]
    new_shape = [
        int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))
    ]
    # Resize the array
    resized_array = (
        F.interpolate(array, size=new_shape, mode="trilinear", align_corners=False)
        .cpu()
        .numpy()
    )
    return resized_array


def examination_to_tensor(path):
    path = Path(path)
    image = tio.ScalarImage(path)

    if image.data.dtype is not torch.float64:
        affine = image.affine
        tensor_float64 = image.data.to(torch.float64)
        image = tio.ScalarImage(tensor=tensor_float64, affine=affine)

    to_orientation = tio.ToOrientation("SLP")
    image = to_orientation(image)
    affine = image.affine
    img_data = image.data.squeeze(0)

    if "".join(path.suffixes) != ".nii.gz":
        reader = sitk.ImageSeriesReader()
        file_names = reader.GetGDCMSeriesFileNames(path)
        first_file_path = file_names[0]
        file_reader = sitk.ImageFileReader()
        file_reader.SetFileName(first_file_path)
        file_reader.ReadImageInformation()

        intercept = float(file_reader.GetMetaData("0028|1052"))
        slope = float(file_reader.GetMetaData("0028|1053"))
    else:
        slope = 1
        intercept = 0

    target_x_spacing = 0.75
    target_y_spacing = 0.75
    target_z_spacing = 1.5

    current = image.spacing
    target = (target_z_spacing, target_x_spacing, target_y_spacing)

    img_data = slope * img_data + intercept  # USING ONLY FOR v1 DATA
    # img_data = 1 * img_data + 0
    hu_min, hu_max = -1000, 1000
    img_data = np.clip(img_data, hu_min, hu_max)

    img_data = img_data.unsqueeze(0).unsqueeze(0)

    img_data = resize_array(img_data, current, target)
    img_data = img_data[0][0]
    img_data = np.transpose(img_data, (1, 2, 0))

    img_data = (((img_data) / 1000)).astype(np.float32)

    tensor = torch.tensor(img_data)
    # Get the dimensions of the input tensor
    target_shape = (480, 480, 240)

    # Extract dimensions
    h, w, d = tensor.shape

    # Calculate cropping/padding values
    dh, dw, dd = target_shape
    h_start = max((h - dh) // 2, 0)
    h_end = min(h_start + dh, h)
    w_start = max((w - dw) // 2, 0)
    w_end = min(w_start + dw, w)
    d_start = max((d - dd) // 2, 0)
    d_end = min(d_start + dd, d)

    # Crop
    tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

    # Pad
    pad_h_before = (dh - tensor.size(0)) // 2
    pad_h_after = dh - tensor.size(0) - pad_h_before
    pad_w_before = (dw - tensor.size(1)) // 2
    pad_w_after = dw - tensor.size(1) - pad_w_before
    pad_d_before = (dd - tensor.size(2)) // 2
    pad_d_after = dd - tensor.size(2) - pad_d_before

    tensor = torch.nn.functional.pad(
        tensor,
        (
            pad_d_before,
            pad_d_after,
            pad_w_before,
            pad_w_after,
            pad_h_before,
            pad_h_after,
        ),
        value=-1,
    )

    tensor = tensor.permute(2, 0, 1)
    tensor = tensor.unsqueeze(0)

    return tensor


def training_examination_to_tensor(data_path, return_study_id=False):
    """Функция для предобработки исследований

    Args:
        data_path (str): Путь к исследованию

    Returns:
        Tensor: предоработанное исследование
    """

    # Предупреждение об ошибке записи тега в DICOM файле, не влияет на результат
    data_path = Path(data_path)

    study_type = (
        ExaminationType.NIFTI
        if ".nii" in Path(data_path).suffixes
        else ExaminationType.DICOM
    )

    with TemporaryDirectory() as archive_temp_dir, TemporaryDirectory() as thinnest_series_dir:
        if study_type is ExaminationType.DICOM:
            logger.info(f"Обработка DICOM исследования")
            if data_path.is_file() and str(data_path).endswith(
                (".zip", ".tar", ".tgz", ".tar.gz")
            ):
                data_path = TrainingPreprocessor.extract_archive_to_dir(data_path, archive_temp_dir)

            dicom_series = TrainingPreprocessor.read_ct_series(data_path)
            if len(dicom_series.keys()) >= 1:
                thinnest_series_paths = (
                    TrainingPreprocessor.get_thinnest_slice_series_with_windows_paths_(dicom_series)
                )
                TrainingPreprocessor.copy_files_to_directory(thinnest_series_paths, thinnest_series_dir)
                data_path = thinnest_series_dir
            study_id, series_id = TrainingPreprocessor.get_study_id_and_series_id(data_path)
        else:
            logger.info(f"Обработка NIFTI исследования")

        logger.info(f"Чтение финальной серии из {data_path}")
        image = tio.ScalarImage(data_path)

        if image.data.dtype is not torch.float64:
            affine = image.affine
            tensor_float64 = image.data.to(torch.float64)
            image = tio.ScalarImage(tensor=tensor_float64, affine=affine)

    ### ---(SLOPE / INTERCEPT) CALCULATION---

    # if study_type is DICOM:
    #     image_reader = sitk.ImageSeriesReader()
    #     file_reader = sitk.ImageFileReader()
    #     file_names = image_reader.GetGDCMSeriesFileNames(data_path)
    #     first_file_path = file_names[0]
    #     file_reader.SetFileName(first_file_path)
    #     file_reader.ReadImageInformation()

    #     intercept = float(file_reader.GetMetaData("0028|1052"))
    #     slope = float(file_reader.GetMetaData("0028|1053"))
    # else:
    #     slope = 1
    #     intercept = 0

    to_orientation = tio.ToOrientation("SLP")
    image = to_orientation(image)
    affine = image.affine

    current = image.spacing
    img_data = image.data.squeeze(0)

    target_x_spacing = 0.75
    target_y_spacing = 0.75
    target_z_spacing = 1.5

    target = (target_z_spacing, target_x_spacing, target_y_spacing)

    # img_data = slope * img_data + intercept

    img_data = 1 * img_data + 0
    hu_min, hu_max = -1000, 1000
    # print(img_data.min(), img_data.max())
    img_data = np.clip(img_data, hu_min, hu_max)

    img_data = img_data.unsqueeze(0).unsqueeze(0)

    img_data = resize_array(img_data, current, target)
    img_data = img_data[0][0]
    img_data = np.transpose(img_data, (1, 2, 0))

    img_data = (((img_data) / 1000)).astype(np.float32)

    tensor = torch.tensor(img_data)
    # Get the dimensions of the input tensor
    target_shape = (480, 480, 240)

    # Extract dimensions
    h, w, d = tensor.shape

    # Calculate cropping/padding values
    dh, dw, dd = target_shape
    h_start = max((h - dh) // 2, 0)
    h_end = min(h_start + dh, h)
    w_start = max((w - dw) // 2, 0)
    w_end = min(w_start + dw, w)
    d_start = max((d - dd) // 2, 0)
    d_end = min(d_start + dd, d)

    # Crop
    tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

    # Pad
    pad_h_before = (dh - tensor.size(0)) // 2
    pad_h_after = dh - tensor.size(0) - pad_h_before
    pad_w_before = (dw - tensor.size(1)) // 2
    pad_w_after = dw - tensor.size(1) - pad_w_before
    pad_d_before = (dd - tensor.size(2)) // 2
    pad_d_after = dd - tensor.size(2) - pad_d_before

    tensor = torch.nn.functional.pad(
        tensor,
        (
            pad_d_before,
            pad_d_after,
            pad_w_before,
            pad_w_after,
            pad_h_before,
            pad_h_after,
        ),
        value=-1,
    )

    tensor = tensor.permute(2, 0, 1)
    tensor = tensor.unsqueeze(0)

    if return_study_id:
        return tensor, study_id, series_id
    else:
        return tensor


def inference_examination_to_tensor(data_path):
    """Функция для предобработки исследований

    Args:
        data_path (str): Путь к исследованию

    Returns:
        Tensor: предоработанное исследование
    """
    data_path = Path(data_path)

    processor = InferencePreprocessor()

    study_type = (
        ExaminationType.NIFTI
        if ".nii" in Path(data_path).suffixes
        else ExaminationType.DICOM
    )

    with TemporaryDirectory() as archive_temp_dir, TemporaryDirectory() as thinnest_series_dir:
        if study_type is ExaminationType.DICOM:
            logger.info(f"Обработка DICOM исследования")
            if data_path.is_file() and str(data_path).endswith(
                (".zip", ".tar", ".tgz", ".tar.gz")
            ):
                data_path = processor.extract_archive_to_dir(data_path, archive_temp_dir)

            dicom_series = processor.get_valid_series(data_path)
            if len(dicom_series.keys()) >= 1:
                thinnest_series_paths = (
                    processor.get_thinnest_slice_series_with_windows_paths_(dicom_series)
                )
                processor.copy_files_to_directory(thinnest_series_paths, thinnest_series_dir)
                data_path = thinnest_series_dir
            study_id, series_id = processor.get_study_id_and_series_id(data_path)
        else:
            logger.info(f"Обработка NIFTI исследования")

        try:
            logger.info(f"Чтение финальной серии из {data_path}")
            image = tio.ScalarImage(data_path)

            if image.data.dtype is not torch.float64:
                affine = image.affine
                tensor_float64 = image.data.to(torch.float64)
                image = tio.ScalarImage(tensor=tensor_float64, affine=affine)
        except Exception:
            raise MissingRequiredTagsException()

    ### ---(SLOPE / INTERCEPT) CALCULATION---

    # if study_type is DICOM:
    #     image_reader = sitk.ImageSeriesReader()
    #     file_reader = sitk.ImageFileReader()
    #     file_names = image_reader.GetGDCMSeriesFileNames(data_path)
    #     first_file_path = file_names[0]
    #     file_reader.SetFileName(first_file_path)
    #     file_reader.ReadImageInformation()

    #     intercept = float(file_reader.GetMetaData("0028|1052"))
    #     slope = float(file_reader.GetMetaData("0028|1053"))
    # else:
    #     slope = 1
    #     intercept = 0

    to_orientation = tio.ToOrientation("SLP")
    image = to_orientation(image)
    affine = image.affine

    current = image.spacing
    img_data = image.data.squeeze(0)

    target_x_spacing = 0.75
    target_y_spacing = 0.75
    target_z_spacing = 1.5

    target = (target_z_spacing, target_x_spacing, target_y_spacing)

    # img_data = slope * img_data + intercept

    img_data = 1 * img_data + 0
    hu_min, hu_max = -1000, 1000
    # print(img_data.min(), img_data.max())
    img_data = np.clip(img_data, hu_min, hu_max)

    img_data = img_data.unsqueeze(0).unsqueeze(0)

    img_data = resize_array(img_data, current, target)
    img_data = img_data[0][0]
    img_data = np.transpose(img_data, (1, 2, 0))

    img_data = (((img_data) / 1000)).astype(np.float32)

    tensor = torch.tensor(img_data)
    # Get the dimensions of the input tensor
    target_shape = (480, 480, 240)

    # Extract dimensions
    h, w, d = tensor.shape

    # Calculate cropping/padding values
    dh, dw, dd = target_shape
    h_start = max((h - dh) // 2, 0)
    h_end = min(h_start + dh, h)
    w_start = max((w - dw) // 2, 0)
    w_end = min(w_start + dw, w)
    d_start = max((d - dd) // 2, 0)
    d_end = min(d_start + dd, d)

    # Crop
    tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

    # Pad
    pad_h_before = (dh - tensor.size(0)) // 2
    pad_h_after = dh - tensor.size(0) - pad_h_before
    pad_w_before = (dw - tensor.size(1)) // 2
    pad_w_after = dw - tensor.size(1) - pad_w_before
    pad_d_before = (dd - tensor.size(2)) // 2
    pad_d_after = dd - tensor.size(2) - pad_d_before

    tensor = torch.nn.functional.pad(
        tensor,
        (
            pad_d_before,
            pad_d_after,
            pad_w_before,
            pad_w_after,
            pad_h_before,
            pad_h_after,
        ),
        value=-1,
    )

    tensor = tensor.permute(2, 0, 1)
    tensor = tensor.unsqueeze(0)

    return {
        "tensor": tensor,
        "study_id": study_id,
        "series_id": series_id,
        "warnings": set(processor.warnings)
    }




def examination_to_tensor_vista3d(data_path):
    resample_to_spacing = (1.5, 1.5, 1.5)

    monai_transforms = Compose(
        [
            LoadImaged(
                keys=["image"],
                image_only=True,
            ),
            EnsureChannelFirstd(
                keys=["image"],
            ),
            ScaleIntensityRanged(
                keys="image",
                a_min=-963.8247715525971,
                a_max=1053.678477684517,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(
                keys=["image"],
                pixdim=resample_to_spacing,
                mode=["bilinear"],
                align_corners=True,
            ),
            CastToTyped(keys=["image"], dtype=[torch.float32]),
        ]
    )
    tensor = monai_transforms({"image": data_path})
    return tensor["image"].as_tensor()


class CTRATECachingDataset(Dataset):
    def __init__(self, indexes, downloaded_folder, saving_folders):
        super().__init__()

        self.split = "train" if "train" in downloaded_folder else "validation"
        self.downloaded_folder = downloaded_folder
        self.saving_folders = saving_folders

        # Load dataset index
        self.ds = load_dataset(
            REPO_CT_RATE_ID,
            "labels",
            token=HF_TOKEN,
            cache_dir=CACHE_DIR,
        )[self.split].to_pandas()
        indexes = list(filter(lambda x: x < len(self.ds), indexes))
        self.ds = self.ds.iloc[indexes]

        # Load metadata for slope/intercept etc
        meta_file = (
            "train_metadata.csv" if self.split == "train" else "validation_metadata.csv"
        )
        meta_file = CT_RATE_DIR / "dataset" / "metadata" / meta_file
        self.meta_df = pd.read_csv(meta_file)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        try:
            # change split
            row = self.ds.iloc[idx]
            vol_name = row["VolumeName"]

            # Derive HuggingFace subfolder structure
            folder1, folder2, folder3 = vol_name.split("_")[:3]
            folder = f"{folder1}_{folder2}"
            subfolder = f"dataset/{self.downloaded_folder}/{folder}/{folder}_{folder3}"

            save_roots = [
                CT_RATE_DIR / "dataset" / folder for folder in self.saving_folders
            ]
            save_paths = [
                save_root
                / f"{folder}/{folder}_{folder3}"
                / (vol_name.replace(".gz", ".pt"))
                for save_root in save_roots
            ]

            if all(os.path.exists(save_path) for save_path in save_paths):
                logger.success(f"all tensors for {str(vol_name)} are exists")
                return [], [], str(CT_RATE_DIR / subfolder / vol_name), vol_name

            # Download the volume file
            logger.info(f"Downloading {vol_name}")
            local_file = download_file(subfolder, vol_name, CT_RATE_DIR)
            logger.info(f"Downloaded {vol_name}")

            # Load volume into tensor
            logger.info(f"Preprocessing {vol_name} with CT_CLIP")
            tensor = examination_to_tensor(local_file)
            logger.info(f"Preprocessing {vol_name} with Vista3D")
            tensor_vista3d = examination_to_tensor_vista3d(local_file)
            logger.info(f"Preprocessed {vol_name}")

            return tensor, tensor_vista3d, str(local_file), vol_name
        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error(e)
            logger.error(local_file)
            return [], [], str(local_file), vol_name


class MosMedDataCachingDataset(Dataset):
    def __init__(self, root_folder, folders, saving_folders):
        super().__init__()

        self.root_folder = Path(root_folder)
        self.folders = [self.root_folder / folder for folder in folders]
        self.objects = self.find_all_data_objects()
        self.saving_folders = saving_folders

    def find_all_data_objects(self):
        """
        Возвращает список путей к файлам, если они имеют формат .tar.gz или .nii.gz,
        либо путь к папке, если у нее все дочерние элементы — только файлы (папка последнего уровня).
        """
        result = []

        def _get_all_archives(folder):
            all_archives = []
            for path in folder.rglob("*"):
                if path.is_file():
                    if str(path).endswith(
                        (".tar", ".tgz", ".tar.gz", ".nii", ".nii.gz", ".zip")
                    ):
                        all_archives.append(path)
            return all_archives

        def _get_all_folders(folder, need_to_be_last_level=False):
            all_folders = []
            find_paths = (
                folder.rglob("*") if need_to_be_last_level else folder.glob("*")
            )
            for path in find_paths:
                if path.is_dir():
                    if need_to_be_last_level:
                        if is_leaf_dir(path):
                            all_folders.append(path)
                    else:
                        all_folders.append(path)
            return all_folders

        for folder_path in self.folders:
            if not folder_path.exists():
                continue

            try:
                if folder_path.name == "MosMedData-LDCT-LUNGCR-type I-v 1":
                    examinations = _get_all_folders(
                        folder_path / "studies", need_to_be_last_level=False
                    )
                elif folder_path.name == "MosMedData-CT-COVID19-type VII-v 1":
                    examinations = _get_all_folders(
                        folder_path / "dicom", need_to_be_last_level=True
                    )
                elif folder_path.name == "MosMedData-CT-COVID19-type I-v 4":
                    examinations = _get_all_folders(
                        folder_path / "studies", need_to_be_last_level=False
                    )
                elif folder_path.name == "COVID19_1110":
                    examinations = _get_all_archives(folder_path / "studies")
                elif folder_path.name == "CT_LUNGCANCER_500":
                    examinations = _get_all_archives(folder_path / "dicom")
                result.extend(examinations)
                logger.info(f"Found {len(examinations)} examinations in {folder_path}")
            except Exception as e:
                logger.error(traceback.format_exc())
                logger.error(
                    "Пропускаем папку " + str(folder_path) + " из-за ошибки " + str(e)
                )
                continue

        return result

    def __len__(self):
        return len(self.objects)

    def __getitem__(self, idx):
        try:
            # change split
            examination_path = self.objects[idx]

            save_roots = [
                self.root_folder / "embeddings" / folder
                for folder in self.saving_folders
            ]
            save_paths = [
                save_root
                / examination_path.relative_to(self.root_folder).with_suffix(
                    examination_path.suffix + ".pt"
                )
                for save_root in save_roots
            ]

            if all(os.path.exists(save_path) for save_path in save_paths):
                logger.success(f"all tensors for {examination_path.name} are exists")
                return [], [], str(examination_path), examination_path.name

            # Load volume into tensor
            logger.info(f"Preprocessing {examination_path} with CT_CLIP")
            tensor = examination_to_tensor_updated(examination_path)
            logger.info(f"Preprocessing {examination_path} with Vista3D")
            tensor_vista3d = examination_to_tensor_vista3d(examination_path)
            logger.info(f"Preprocessed {examination_path}")

            return tensor, tensor_vista3d, str(examination_path), examination_path.name
        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error(e)
            logger.error(examination_path)
            return [], [], str(examination_path), examination_path.name
