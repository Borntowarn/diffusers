from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from src import (CACHE_DIR, CT_RATE_DIR, CT_RATE_WEIGHTS_DIR, HF_TOKEN,
                 REPO_CT_RATE_ID, logger)


def download_file(subfolder, file, local_dir):
    local_path = Path(hf_hub_download(
        repo_id=REPO_CT_RATE_ID,
        repo_type='dataset',
        token=HF_TOKEN,
        subfolder=subfolder,
        filename=file,
        cache_dir=CACHE_DIR,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    ))
    logger.info(f"Downloaded {file} to {local_path}")
    return local_path

def download_files(subfolder, files, local_dir):
    downloaded_files = []
    for file in files:
        downloaded_files.append(download_file(subfolder, file, local_dir))
    return downloaded_files

def download_metadata():
    subfolder = 'dataset/metadata'
    files = [
        'Metadata_Attributes.xlsx',
        'no_chest_train.txt',
        'no_chest_valid.txt',
        'train_metadata.csv',
        'validation_metadata.csv'
    ]
    return download_files(subfolder, files, CT_RATE_DIR)
        

def download_anatomy_segmentation_labels():
    subfolder = 'dataset/anatomy_segmentation_labels'
    files = [
        'train_label_summary.xlsx',
        'valid_label_summary.xlsx',
    ]
    return download_files(subfolder, files, CT_RATE_DIR)
        

def download_multi_abnormality_labels():
    subfolder = 'dataset/multi_abnormality_labels'
    files = [
        'train_predicted_labels.csv',
        'valid_predicted_labels.csv'
    ]
    return download_files(subfolder, files, CT_RATE_DIR)
        

def download_radiology_text_reports():
    subfolder = 'dataset/radiology_text_reports'
    files = [
        'train_reports.csv',
        'validation_reports.csv'
    ]
    return download_files(subfolder, files, CT_RATE_DIR)
        

def download_data(directory_name, total = 0, filenames = None):
    if total == 0 and filenames is None:
        return

    split = 'validation' if 'valid' in directory_name else 'train'
    directory_name = f'dataset/{directory_name}/'

    ds = load_dataset(REPO_CT_RATE_ID, "labels", token=HF_TOKEN, cache_dir=CACHE_DIR)

    if filenames is not None:
        data = ds[split].filter(lambda x: x['VolumeName'] in filenames)
    else:
        data = ds[split][:total]

    downloaded_files = []
    for name in data['VolumeName']:
        folder1 = name.split('_')[0]
        folder2 = name.split('_')[1]
        folder = folder1 + '_' + folder2
        folder3 = name.split('_')[2]
        subfolder = folder + '_' + folder3
        subfolder = directory_name + folder + '/' + subfolder

        downloaded_files.append(download_file(subfolder, name, CT_RATE_DIR))

    return downloaded_files

def download_CLIP(models):
    subfolder = 'models/CT-CLIP-Related'
    files = [
        'CT-CLIP_v2.pt',
        'CT_LiPro_v2.pt',
        'CT_VocabFine_v2.pt',
    ]
    return download_files(subfolder, files, CT_RATE_WEIGHTS_DIR)


