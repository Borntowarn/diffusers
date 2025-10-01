from src.utils.downloading import (
    download_anatomy_segmentation_labels,
    download_data,
    download_metadata,
    download_multi_abnormality_labels,
    download_radiology_text_reports,
)

if __name__ == "__main__":
    # download_metadata()
    # download_anatomy_segmentation_labels()
    # download_multi_abnormality_labels()
    # download_radiology_text_reports()

    download_data("valid", total=2)
    # download_data('valid')

    # download_data("train_fixed", total=5)
    # download_data('train', total=1)
