import os
import time
import traceback
from pathlib import Path

import pandas as pd
import torch
from src import CT_RATE_WEIGHTS_DIR, logger
from src.data import inference_examination_to_tensor
from src.modeling import MLP, ProjectionVIT

HEADERS = [
    "path_to_study",
    "study_uid",
    "series_uid",
    "probability_of_pathology",
    "pathology",
    "processing_status",
    "time_of_processing",
    "most_dangerous_pathology_type",
]

NAME2INDEX = {
    "Medical material": 0,
    "Arterial wall calcification": 1,
    "Cardiomegaly": 2,
    "Pericardial effusion": 3,
    "Coronary artery wall calcification": 4,
    "Hiatal hernia": 5,
    "Lymphadenopathy": 6,
    "Emphysema": 7,
    "Atelectasis": 8,
    "Lung nodule": 9,
    "Lung opacity": 10,
    "Pulmonary fibrotic sequela": 11,
    "Pleural effusion": 12,
    "Mosaic attenuation pattern": 13,
    "Peribronchial thickening": 14,
    "Consolidation": 15,
    "Bronchiectasis": 16,
    "Interlobular septal thickening": 17,
    "COVID-19": 18,
    "Cancer": 19,
}

SEVERITY_ORDER = [
    "Cancer",
    "COVID-19",
    "Pulmonary fibrotic sequela",
    "Pleural effusion",
    "Pericardial effusion",
    "Cardiomegaly",
    "Consolidation",
    "Emphysema",
    "Bronchiectasis",
    "Atelectasis",
    "Lung nodule",
    "Lung opacity",
    "Mosaic attenuation pattern",
    "Interlobular septal thickening",
    "Peribronchial thickening",
    "Lymphadenopathy",
    "Arterial wall calcification",
    "Coronary artery wall calcification",
    "Hiatal hernia",
    "Medical material",
]

DESEASES2THRESHOLDS = {
    "Medical material": 0.5,
    "Arterial wall calcification": 0.5,
    "Cardiomegaly": 0.5,
    "Pericardial effusion": 0.5,
    "Coronary artery wall calcification": 0.5,
    "Hiatal hernia": 0.5,
    "Lymphadenopathy": 0.5,
    "Emphysema": 0.5,
    "Atelectasis": 0.5,
    "Lung nodule": 0.5,
    "Lung opacity": 0.5,
    "Pulmonary fibrotic sequela": 0.5,
    "Pleural effusion": 0.5,
    "Mosaic attenuation pattern": 0.5,
    "Peribronchial thickening": 0.5,
    "Consolidation": 0.5,
    "Bronchiectasis": 0.5,
    "Interlobular septal thickening": 0.5,
    "COVID-19": 0.5,
    "Cancer": 0.5,
}

BINARY_THRESHOLD = 0.5


def postprocess(result_binary, result_multilabel):
    """Постобработка результатов инференса для получения финального диагноза.

    Анализирует результаты моделей и определяет:
    1. Наличие патологии на основе бинарной модели
    2. Наиболее опасный тип патологии по порядку приоритета
    3. Вероятность наличия патологии

    Args:
        result (dict): Результаты инференса, содержащий:
            - multilabel: Вероятности для каждого типа патологии
            - binary: Вероятность наличия патологии

    Returns:
        tuple: Кортеж из трех элементов:
            - bool: Наличие патологии (True/False)
            - float: Вероятность наличия патологии
            - str: Название наиболее опасной патологии (пустая строка если патологии нет)
    """
    binary_desease_probability = result_binary[0][-1]
    is_desease = binary_desease_probability > BINARY_THRESHOLD

    most_dangerous_pathology_type = ""
    if is_desease:
        most_dangerous_pathology_type = "Невозможно определить патологию"
        for desease_name in SEVERITY_ORDER:
            desease_index = NAME2INDEX[desease_name]
            multilabel_desease_probability = result_multilabel[0][desease_index]
            if multilabel_desease_probability > DESEASES2THRESHOLDS[desease_name]:
                most_dangerous_pathology_type = desease_name
                break

    return is_desease, binary_desease_probability, most_dangerous_pathology_type


@torch.no_grad()
def run_inference(input_path: str, output_csv: str) -> None:
    input_path = Path(input_path)
    if input_path.is_file():
        items = [input_path]
    else:
        items = list(input_path.iterdir())

    if not items:
        logger.error("Не найдено ни одного исследования для обработки")
        return

    vit_model = ProjectionVIT()
    vit_model.load_state_dict(
        torch.load(CT_RATE_WEIGHTS_DIR / "ProjectionVIT_LiPro_V2.pt")
    )
    vit_model.eval().cuda()

    binary_model = MLP(
        input_size=512,
        activation="gelu",
        dropout=0.2,
        hidden_sizes=[256, 128],
        num_classes=2,
    )
    binary_model.load_state_dict(torch.load(CT_RATE_WEIGHTS_DIR / "model_binary.pth"))
    binary_model.eval().cuda()

    multilabel_model = MLP(
        input_size=512,
        activation="leaky_relu",
        dropout=0.2,
        num_classes=20,
        hidden_sizes=[512, 256, 128],
    )
    multilabel_model.load_state_dict(
        torch.load(CT_RATE_WEIGHTS_DIR / "model_multilabel.pth")
    )
    multilabel_model.eval().cuda()

    all_errors = []
    rows = []
    for idx, item in enumerate(items):
        try:
            start_time_preprocess = time.time()
            result = inference_examination_to_tensor(item)

            tensor = result["tensor"]
            study_id = result["study_id"]
            series_id = result["series_id"]
            warnings = result["warnings"]

            logger.info(f"Processing {item.name}")

            vit_embed = vit_model(tensor.unsqueeze(0).cuda())

            logger.info(f"Vit embed: {vit_embed.shape}")

            vit_embed = torch.nn.functional.normalize(vit_embed, dim=-1)

            binary_logits = binary_model(vit_embed)
            multilabel_logits = multilabel_model(vit_embed)

            logger.info(f"Binary logits shape: {binary_logits.shape}")
            logger.info(f"Multilabel logits shape: {multilabel_logits.shape}")

            binary_probs = torch.softmax(binary_logits, dim=-1).tolist()
            multilabel_probs = torch.sigmoid(multilabel_logits).tolist()

            logger.info(f"Binary probs: {binary_probs}")
            logger.info(f"Multilabel probs: {multilabel_probs}")

            is_desease, desease_probability, most_dangerous_pathology_type = (
                postprocess(binary_probs, multilabel_probs)
            )
            end_time_postprocess = time.time()
            rows.append(
                {
                    "path_to_study": item.name,
                    "study_uid": study_id,
                    "series_uid": series_id,
                    "probability_of_pathology": round(desease_probability, 4),
                    "pathology": int(is_desease),
                    "processing_status": "Success",
                    "time_of_processing": round(
                        end_time_postprocess - start_time_preprocess, 4
                    ),
                    "most_dangerous_pathology_type": most_dangerous_pathology_type,
                }
            )
            if len(warnings) > 0:
                all_errors.append(item.name + "\n\t" + "\n\t".join(warnings))
        except Exception as e:
            rows.append(
                {
                    "path_to_study": item.name,
                    "study_uid": "",
                    "series_uid": "",
                    "probability_of_pathology": "",
                    "pathology": "",
                    "processing_status": "Failure",
                    "time_of_processing": "",
                    "most_dangerous_pathology_type": "",
                }
            )
            all_errors.append(item.name + "\n\t" + str(e))
            logger.error(traceback.format_exc())
            logger.error(f"Error processing file {item.name}: {e}")

    if rows:
        logger.info(f"Сохраняем результаты в {output_csv}")
        # Сохранение результатов
        df = pd.DataFrame(rows, columns=HEADERS)
        out_path = Path(output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(out_path, index=False)

        logger.info(f"Сохраняем ошибки в output/errors.txt")
        with open("output/warnings_and_errors.txt", "w") as f:
            f.write("\n".join(all_errors))
    else:
        logger.error(
            "Не найдено ни одного исследования для обработки. Результаты не сохранены."
        )


def main():
    input_path = os.getenv("INPUT_FOLDER", "/input")
    output_csv = os.getenv("OUTPUT_CSV", "./output/output.xlsx")

    run_inference(input_path, output_csv)


if __name__ == "__main__":
    main()
