"""
Веб-интерфейс для системы анализа медицинских изображений грудной клетки.

Этот модуль предоставляет Gradio-интерфейс для загрузки и анализа
медицинских изображений. Поддерживает загрузку DICOM, NIFTI файлов
и архивов, асинхронную обработку через RabbitMQ и отображение результатов.

Основные компоненты:
- Gradio интерфейс для загрузки файлов
- Асинхронная обработка через RabbitMQ
- Отображение результатов в табличном формате
- Инструкции по использованию системы
"""

import asyncio
import os
from threading import Thread

import gradio as gr
import yaml
from loguru import logger
import pandas as pd

from .src.rabbit_wrapper import RabbitWrapper
from .src.guide import TEXT
from tempfile import NamedTemporaryFile
from pathlib import Path

config_path = os.getenv("CONFIG_PATH", "../configs/config.yaml")
config = yaml.safe_load(Path(config_path).read_text())["frontend"]
publisher = RabbitWrapper(config)
consumer = RabbitWrapper(config)

columns = [
    "path_to_study",
    "study_uid",
    "series_uid",
    "probability_of_pathology",
    "pathology",
    "processing_status",
    "time_of_processing",
    "most_dangerous_pathology_type",
]

# создаём DataFrame с 2 пустыми строками
empty_df = pd.DataFrame([{column: [''] for column in columns} for _ in range(3)])


async def process_file(files_list: list[str], request: gr.Request):
    """Асинхронная обработка загруженных медицинских файлов.

    Принимает список файлов от Gradio, создает временные файлы,
    отправляет их на обработку через RabbitMQ и ожидает результаты.

    Args:
        files_list (list[str]): Список путей к загруженным файлам
        request (gr.Request): Объект запроса Gradio для получения session_hash

    Returns:
        tuple: Кортеж из обновленной таблицы результатов и сообщения
               (None, None) если файлы не загружены или произошла ошибка
    """
    if files_list:
        dfs_desease = []
        session_ids = []
        temp_files = []
        warnings = []
        # Сначала отправляем все запросы
        for i, file in enumerate(files_list):
            temp_file = NamedTemporaryFile(
                dir="../storage", delete=False, suffix="".join(Path(file).suffixes)
            )
            temp_file.write(open(file, "rb").read())
            temp_files.append(temp_file)
            temp_file.flush()
            logger.info(
                f"Writing Gradio file to temp file {str(Path(temp_file.name).resolve())}"
            )
            try:
                await publisher.publish(
                    {
                        "file_path": str(Path(temp_file.name).resolve()),
                        "study_name": str(Path(file).name),
                        "session_id": request.session_hash + f"_{i}",
                    },
                    config["OUTPUT_TOPIC"],
                )
                session_ids.append(request.session_hash + f"_{i}")
            except ValueError:
                return None, None, None, None

        # Затем ждем все результаты
        for session_id in session_ids:
            result = await consumer.get_result(session_id)
            if result:
                dfs_desease.append(pd.DataFrame(result["df_desease"]))
                warnings.append(result["warnings"])

        for temp_file in temp_files:
            if os.path.exists(temp_file.name):
                temp_file.close()
                os.remove(temp_file.name)

        if dfs_desease:
            # Берём message из последнего результата (или можно изменить логику по необходимости)
            report_file = NamedTemporaryFile(delete=False, suffix=".xlsx", prefix="/tmp/gradio/result_")
            pd.concat([*dfs_desease]).to_excel(report_file, index=False)
            return pd.concat([*dfs_desease]), {}, "\n".join(warnings), report_file.name

    return None, None, None, None


def clean():
    """Очистка интерфейса от всех данных.

    Returns:
        tuple: Кортеж из None значений для сброса всех полей интерфейса
    """
    return None, None, None, None, None


def start_background_loop() -> None:
    """Запуск фонового цикла для прослушивания RabbitMQ.

    Создает новый event loop и запускает прослушивание очереди
    для получения результатов обработки.
    """
    rabbit_loop = asyncio.new_event_loop()
    logger.info(f"Listening {config['INPUT_TOPIC']}")
    rabbit_loop.run_until_complete(consumer.consume(config["INPUT_TOPIC"]))
    rabbit_loop.run_forever()


def main():
    """Основная функция запуска веб-интерфейса.

    Запускает фоновый поток для прослушивания RabbitMQ,
    создает Gradio интерфейс и запускает веб-сервер.
    """
    Thread(target=start_background_loop, daemon=True).start()

    with gr.Blocks(
        fill_height=True,
        theme=gr.themes.Soft(
            font=gr.themes.Default().font, font_mono=gr.themes.Default().font_mono
        ),
    ) as demo:
        with gr.Row(scale=9, max_height=1000):
            with gr.Column(scale=2):
                input_file = gr.File(
                    label="Входное исследование / набор исследований",
                    type="filepath",
                    file_count="multiple",
                    min_width=1,
                    scale=1,
                    file_types=[".gz", ".dcm", ".zip"],
                )
                process_button = gr.Button("Обработать")

            with gr.Column(scale=7):
                title = gr.Markdown(
                    "<h1><center>ИССЛЕДОВАНИЕ КОМПЬЮТЕРНОЙ ТОМОГРАФИИ ОРГАНОВ ГРУДНОЙ КЛЕТКИ</center></h1>",
                    show_label=False,
                    container=True,
                )
                output_table_desease = gr.Dataframe(
                    headers=columns,
                    value=empty_df,
                    col_count=(8, "fixed"),
                    row_count=(2, "dynamic"),
                    interactive=False,
                    show_copy_button=True,
                    max_height=600,
                )
                clean_button = gr.Button("Очистить")

        with gr.Row():
            warnings = gr.Textbox(
                label="Предупреждения и ошибки",
                interactive=False,
                lines=10,
                max_lines=10,
            )

        with gr.Row():
            download_File = gr.File(label="Скачать результаты в XLSX", type="binary")

        with gr.Row():
            description = gr.Markdown(TEXT, container=True)

        message = gr.JSON(label="Сообщение", visible=False)

        process_button.click(
            process_file,
            inputs=[input_file],
            outputs=[output_table_desease, message, warnings, download_File],
        )
        clean_button.click(
            clean,
            outputs=[
                input_file,
                output_table_desease,
                message,
                warnings,
                download_File,
            ],
        )

    demo.queue(default_concurrency_limit=10)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        allowed_paths=["/home/borntowarn/projects"],
    )


if __name__ == "__main__":
    """Точка входа в веб-приложение.

    Запускает Gradio интерфейс для анализа медицинских изображений
    на порту 7860 с поддержкой множественных файлов и асинхронной обработки.
    """
    main()
