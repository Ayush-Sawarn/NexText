from TextSummariser.config.configuration import ConfigurationManager
from TextSummariser.components.data_ingestion import DataIngestion
from TextSummariser.logging import logger

class DataIngestionTrainingPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        try:
            config= ConfigurationManager()
            data_ingestion_config= config.get_data_ingestion_config()
            data_ingestion= DataIngestion(config= data_ingestion_config)
            data_ingestion.download_file()
            data_ingestion.extract_zip_file()
        except Exception as e:
            raise e