from TextSummariser.config.configuration import ConfigurationManager
from TextSummariser.components.model_trainer import ModelTrainer
from TextSummariser.logging import logger


class ModelTrainerTrainingPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            model_trainer_config = config.get_model_trainer_config()
            model_trainer = ModelTrainer(config=model_trainer_config)
            model_trainer.train()
        except Exception as e:
            raise e 