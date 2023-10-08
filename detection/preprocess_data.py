from lstm_detection import DetectionPreprocessPipeline, DetectionTrainingModelPipeline

pipeline = DetectionPreprocessPipeline(configs='configs.json')
# pipeline = DetectionTrainingModelPipeline(configs='configs.json')

if __name__ == "__main__":
    pipeline()