from lstm_detection import DetectionPreprocessPipeline, DetectionTrainingModelPipeline

pipeline = DetectionPreprocessPipeline(configs='configs.json')
# pipeline = DetectionTrainingModelPipeline(configs='configs.json')

if __name__ == "__main__":
    if pipeline.configs['DEBUG']:
        pipeline()
    
    else:
        print(".......This script only can runs in debug mode......!")