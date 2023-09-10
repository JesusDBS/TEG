from lstm_detection import DetectionPreprocessPipeline

preprocess_pipeline = DetectionPreprocessPipeline(configs='configs.json')

if __name__ == "__main__":
    if preprocess_pipeline.configs['DEBUG']:
        preprocess_pipeline()
    
    else:
        print(".......This script only can runs in debug mode......!")