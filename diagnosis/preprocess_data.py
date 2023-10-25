from lstm_diagnosis import DiagnosisRegressionPreprocessingPipeline, DiagnosisTestingRegressionModel

# pipeline = DiagnosisRegressionPreprocessingPipeline(configs='configs.json')
pipeline = DiagnosisTestingRegressionModel(configs='configs.json')

if __name__ == '__main__':
    pipeline()