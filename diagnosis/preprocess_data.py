from lstm_diagnosis import DiagnosisRegressionPreprocessingPipeline

pipeline = DiagnosisRegressionPreprocessingPipeline(configs='configs.json')

if __name__ == '__main__':
    pipeline()