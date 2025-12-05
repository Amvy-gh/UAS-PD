import argparse
from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.outlier_detector import OutlierDetector
from src.visualizer import Visualizer
from src.evaluator import Evaluator


def main(filepath: str):
    # 1. Load
    loader = DataLoader("../dataset_original/Gelombang_Gabungan.csv")
    df = loader.load()
    report = loader.inspect()

    # 2. Preprocess
    prep = Preprocessor(df)
    prep.handle_missing(drop_threshold=0.01)
    prep.remove_redundant_columns()
    prep.encode_categoricals()
    prep.scale_features()
    df_ready = prep.get_processed(use_selected=False)

    # 3. Detect outliers
    detector = OutlierDetector(df_ready)
    df_labeled = detector.detect()

    # 4. Visualize  
    viz = Visualizer(df_labeled)
    viz.plot_outlier_steps()

    # 5. Evaluate
    evaluator = Evaluator(df_labeled)
    evaluator.evaluate()

    print('\nPipeline completed. Check generated images and outputs.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=False, default='Gelombang_Gabungan.csv', help='Path to CSV file')
    args = parser.parse_args()
    main(args.data)