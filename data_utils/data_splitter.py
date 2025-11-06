import pickle
import random
import os
from typing import List, Dict, Any

INPUT_FILE = 'formatted_annotations.pkl'

TRAIN_IDS_FILE = 'split_dataset/train_ids.txt'
VAL_IDS_FILE = 'split_dataset/val_ids.txt'
TEST_IDS_FILE = 'split_dataset/test_ids.txt'

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15

RANDOM_SEED = 42


def load_data(file_path: str) -> List[Dict[str, Any]]:
    """Loads the formatted annotation list from the pickle file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}. Please run data_parser_formatter.py first.")

    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_ids(ids: List[str], file_path: str):
    """Saves a list of image IDs to a text file, one ID per line."""
    with open(file_path, 'w') as f:
        f.write('\n'.join(ids))
    print(f"Saved {len(ids)} IDs to {file_path}")


if __name__ == "__main__":
    try:
        random.seed(RANDOM_SEED)

        all_records = load_data(INPUT_FILE)

        print(f"Loaded {len(all_records)} total thermal images (positive + negative).")

        all_ids = [record['image_id'] for record in all_records]

        random.shuffle(all_ids)

        total_size = len(all_ids)

        train_end_idx = int(total_size * TRAIN_RATIO)
        val_end_idx = train_end_idx + int(total_size * VAL_RATIO)

        train_ids = all_ids[:train_end_idx]
        val_ids = all_ids[train_end_idx:val_end_idx]
        test_ids = all_ids[val_end_idx:]

        assert len(train_ids) + len(val_ids) + len(test_ids) == total_size

        print("-" * 50)
        print(f"Split Ratios: Train ({TRAIN_RATIO*100:.0f}%), Val ({VAL_RATIO*100:.0f}%), Test ({100 - TRAIN_RATIO*100 - VAL_RATIO*100:.0f}%)")
        print(f"Train Set Size: {len(train_ids)}")
        print(f"Validation Set Size: {len(val_ids)}")
        print(f"Test Set Size: {len(test_ids)}")

        save_ids(train_ids, TRAIN_IDS_FILE)
        save_ids(val_ids, VAL_IDS_FILE)
        save_ids(test_ids, TEST_IDS_FILE)

        print("-" * 50)
        print("Data splitting complete. IDs saved to disk.")

    except Exception as e:
        print(f"An error occurred during splitting: {e}")