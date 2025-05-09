import warnings
warnings.filterwarnings("ignore", message="It is not recommended to directly access")
import os
import pickle
import json
import datetime

def save_dataset_split(train_dataset, valid_dataset, test_dataset, dataset_name, seed, split_dir='./dataset_splits'):
    os.makedirs(split_dir, exist_ok=True)

    dataset_split_dir = os.path.join(split_dir, dataset_name)
    os.makedirs(dataset_split_dir, exist_ok=True)

    seed_dir = os.path.join(dataset_split_dir, f'seed_{seed}')
    os.makedirs(seed_dir, exist_ok=True)

    try:
        with open(os.path.join(seed_dir, 'train_dataset.pkl'), 'wb') as f:
            pickle.dump(train_dataset, f)
        with open(os.path.join(seed_dir, 'valid_dataset.pkl'), 'wb') as f:
            pickle.dump(valid_dataset, f)
        with open(os.path.join(seed_dir, 'test_dataset.pkl'), 'wb') as f:
            pickle.dump(test_dataset, f)

        metadata = {
            'seed': seed,
            'dataset': dataset_name,
            'train_size': len(train_dataset),
            'valid_size': len(valid_dataset),
            'test_size': len(test_dataset),
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        }
        with open(os.path.join(seed_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)

        print(f"[INFO] The dataset segmentation has been saved to the: {seed_dir}")
        return True
    except Exception as e:
        print(f"[ERROR] Error while saving dataset split: {e}")
        return False

def load_dataset_split(dataset_name, seed, split_dir='./dataset_splits'):
    dataset_split_dir = os.path.join(split_dir, dataset_name)
    seed_dir = os.path.join(dataset_split_dir, f'seed_{seed}')

    if not os.path.exists(seed_dir):
        print(f"[WARNING] The specified split directory could not be found: {seed_dir}")
        return None, None, None

    try:
        metadata_path = os.path.join(seed_dir, 'metadata.json')
        if not os.path.exists(metadata_path):
            print(f"[WARNING] Metadata file not found: {metadata_path}")
            return None, None, None

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        if metadata['seed'] != seed or metadata['dataset'] != dataset_name:
            print(f"[WARNING] Metadata mismatch: seed of expectation={seed}，dataset={dataset_name}")
            return None, None, None

        with open(os.path.join(seed_dir, 'train_dataset.pkl'), 'rb') as f:
            train_dataset = pickle.load(f)
        with open(os.path.join(seed_dir, 'valid_dataset.pkl'), 'rb') as f:
            valid_dataset = pickle.load(f)
        with open(os.path.join(seed_dir, 'test_dataset.pkl'), 'rb') as f:
            test_dataset = pickle.load(f)

        print(f"[INFO] Successfully loaded dataset splits from: {seed_dir}")
        print(
            f"[INFO] training set: {len(train_dataset)}, validation set: {len(valid_dataset)}, test set: {len(test_dataset)}个样本")
        return train_dataset, valid_dataset, test_dataset
    except Exception as e:
        print(f"[ERROR] Error loading dataset split: {e}")
        return None, None, None