import torch
import torch.utils.data
from torch.utils.data import DataLoader
import os
import pickle
import numpy as np
from typing import List, Dict, Any

from data_utils.dataset_class import WaterfowlDataset, collate_fn
from model_utils.config_model_base import get_model_instance_segmentation, NUM_CLASSES
from model_utils import DEVICE, CHECKPOINT_DIR, MODEL_NAME_PREFIX
from data_utils.data_augmentation import get_transform
from data_utils.dataset_class import load_split_ids

CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, f'{MODEL_NAME_PREFIX}_best.pth')
OUTPUT_ERROR_FILE = 'error_analysis_ids.pkl'
CONFIDENCE_THRESHOLD = 0.75


def analyze_batch(predictions: List[Dict], targets: List[Dict], image_ids: List[str], conf_threshold: float) -> Dict[str, List[str]]:
    """Analyzes a batch of predictions to categorize images by error type."""

    results = {'TP': [], 'FN': [], 'FP': []}

    for pred, target, img_id in zip(predictions, targets, image_ids):
        high_conf_preds = pred['scores'] > conf_threshold
        pred_boxes = pred['boxes'][high_conf_preds]

        gt_boxes = target['boxes']

        num_gt = gt_boxes.shape[0]
        num_pred = pred_boxes.shape[0]

        if num_gt > 0 and num_pred > 0 and abs(num_gt - num_pred) <= 1:
            results['TP'].append(img_id)

        elif num_gt > 2 and num_pred < 2:
            results['FN'].append(img_id)

        elif (num_gt == 0 and num_pred > 0) or (num_pred > num_gt + 2):
            results['FP'].append(img_id)

    return results


def main():
    print(f"Starting error analysis on device: {DEVICE}")

    model = get_model_instance_segmentation(NUM_CLASSES)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()

    test_dataset = WaterfowlDataset(split='test', transforms=get_transform(train=False))
    test_loader = DataLoader(
        test_dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=collate_fn
    )

    test_ids = load_split_ids('split_dataset/test_ids.txt')

    all_error_ids = {'TP': set(), 'FN': set(), 'FP': set()}

    print("Running inference and error categorization...")
    with torch.no_grad():
        for i, (images, targets) in enumerate(test_loader):
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            predictions = model(images)

            batch_indices = [t['image_id'].item() for t in targets]
            batch_img_ids = [test_dataset.records[idx]['image_id'] for idx in batch_indices]

            batch_errors = analyze_batch(predictions, targets, batch_img_ids, CONFIDENCE_THRESHOLD)

            for k in all_error_ids.keys():
                all_error_ids[k].update(batch_errors[k])

    final_error_ids = {
        'TP': list(all_error_ids['TP'])[:3],
        'FN': list(all_error_ids['FN'])[:3],
        'FP': list(all_error_ids['FP'])[:3],
    }

    with open(OUTPUT_ERROR_FILE, 'wb') as f:
        pickle.dump(final_error_ids, f)

    print("\n--- Error Analysis Complete ---")
    for k, v in final_error_ids.items():
        print(f"Selected {len(v)} IDs for {k} Visualization: {v}")


if __name__ == '__main__':
    main()