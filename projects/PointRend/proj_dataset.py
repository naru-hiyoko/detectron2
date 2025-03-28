import sys
import os
import os.path as osp
import argparse
from glob import glob
from tqdm import tqdm
import cv2

from detectron2.data import DatasetCatalog, MetadataCatalog


def collect_files(img_dir, label_dir):
    suffix = '.jpg'
    files = []
    for img_file in tqdm(glob(osp.join(img_dir, '*.jpg'))):
        try:
            height, width, _ = cv2.imread(img_file).shape
        except:
            continue

        filename = osp.basename(img_file).replace('.jpg', '.png')
        label_file = osp.join(label_dir, filename)
        files.append({
            'file_name': img_file,
            'sem_seg_file_name': label_file,
            'height': height,
            'width': width,
        })
    return files

def generate_catalog(dataset_root):
    img_dir = 'images'
    label_dir = 'labels'

    for split in ['train', 'val']:
        catalog = collect_files(osp.join(dataset_root, img_dir, split),
                                osp.join(dataset_root, label_dir, split))

        DatasetCatalog.register('proj_' + split, lambda : catalog)
        MetadataCatalog.get('proj_' + split).set(
            thing_classes=['bg', 'pole', 'beam', 'rail'],
            stuff_classes=['bg', 'pole', 'beam', 'rail'],
            evaluator_type='sem_seg',
            ignore_label=255)

if __name__ == '__main__':
    # generate_catalog('D:/datasets/proj_all_train_val')
    meta = MetadataCatalog.get('cityscapes_fine_sem_seg_val')
    # meta = MetadataCatalog.get('proj_train')
    print(meta)
