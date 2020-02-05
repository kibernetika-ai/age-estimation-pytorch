import argparse
import os
import sys

import better_exceptions
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import cv2
from torch.utils.data import Dataset
from imgaug import augmenters as iaa


class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.OneOf([
                iaa.Sometimes(0.25, iaa.AdditiveGaussianNoise(scale=0.1 * 255)),
                iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 2.0)))
                ]),
            iaa.Affine(
                rotate=(-10, 10), mode="edge",
                scale={"x": (0.95, 1.05), "y": (0.95, 1.05)},
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}
            ),
            iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True),
            iaa.GammaContrast((0.3, 2)),
            iaa.Fliplr(0.5),
        ])

    def __call__(self, img):
        img = np.array(img)
        img = self.aug.augment_image(img)
        return img


class FaceDatasets(Dataset):
    def __init__(self, appa_real_dir, utk_dir, data_type, img_size=224, augment=False, age_stddev=1.0):
        assert(data_type in ("train", "valid", "test"))
        csv_path = Path(appa_real_dir).joinpath(f"gt_avg_{data_type}.csv")
        img_dir = Path(appa_real_dir).joinpath(data_type)
        self.img_size = img_size
        self.augment = augment
        self.age_stddev = age_stddev

        if utk_dir is not None:
            self.utk = UTKFaceDataset(utk_dir)
        else:
            self.utk = None

        if augment:
            self.transform = ImgAugTransform()
        else:
            self.transform = lambda i: i

        self.x = []
        self.y = []
        self.std = []
        df = pd.read_csv(str(csv_path))
        ignore_path = Path(__file__).resolve().parent.joinpath("ignore_list.csv")
        ignore_img_names = list(pd.read_csv(str(ignore_path))["img_name"].values)

        for _, row in df.iterrows():
            img_name = row["file_name"]

            if img_name in ignore_img_names:
                continue

            img_path = img_dir.joinpath(img_name + "_face.jpg")
            assert(img_path.is_file())
            self.x.append(str(img_path))
            self.y.append(row["apparent_age_avg"])
            self.std.append(row["apparent_age_std"])

    def __len__(self):
        if self.utk:
            return len(self.y) + len(self.utk.files)
        else:
            return len(self.y)

    def __getitem__(self, idx):
        real_idx = idx // 2 + 1 if self.utk else idx
        if idx % 2 == 0 or self.utk is None:
            from_appa = True
            if real_idx >= len(self.y):
                from_appa = False
        else:
            from_appa = False
            if real_idx >= len(self.utk.files):
                from_appa = True

        if from_appa:
            img_path = self.x[real_idx]
            age = self.y[real_idx]
        else:
            img_path = self.utk.files[idx // 2 + 1]
            age = self.utk.get_label(img_path)

        if self.augment:
            if idx // 2 + 1 < len(self.std):
                age += np.random.randn() * self.std[idx // 2 + 1] * self.age_stddev

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        sys.stdout.flush()
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = self.transform(img).astype(np.float32)
        return torch.from_numpy(np.transpose(img, (2, 0, 1))), np.clip(round(age), 0, 100)


class UTKFaceDataset(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        files = []
        for base in os.listdir(self.data_dir):
            files.append(os.path.join(self.data_dir, base))

        self.files = files

    def get_label(self, file_name):
        base_name = os.path.basename(file_name)
        return int(base_name.split('_')[0])


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--appa_real_dir", type=str, required=True)
    parser.add_argument("--utk_dir", type=str, required=True)
    args = parser.parse_args()
    dataset = FaceDatasets(args.appa_real_dir, args.utk_dir, "train")
    print("train dataset len: {}".format(len(dataset)))
    dataset = FaceDatasets(args.appa_real_dir, args.utk_dir, "valid")
    print("valid dataset len: {}".format(len(dataset)))
    dataset = FaceDatasets(args.appa_real_dir, args.utk_dir, "test")
    print("test dataset len: {}".format(len(dataset)))


if __name__ == '__main__':
    main()
