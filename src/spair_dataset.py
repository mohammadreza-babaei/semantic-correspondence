# From: https://huggingface.co/datasets/0jl/SPair-71k/tree/main

"""SPair-71k: A Large-scale Benchmark for Semantic Correspondence"""

import json
import os
import numpy as np
import datasets
from datasets import BuilderConfig, Features, Value, SplitGenerator, Split, ClassLabel, Array2D, Sequence, Image
from pathlib import Path


_CITATION = """\
@article{min2019spair,
   title={SPair-71k: A Large-scale Benchmark for Semantic Correspondence},
   author={Juhong Min and Jongmin Lee and Jean Ponce and Minsu Cho},
   journal={arXiv prepreint arXiv:1908.10543},
   year={2019}
}
"""

_DESCRIPTION = """\
Establishing visual correspondences under large intra-class variations, which is often referred to as semantic correspondence or semantic matching, remains a challenging problem in computer vision. Despite its significance, however, most of the datasets for semantic correspondence are limited to a small amount of image pairs with similar viewpoints and scales. In this paper, we present a new large-scale benchmark dataset of semantically paired images, SPair-71k, which contains 70,958 image pairs with diverse variations in viewpoint and scale. Compared to previous datasets, it is significantly larger in number and contains more accurate and richer annotations. We believe this dataset will provide a reliable testbed to study the problem of semantic correspondence and will help to advance research in this area. We provide the results of recent methods on our new dataset as baselines for further research.

This huggingface version of the dataset is inofficial. It downloads the data from the original source and converts it to the huggingface format. 

## Terms of Use

The SPair-71k data includes images and metadata obtained from the [PASCAL-VOC](http://host.robots.ox.ac.uk/pascal/VOC/) and [flickr](https://www.flickr.com/) website. Use of these images and metadata must respect the corresponding [terms of use](https://www.flickr.com/help/terms).
"""

_HOMEPAGE = "https://cvlab.postech.ac.kr/research/SPair-71k/"

_LICENSE = """
# Terms of use

The SPair-71k data includes images and metadata obtained from the PASCAL-VOC and flickr website. Use of these images and metadata must respect the corresponding terms of use.
"""

_URL = "https://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz"


class SPair71k(datasets.GeneratorBasedBuilder):
    """SPair-71k: A Large-scale Benchmark for Semantic Correspondence"""

    VERSION = datasets.Version("0.5.0")
    _url_override = None

    BUILDER_CONFIGS = [
        BuilderConfig(name="pairs", version=VERSION, description="SPair-71k: dataset of image pairs"),
        BuilderConfig(name="data", version=VERSION, description="SPair-71k: dataset of image data"),
    ]

    DEFAULT_CONFIG_NAME = "pairs"

    @classmethod
    def set_url_override(cls, url):
        """
        Set a custom URL or local file path for the dataset.
        
        Args:
            url: A URL string or local file path (with file:// prefix or absolute path)
        """
        cls._url_override = url

    @classmethod
    def get_data_url(cls):
        """
        Get the data URL, checking for overrides in order of priority:
        1. Class-level override (set via set_url_override)
        2. Environment variable SPAIR_URL
        3. Default URL
        """
        if cls._url_override:
            return cls._url_override
        
        env_url = os.environ.get('SPAIR_URL')
        if env_url:
            return env_url
        
        return _URL

    def _info(self):
        if self.config.name == "pairs":
            features = Features({
                "pair_id": Value("uint32"),
                "src_img": Image(),
                "src_segmentation": Image(),
                "src_data_index": Value("uint32"),
                "src_name": Value("string"),
                "src_imsize": Sequence(Value("uint32"), length=3),
                "src_bndbox": Sequence(Value("uint32"), length=4),
                "src_pose": ClassLabel(names=['Unspecified', 'Frontal', 'Left', 'Rear', 'Right']),
                "src_kps": Array2D(dtype="uint32", shape=(None,2)),
                "trg_img": Image(),
                "trg_segmentation": Image(),
                "trg_data_index": Value("uint32"),
                "trg_name": Value("string"),
                "trg_imsize": Sequence(Value("uint32"), length=3),
                "trg_bndbox": Sequence(Value("uint32"), length=4),
                "trg_pose": ClassLabel(names=['Unspecified', 'Frontal', 'Left', 'Rear', 'Right']),
                "trg_kps": Array2D(dtype="uint32", shape=(None,2)),
                "kps_ids": Sequence(Value("uint32")),
                "category": Value("string"),
                "category_id": ClassLabel(names=['cat', 'pottedplant', 'train', 'bicycle', 'car', 'bus', 'aeroplane', 'dog', 'bird', 'chair', 'motorbike', 'cow', 'bottle', 'person', 'boat', 'sheep', 'horse', 'tvmonitor']),
                "viewpoint_variation": Value("uint8"),
                "scale_variation": Value("uint8"),
                "truncation": Value("uint8"),
                "occlusion": Value("uint8"),
            })
        elif self.config.name == "data":
            features = Features({
                "img": Image(),
                "name": Value('string'),
                "segmentation": Image(),
                "filename": Value("string"),
                "src_database": Value("string"),
                "src_annotation": Value("string"),
                "src_image": Value("string"),
                "image_width": Value("uint32"),
                "image_height": Value("uint32"),
                "image_depth": Value("uint32"),
                "category": Value("string"),
                "category_id": ClassLabel(names=['cat', 'pottedplant', 'train', 'bicycle', 'car', 'bus', 'aeroplane', 'dog', 'bird', 'chair', 'motorbike', 'cow', 'bottle', 'person', 'boat', 'sheep', 'horse', 'tvmonitor']),
                "pose": Value("string"),
                "truncated": Value("uint8"),
                "occluded": Value("uint8"),
                "difficult": Value("uint8"),
                "bndbox": Sequence(Value("uint32"), length=4),
                "kps": Array2D(dtype="int32", shape=(None, 2)),
                "azimuth_id": Value("uint8"),
            })
        else:
            raise ValueError(f"Unknown configuration name {self.config.name}")
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # make sure to download the data even when streaming
        if not isinstance(dl_manager, datasets.DownloadManager):
            dl_manager = datasets.DownloadManager(dl_manager._dataset_name)
        
        # Get the URL (checking for overrides)
        data_url = self.get_data_url()
        
        # Check if it's a local file path
        if data_url.startswith('file://'):
            # Remove file:// prefix
            local_path = data_url[7:]
        elif os.path.isabs(data_url) or os.path.exists(data_url):
            # It's a local absolute or relative path
            local_path = data_url
        else:
            # It's a remote URL, download and extract
            local_path = dl_manager.download_and_extract(data_url)
        
        data_path = local_path

        if self.config.name == "pairs":
            return [
                SplitGenerator(
                    name=Split.TRAIN,
                    gen_kwargs={
                        "path": data_path,
                        "split": "trn",
                    },
                ),
                SplitGenerator(
                    name=Split.VALIDATION,
                    gen_kwargs={
                        "path": data_path,
                        "split": "val",
                    },
                ),
                SplitGenerator(
                    name=Split.TEST,
                    gen_kwargs={
                        "path": data_path,
                        "split": "test"
                    },
                ),
            ]
        elif self.config.name == "data":
            # the data is not split
            return [
                SplitGenerator(
                    name=Split.TRAIN,
                    gen_kwargs={
                        "path": data_path,
                        "split": "data",
                    },
                ),
            ]
        else:
            raise ValueError(f"Unknown configuration name {self.config.name}")

    def _generate_examples(self, path, split):
        path = Path(path) / 'SPair-71k'
        image_indices = {f'{cat}/{name}': i for i, (name, cat) in enumerate(sorted((img_name.name[:-4], folder.name) for folder in (path / 'JPEGImages').glob('*') for img_name in folder.glob('*.jpg')))}
        if self.config.name == 'pairs':
            for pair_file in (path / 'PairAnnotation' / split).glob('*.json'):
                with open(pair_file, 'r') as f:
                    data = json.load(f)
                category = data['category']
                src_name = f'{category}/{data["src_imname"][:-4]}'
                trg_name = f'{category}/{data["trg_imname"][:-4]}'
                yield data["pair_id"], {
                    "pair_id": data["pair_id"],
                    "src_img": {"path": str(path / 'JPEGImages' / f'{src_name}.jpg'), "bytes": None},
                    "src_segmentation": {"path": str(path / 'Segmentation' / f'{src_name}.png'), "bytes": None},
                    "src_data_index": image_indices[src_name],
                    "src_name": src_name,
                    "src_imsize": data["src_imsize"],
                    "src_bndbox": data["src_bndbox"],
                    "src_pose": data["src_pose"],
                    "src_kps": data["src_kps"],
                    "trg_img": {"path": str(path / 'JPEGImages' / f'{trg_name}.jpg'), "bytes": None},
                    "trg_segmentation": {"path": str(path / 'Segmentation' / f'{trg_name}.png'), "bytes": None},
                    "trg_data_index": image_indices[trg_name],
                    "trg_name": trg_name,
                    "trg_imsize": data["trg_imsize"],
                    "trg_bndbox": data["trg_bndbox"],
                    "trg_pose": data["trg_pose"],
                    "trg_kps": data["trg_kps"],
                    "kps_ids": data["kps_ids"],
                    "category": category,
                    "category_id": category,
                    "viewpoint_variation": data["viewpoint_variation"],
                    "scale_variation": data["scale_variation"],
                    "truncation": data["truncation"],
                    "occlusion": data["occlusion"],
                }
        elif self.config.name == 'data':
            for img_name, i in image_indices.items():
                annotation = json.loads((path / 'ImageAnnotation' / f'{img_name}.json').read_text())
                yield i, {
                    "img": {"path": str(path / 'JPEGImages' / f'{img_name}.jpg'), "bytes": None},
                    "name": img_name,
                    "segmentation": {"path": str(path / 'Segmentation' / f'{img_name}.png'), "bytes": None},
                    "filename": annotation['filename'],
                    "src_database": annotation['src_database'],
                    "src_annotation": annotation['src_annotation'],
                    "src_image": annotation['src_image'],
                    "image_width": annotation['image_width'],
                    "image_height": annotation['image_height'],
                    "image_depth": annotation['image_depth'],
                    "category": annotation['category'],
                    "category_id": annotation['category'],
                    "pose": annotation['pose'],
                    "truncated": annotation['truncated'],
                    "occluded": annotation['occluded'],
                    "difficult": annotation['difficult'],
                    "bndbox": annotation['bndbox'],
                    "kps": [[*kp] if kp else [-1, -1] for kp in annotation['kps'].values()],
                    "azimuth_id": annotation['azimuth_id'],
                }
