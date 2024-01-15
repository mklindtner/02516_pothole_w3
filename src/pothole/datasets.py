import json
from pathlib import Path
import xml.etree.ElementTree as ET


from PIL import Image
import torch
import torchvision.transforms.v2 as transforms


DEFAULT_BASE_PATH = Path(__file__).parent.parent.parent / 'data/Potholes'


DEFAULT_IMAGE_TRANSFORM = transforms.Compose(
    [        
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_xml_file(path):
    """Load Pascal Voc XML Annotation file.

    Strongly inspired by https://stackoverflow.com/a/53832130.
    """

    path = Path(path)

    tree = ET.parse(path)
    root = tree.getroot()

    list_with_all_boxes = []

    for boxes in root.iter('object'):

        filename = root.find('filename').text

        ymin, xmin, ymax, xmax = None, None, None, None

        ymin = int(boxes.find("bndbox/ymin").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmax = int(boxes.find("bndbox/xmax").text)

        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)

    return filename, list_with_all_boxes


class PotholeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split,
        image_transform=DEFAULT_IMAGE_TRANSFORM,
        base_path=DEFAULT_BASE_PATH,
        subdir='annotated-images',
    ):
        self.image_transform = image_transform

        with (base_path / 'splits.json').open('rt') as file:
            splits_data = json.loads(file.read())

        testval_split = len(splits_data['test']) // 2

        splits_data['validation'], splits_data['test'] = (
            splits_data['test'][:testval_split],
            splits_data['test'][testval_split:],
        )

        self.image_files = []
        self.boxes = []

        if split == 'all':
            for cur in 'train', 'validation', 'test':
                self.load_samples(base_path / subdir, splits_data[cur])

        else:
            self.load_samples(base_path / subdir, splits_data[split])

    def load_samples(self, path, files):
        for xmlfile in files:
            image_name, boxes = load_xml_file(path / xmlfile)

            self.image_files.append(path / image_name)
            self.boxes.append(boxes)

    def __len__(self):
        """Return the total number of samples."""

        return len(self.image_files)

    def __getitem__(self, idx):
        """Generate one sample of data."""
        image = Image.open(self.image_files[idx])

        X = self.image_transform(image)
        Y = torch.tensor(self.boxes[idx])       
        return X, Y
