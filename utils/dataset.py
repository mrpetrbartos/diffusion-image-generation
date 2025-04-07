from datasets import DatasetDict, load_dataset
from torch.utils.data import Dataset
from torchvision import transforms


class LoadDataset(Dataset):
    def __init__(self, dataset_name, split, image_size, caption_detail=0):
        """
        Args:
            dataset_name (str): The dataset to load from Hugging Face.
            split (str): The split of the dataset (e.g., "train", "test").
            image_size (int): The size to which images will be resized.
            caption_detail (int): The level of detail for the captions,
                ranging from 0 to 2, where 2 is most detailed.
        """
        # Load the dataset from Hugging Face
        self.dataset = load_dataset(dataset_name, split="train")
        self.split = split
        self.image_size = image_size
        self.caption_detail = caption_detail
        self.seed = 42

        self._subset_dataset(0.999)
        self._split_dataset(0.05)

        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.detail_map = {
            0: "caption.txt",
            1: "detailed_caption.txt",
            2: "more_detailed_caption.txt",
        }

    def __len__(self):
        return len(self.dataset[self.split])

    def __getitem__(self, idx):
        # Get the image from the dataset
        item = self.dataset[self.split][idx]

        image = item["jpg"].convert("RGB")
        image = self.transform(image)

        caption = item[self.detail_map[self.caption_detail]]

        return image, caption

    def _subset_dataset(self, subset_ratio):
        subset = self.dataset.train_test_split(train_size=subset_ratio, seed=self.seed)["train"]

        self.dataset = subset

    def _split_dataset(self, split_ratio):
        train_test_split = self.dataset.train_test_split(test_size=split_ratio, seed=self.seed)
        dataset = DatasetDict({"train": train_test_split["train"], "test": train_test_split["test"]})

        self.dataset = dataset
