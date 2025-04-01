from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision import transforms


class LoadDataset(Dataset):
    def __init__(self, dataset_name, split, image_size):
        """
        Args:
            dataset_name (str): The dataset to load from Hugging Face.
            split (str): The split of the dataset (e.g., "train", "test").
            image_size (int): The size to which images will be resized.
        """
        # Load the dataset from Hugging Face
        self.dataset = load_dataset(dataset_name, split=split)
        self.split = split
        self.image_size = image_size

        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get the image from the dataset
        item = self.dataset[idx]
        image = item["image"].convert("RGB")
        image = self.transform(image)

        return image
