from torch.utils.data import Dataset


class AccentDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data["labels"])

    def __getitem__(self, index):
        item = {
            "features": self.data["features"][index],
            "labels": self.data["labels"][index],
        }

        if "ids" in self.data.keys():
            item["ids"] = self.data["ids"][index]

        return item
