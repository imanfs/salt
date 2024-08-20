import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


# Step 1: Create a dummy dataset
class DummyDataset(Dataset):
    def __init__(self):  # , num_samples=64):
        super().__init__()
        # self.data = np.random.Generator(num_samples, 10)  # 1000 samples, 10 features
        # self.labels = np.random.randint(0, 2, num_samples)  # Binary labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


# Step 2: Define a simple model using PyTorch Lightning
class SimpleModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(10, 64)
        self.layer_2 = nn.Linear(64, 2)
        self.criterion = nn.CrossEntropyLoss()
        # self.automatic_optimization = False

    def forward(self, x):
        x = torch.relu(self.layer_1(x))
        return self.layer_2(x)

    def training_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        # self.log("train_loss", loss)
        print("train_loss", loss)
        loss.backward()
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


# Step 3: Create DataLoader instances


train_loader = DataLoader(DummyDataset(), batch_size=32)

# Step 4: Train the model
model = SimpleModel()
trainer = pl.Trainer(max_epochs=5)
trainer.fit(model, train_loader)
