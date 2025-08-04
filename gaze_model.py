import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import cv2
import os
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt

class GazeDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.gaze_categories = ['LEFT', 'CENTER', 'RIGHT', 'UP', 'DOWN']
        
        for category in self.gaze_categories:
            category_path = os.path.join(data_dir, category)
            if os.path.exists(category_path):
                for img_name in os.listdir(category_path):
                    if img_name.endswith(('.jpg', '.png')):
                        self.samples.append({
                            'image_path': os.path.join(category_path, img_name),
                            'gaze': category
                        })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = cv2.imread(sample['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
        
        gaze = sample['gaze']
        gaze_idx = self.gaze_categories.index(gaze)
        
        return image, gaze_idx

class DatasetManager:
    def __init__(self, source_dir, target_dir="processed_dataset"):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.image_size = (224, 224)
    
    def create_directory_structure(self):
        splits = ['train', 'val']
        gaze_dirs = ['LEFT', 'CENTER', 'RIGHT', 'UP', 'DOWN']
        
        for split in splits:
            for gaze_dir in gaze_dirs:
                path = os.path.join(self.target_dir, split, gaze_dir)
                os.makedirs(path, exist_ok=True)
    
    def preprocess_gaze(self, gaze_direction):
        pitch = np.arcsin(-gaze_direction[1])
        yaw = np.arctan2(-gaze_direction[0], -gaze_direction[2])
        
        if yaw < -np.pi/3:
            h_gaze = "LEFT"
        elif yaw > np.pi/3:
            h_gaze = "RIGHT"
        else:
            h_gaze = "CENTER"
            
        if pitch < -np.pi/6:
            v_gaze = "UP"
        elif pitch > np.pi/6:
            v_gaze = "DOWN"
        else:
            v_gaze = "CENTER"
            
        return h_gaze, v_gaze
    
    def process_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size)
        return image
    
    def setup_dataset(self):
        self.create_directory_structure()
        
        for subject_id in tqdm(range(15), desc="Processing subjects"):
            subject_dir = os.path.join(self.source_dir, f'p{subject_id:02d}')
            if not os.path.exists(subject_dir):
                continue
            
            h5_path = os.path.join(subject_dir, f'p{subject_id:02d}.h5')
            if not os.path.exists(h5_path):
                continue
            
            with h5py.File(h5_path, 'r') as f:
                image_paths = f['path'][:]
                gaze_directions = f['gaze'][:]
                
                image_paths = [path.decode('utf-8') for path in image_paths]
                
                for img_path, gaze in zip(image_paths, gaze_directions):
                    full_path = os.path.join(subject_dir, img_path)
                    if not os.path.exists(full_path):
                        continue
                    
                    image = self.process_image(full_path)
                    if image is None:
                        continue
                    
                    h_gaze, v_gaze = self.preprocess_gaze(gaze)
                    
                    split = 'train' if np.random.random() < 0.8 else 'val'
                    
                    for gaze_dir in [h_gaze, v_gaze]:
                        if gaze_dir != "CENTER":
                            target_path = os.path.join(
                                self.target_dir,
                                split,
                                gaze_dir,
                                f'p{subject_id:02d}_{os.path.basename(img_path)}'
                            )
                            cv2.imwrite(target_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    def get_data_loaders(self, batch_size=32):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        train_dataset = GazeDataset(
            os.path.join(self.target_dir, 'train'),
            transform=transform
        )
        
        val_dataset = GazeDataset(
            os.path.join(self.target_dir, 'val'),
            transform=transform
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
        
        return train_loader, val_loader

class GazeCNN(nn.Module):
    def __init__(self):
        super(GazeCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(self._get_conv_output_size(), 256)
        self.fc2 = nn.Linear(256, 5)
    
    def _get_conv_output_size(self):
        x = torch.randn(1, 3, 224, 224)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        return x.view(1, -1).size(1)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class GazeLSTM(nn.Module):
    def __init__(self, input_size=256, hidden_size=128, num_layers=2):
        super(GazeLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 5)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

class GazeNet(nn.Module):
    def __init__(self):
        super(GazeNet, self).__init__()
        self.cnn = GazeCNN()
        self.lstm = GazeLSTM()
    
    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w)
        x = self.cnn(x)
        x = x.view(batch_size, seq_len, -1)
        x = self.lstm(x)
        return x

class GazeTrainer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=3)
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        return total_loss / len(train_loader), 100. * correct / total
    
    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        val_loss /= len(val_loader)
        accuracy = 100. * correct / total
        self.scheduler.step(val_loss)
        return val_loss, accuracy
    
    def train(self, train_loader, val_loader, epochs=50):
        best_val_acc = 0
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'best_gaze_model.pth')
        
        self.plot_history(history)
        return history
    
    def plot_history(self, history):
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.title('Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train Acc')
        plt.plot(history['val_acc'], label='Val Acc')
        plt.title('Accuracy History')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    source_dir = r"C:\Users\JAG\Downloads\MPIIFaceGaze"
    manager = DatasetManager(source_dir)
    manager.setup_dataset()
    
    train_loader, val_loader = manager.get_data_loaders()
    
    model = GazeNet().to(device)
    trainer = GazeTrainer(model, device)
    
    history = trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    main() 