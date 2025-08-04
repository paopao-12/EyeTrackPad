import os
import h5py
import numpy as np
import cv2
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

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
    
    def print_dataset_stats(self):
        total_images = 0
        train_images = 0
        val_images = 0
        
        for split in ['train', 'val']:
            split_path = os.path.join(self.target_dir, split)
            if not os.path.exists(split_path):
                continue
            
            for gaze_dir in os.listdir(split_path):
                gaze_path = os.path.join(split_path, gaze_dir)
                if os.path.isdir(gaze_path):
                    num_images = len(os.listdir(gaze_path))
                    total_images += num_images
                    if split == 'train':
                        train_images += num_images
                    else:
                        val_images += num_images
        
        print(f"\nDataset Statistics:")
        print(f"Total images: {total_images}")
        print(f"Training images: {train_images}")
        print(f"Validation images: {val_images}")

def main():
    source_dir = r"C:\Users\JAG\Downloads\MPIIFaceGaze"
    manager = DatasetManager(source_dir)
    manager.setup_dataset()
    manager.print_dataset_stats()

if __name__ == "__main__":
    main() 