import cv2
import torch
import numpy as np
import mediapipe as mp
from gaze_model import GazeNet
import time
from collections import deque

class GazeTester:
    def __init__(self, model_path='best_gaze_model.pth', sequence_length=5):
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load the trained model
        self.model = GazeNet(sequence_length=sequence_length).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        
        # Initialize sequence buffer
        self.sequence_buffer = deque(maxlen=sequence_length)
        
        # Eye landmarks indices
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        
        # Gaze direction mapping
        self.gaze_mapping = {
            0: "LEFT-UP",
            1: "LEFT-CENTER",
            2: "LEFT-DOWN",
            3: "RIGHT-UP",
            4: "RIGHT-CENTER",
            5: "RIGHT-DOWN"
        }
        
        # Initialize gaze history for smoothing
        self.gaze_history = deque(maxlen=10)
        
    def preprocess_eye(self, frame, landmarks, eye_points):
        """Extract and preprocess eye region"""
        frame_h, frame_w = frame.shape[:2]
        
        # Get eye landmarks
        eye = np.array([(landmarks.landmark[point].x * frame_w, 
                         landmarks.landmark[point].y * frame_h) 
                        for point in eye_points])
        
        # Get bounding box with padding
        x_min, y_min = np.min(eye, axis=0)
        x_max, y_max = np.max(eye, axis=0)
        
        # Add padding
        padding = 10
        x_min = max(0, int(x_min - padding))
        y_min = max(0, int(y_min - padding))
        x_max = min(frame_w, int(x_max + padding))
        y_max = min(frame_h, int(y_max + padding))
        
        # Extract eye region
        eye_region = frame[y_min:y_max, x_min:x_max]
        
        # Resize to model input size
        eye_region = cv2.resize(eye_region, (224, 224))
        
        return eye_region, (x_min, y_min, x_max, y_max)
    
    def get_gaze_direction(self, frame):
        """Process frame and get gaze direction"""
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Extract and preprocess both eyes
                left_eye, left_bbox = self.preprocess_eye(frame, face_landmarks, self.LEFT_EYE)
                right_eye, right_bbox = self.preprocess_eye(frame, face_landmarks, self.RIGHT_EYE)
                
                # Combine eyes horizontally
                combined_eyes = np.hstack((left_eye, right_eye))
                
                # Convert to tensor
                eye_tensor = torch.from_numpy(combined_eyes).float()
                eye_tensor = eye_tensor.permute(2, 0, 1).unsqueeze(0)  # Add batch dimension
                eye_tensor = eye_tensor.to(self.device)
                
                # Add to sequence buffer
                self.sequence_buffer.append(eye_tensor)
                
                # Only predict when we have enough frames
                if len(self.sequence_buffer) == self.sequence_buffer.maxlen:
                    # Stack sequence
                    sequence = torch.stack(list(self.sequence_buffer))
                    
                    # Get prediction
                    with torch.no_grad():
                        outputs, _ = self.model(sequence.unsqueeze(0))
                        predicted = outputs.argmax(dim=1).item()
                    
                    # Get gaze direction
                    gaze_direction = self.gaze_mapping[predicted]
                    
                    # Add to history
                    self.gaze_history.append(gaze_direction)
                    
                    # Get most common gaze direction (smoothing)
                    if len(self.gaze_history) > 0:
                        gaze_direction = max(set(self.gaze_history), key=self.gaze_history.count)
                    
                    # Draw eye regions
                    cv2.rectangle(frame, (left_bbox[0], left_bbox[1]), 
                                (left_bbox[2], left_bbox[3]), (0, 255, 0), 2)
                    cv2.rectangle(frame, (right_bbox[0], right_bbox[1]), 
                                (right_bbox[2], right_bbox[3]), (0, 255, 0), 2)
                    
                    # Display gaze direction
                    cv2.putText(frame, f"Gaze: {gaze_direction}", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    return frame, gaze_direction
        
        return frame, None
    
    def run(self):
        """Main loop for testing"""
        print("Starting gaze detection test...")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for more intuitive interaction
            frame = cv2.flip(frame, 1)
            
            # Process frame
            frame, gaze_direction = self.get_gaze_direction(frame)
            
            # Display the frame
            cv2.imshow('Gaze Detection Test', frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    # Create and run the gaze tester
    tester = GazeTester()
    tester.run()

if __name__ == "__main__":
    main() 