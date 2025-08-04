import cv2
import mediapipe as mp
import numpy as np
import os
import time
from datetime import datetime

class DataCollector:
    def __init__(self, data_dir="training_data"):
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
        
        # Create data directory if it doesn't exist
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        # Create subdirectories for each gaze direction
        self.directions = ["left", "right", "up", "down", "center"]
        for direction in self.directions:
            dir_path = os.path.join(data_dir, direction)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
        
        # Eye landmarks indices (MediaPipe Face Mesh)
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        
        # Initialize collection variables
        self.current_direction = None
        self.collection_active = False
        self.frame_count = 0
        self.max_frames_per_direction = 100

    def extract_eye_region(self, frame, landmarks, eye_points):
        """Extract the eye region from the frame."""
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
        
        # Resize to standard size
        eye_region = cv2.resize(eye_region, (64, 64))
        
        return eye_region

    def run(self):
        """Main loop for data collection."""
        print("Data Collection Mode")
        print("Press 'l' for LEFT, 'r' for RIGHT, 'u' for UP, 'd' for DOWN, 'c' for CENTER")
        print("Press 's' to start/stop collection for the selected direction")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Extract eye regions
                    left_eye = self.extract_eye_region(frame, face_landmarks, self.LEFT_EYE)
                    right_eye = self.extract_eye_region(frame, face_landmarks, self.RIGHT_EYE)
                    
                    # Combine eyes horizontally
                    combined_eyes = np.hstack((left_eye, right_eye))
                    
                    # Display current status
                    status = f"Direction: {self.current_direction if self.current_direction else 'None'}"
                    status += f" | Collection: {'Active' if self.collection_active else 'Inactive'}"
                    status += f" | Frames: {self.frame_count}/{self.max_frames_per_direction}"
                    
                    cv2.putText(frame, status, (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Save frame if collection is active
                    if self.collection_active and self.current_direction:
                        if self.frame_count < self.max_frames_per_direction:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                            filename = f"{timestamp}.jpg"
                            save_path = os.path.join(self.data_dir, self.current_direction, filename)
                            cv2.imwrite(save_path, combined_eyes)
                            self.frame_count += 1
                            
                            # Display progress
                            progress = self.frame_count / self.max_frames_per_direction * 100
                            cv2.putText(frame, f"Progress: {progress:.1f}%", (10, 60),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow('Data Collection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('l'):
                self.current_direction = "left"
                self.collection_active = False
                self.frame_count = 0
            elif key == ord('r'):
                self.current_direction = "right"
                self.collection_active = False
                self.frame_count = 0
            elif key == ord('u'):
                self.current_direction = "up"
                self.collection_active = False
                self.frame_count = 0
            elif key == ord('d'):
                self.current_direction = "down"
                self.collection_active = False
                self.frame_count = 0
            elif key == ord('c'):
                self.current_direction = "center"
                self.collection_active = False
                self.frame_count = 0
            elif key == ord('s'):
                if self.current_direction:
                    self.collection_active = not self.collection_active
                    if self.collection_active:
                        print(f"Started collecting data for {self.current_direction}")
                    else:
                        print(f"Stopped collecting data for {self.current_direction}")
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    collector = DataCollector()
    collector.run() 