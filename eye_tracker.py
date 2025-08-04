import cv2
import mediapipe as mp
import numpy as np
import json
import time
import os
from datetime import datetime

class EyeTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        
        self.calibration_data = {
            'points': {},
            'timestamp': datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        }
        
        self.current_point = 0
        self.calibration_points = [
            (0.2, 0.2), (0.5, 0.2), (0.8, 0.2),
            (0.2, 0.5), (0.5, 0.5), (0.8, 0.5),
            (0.2, 0.8), (0.5, 0.8), (0.8, 0.8)
        ]
        
        self.collecting_data = False
        self.collection_start_time = None
        self.collection_duration = 3.0
        self.samples_per_point = 30
        
        self.is_calibration_mode = True
        self.training_data = []
        self.training_data_dir = "training_data"
        os.makedirs(self.training_data_dir, exist_ok=True)
        
        self.calibration_complete = False
        
        self.gaze_history = []
        self.max_history = 5
        self.smoothing_factor = 0.3
        
        self.screen_width = 1920
        self.screen_height = 1080
    
    def get_iris_center(self, landmarks, eye_indices):
        eye_points = np.array([(landmarks.landmark[idx].x, landmarks.landmark[idx].y) for idx in eye_indices])
        return np.mean(eye_points, axis=0)
    
    def get_horizontal_gaze(self, landmarks, eye_indices, iris_indices):
        eye_center = np.mean([(landmarks.landmark[idx].x, landmarks.landmark[idx].y) for idx in eye_indices], axis=0)
        iris_center = np.mean([(landmarks.landmark[idx].x, landmarks.landmark[idx].y) for idx in iris_indices], axis=0)
        
        eye_width = landmarks.landmark[eye_indices[0]].x - landmarks.landmark[eye_indices[8]].x
        iris_offset = iris_center[0] - eye_center[0]
        
        return iris_offset / (eye_width * 0.5)
    
    def get_vertical_gaze(self, landmarks, eye_indices, iris_indices):
        eye_center = np.mean([(landmarks.landmark[idx].x, landmarks.landmark[idx].y) for idx in eye_indices], axis=0)
        iris_center = np.mean([(landmarks.landmark[idx].x, landmarks.landmark[idx].y) for idx in iris_indices], axis=0)
        
        eye_height = landmarks.landmark[eye_indices[12]].y - landmarks.landmark[eye_indices[4]].y
        iris_offset = iris_center[1] - eye_center[1]
        
        return iris_offset / (eye_height * 0.5)
    
    def get_eye_aspect_ratio(self, landmarks, eye_indices):
        points = np.array([(landmarks.landmark[idx].x, landmarks.landmark[idx].y) for idx in eye_indices])
        
        A = np.linalg.norm(points[1] - points[5])
        B = np.linalg.norm(points[2] - points[4])
        C = np.linalg.norm(points[0] - points[3])
        
        ear = (A + B) / (2.0 * C)
        return ear
    
    def detect_gaze_direction(self, landmarks):
        left_iris_center = self.get_iris_center(landmarks, self.LEFT_IRIS)
        right_iris_center = self.get_iris_center(landmarks, self.RIGHT_IRIS)
        
        left_h_gaze = self.get_horizontal_gaze(landmarks, self.LEFT_EYE, self.LEFT_IRIS)
        right_h_gaze = self.get_horizontal_gaze(landmarks, self.RIGHT_EYE, self.RIGHT_IRIS)
        
        left_v_gaze = self.get_vertical_gaze(landmarks, self.LEFT_EYE, self.LEFT_IRIS)
        right_v_gaze = self.get_vertical_gaze(landmarks, self.RIGHT_EYE, self.RIGHT_IRIS)
        
        h_gaze = (left_h_gaze + right_h_gaze) / 2
        v_gaze = (left_v_gaze + right_v_gaze) / 2
        
        left_ear = self.get_eye_aspect_ratio(landmarks, self.LEFT_EYE)
        right_ear = self.get_eye_aspect_ratio(landmarks, self.RIGHT_EYE)
        ear = (left_ear + right_ear) / 2
        
        return {
            'horizontal': h_gaze,
            'vertical': v_gaze,
            'ear': ear,
            'left_iris': left_iris_center,
            'right_iris': right_iris_center
        }
    
    def calculate_gaze_position(self, gaze_data):
        h_gaze = gaze_data['horizontal']
        v_gaze = gaze_data['vertical']
        
        x = int((h_gaze + 1) * self.screen_width / 2)
        y = int((v_gaze + 1) * self.screen_height / 2)
        
        self.gaze_history.append((x, y))
        if len(self.gaze_history) > self.max_history:
            self.gaze_history.pop(0)
        
        if len(self.gaze_history) > 1:
            x = int(np.mean([p[0] for p in self.gaze_history]))
            y = int(np.mean([p[1] for p in self.gaze_history]))
        
        return x, y
    
    def draw_calibration_ui(self, frame):
        h, w = frame.shape[:2]
        point = self.calibration_points[self.current_point]
        x, y = int(point[0] * w), int(point[1] * h)
        
        cv2.circle(frame, (x, y), 20, (0, 255, 0), 2)
        
        if self.collecting_data:
            elapsed = time.time() - self.collection_start_time
            progress = min(1.0, elapsed / self.collection_duration)
            bar_width = int(w * 0.8)
            bar_height = 20
            bar_x = int(w * 0.1)
            bar_y = h - 40
            
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (0, 0, 0), 2)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + bar_height), (0, 255, 0), -1)
            
            if elapsed >= self.collection_duration:
                self.collecting_data = False
                self.current_point = (self.current_point + 1) % len(self.calibration_points)
                if self.current_point == 0:
                    self.calibration_complete = True
                    self.save_calibration_data()
        
        cv2.putText(frame, f"Point {self.current_point + 1}/{len(self.calibration_points)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.putText(frame, "SPACE: Start/Stop Collection", (10, h - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "LEFT/RIGHT: Navigate Points", (10, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "S: Save Data", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    def draw_tracking_ui(self, frame, gaze_data):
        h, w = frame.shape[:2]
        
        gaze_x, gaze_y = self.calculate_gaze_position(gaze_data)
        
        cv2.putText(frame, f"Gaze: H={gaze_data['horizontal']:.2f}, V={gaze_data['vertical']:.2f}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"EAR: {gaze_data['ear']:.2f}",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Screen Position: ({gaze_x}, {gaze_y})",
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(frame, "SPACE: Start/Stop Data Collection", (10, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "S: Save Training Data", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return gaze_x, gaze_y
    
    def save_calibration_data(self):
        os.makedirs("calibration_data", exist_ok=True)
        filename = os.path.join("calibration_data", f"calibration_data_{self.calibration_data['timestamp']}.json")
        with open(filename, 'w') as f:
            json.dump(self.calibration_data, f)
        print(f"Calibration data saved to {filename}")
    
    def save_training_data(self):
        if not self.training_data:
            print("No training data to save!")
            return
        
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = os.path.join(self.training_data_dir, f"training_data_{timestamp}.json")
        
        with open(filename, 'w') as f:
            json.dump(self.training_data, f)
        print(f"Training data saved to {filename}")
        self.training_data = []
    
    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                
                gaze_data = self.detect_gaze_direction(landmarks)
                
                if self.is_calibration_mode:
                    if self.collecting_data:
                        point_key = f"point_{self.current_point}"
                        if point_key not in self.calibration_data['points']:
                            self.calibration_data['points'][point_key] = []
                        
                        self.calibration_data['points'][point_key].append({
                            'horizontal': float(gaze_data['horizontal']),
                            'vertical': float(gaze_data['vertical']),
                            'ear': float(gaze_data['ear'])
                        })
                    
                    self.draw_calibration_ui(frame)
                    
                    if self.calibration_complete:
                        self.is_calibration_mode = False
                        print("Calibration complete! Switching to tracking mode...")
                else:
                    if self.collecting_data:
                        self.training_data.append({
                            'timestamp': datetime.now().isoformat(),
                            'gaze': {
                                'horizontal': float(gaze_data['horizontal']),
                                'vertical': float(gaze_data['vertical']),
                                'ear': float(gaze_data['ear'])
                            }
                        })
                    
                    gaze_x, gaze_y = self.draw_tracking_ui(frame, gaze_data)
                    
                    # Draw gaze position indicator
                    h, w = frame.shape[:2]
                    screen_x = int(gaze_x * w / self.screen_width)
                    screen_y = int(gaze_y * h / self.screen_height)
                    
                    cv2.circle(frame, (screen_x, screen_y), 10, (0, 0, 255), -1)
                    cv2.circle(frame, (screen_x, screen_y), 15, (0, 0, 255), 2)
            
            cv2.imshow('Eye Tracking', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                if not self.collecting_data:
                    self.collecting_data = True
                    self.collection_start_time = time.time()
            elif key == ord('s'):
                if self.is_calibration_mode:
                    self.save_calibration_data()
                else:
                    self.save_training_data()
            elif self.is_calibration_mode and key in [81, 83]:  # Left/Right arrows
                if key == 81:  # Left arrow
                    self.current_point = (self.current_point - 1) % len(self.calibration_points)
                else:  # Right arrow
                    self.current_point = (self.current_point + 1) % len(self.calibration_points)
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = EyeTracker()
    tracker.run() 