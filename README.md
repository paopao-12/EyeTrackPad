# Eye Trackpad

An eye-tracking system that uses computer vision and deep learning to control mouse movements using eye gaze. This project is designed to help people with physical disabilities (PWDs) by providing an alternative input method using eye movements.

## Features

- Real-time eye tracking using OpenCV and MediaPipe
- Gaze direction detection (left, right, up, down, center)
- Data collection tool for training the CNN model
- Eye aspect ratio calculation for blink detection
- Support for custom training data collection

## Requirements

- Python 3.8 or higher
- Webcam
- Required Python packages (install using `pip install -r requirements.txt`):
  - opencv-python
  - mediapipe
  - numpy
  - tensorflow
  - pyautogui
  - scikit-learn
  - matplotlib

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/eye-trackpad.git
cd eye-trackpad
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Data Collection

To collect training data for the CNN model:

1. Run the data collection script:
```bash
python data_collector.py
```

2. Use the following keys to control data collection:
- 'l': Select LEFT gaze direction
- 'r': Select RIGHT gaze direction
- 'u': Select UP gaze direction
- 'd': Select DOWN gaze direction
- 'c': Select CENTER gaze direction
- 's': Start/stop collecting data for the selected direction
- 'q': Quit the program

The collected data will be saved in the `training_data` directory, organized by gaze direction.

### Eye Tracking

To run the eye tracking system:

1. Run the main script:
```bash
python eye_tracker.py
```

2. The system will:
- Open your webcam
- Detect your face and eyes
- Track your gaze direction
- Display the current gaze direction and eye aspect ratio
- Press 'q' to quit

## Project Structure

- `eye_tracker.py`: Main eye tracking implementation
- `data_collector.py`: Tool for collecting training data
- `requirements.txt`: Required Python packages
- `training_data/`: Directory for storing collected training data

## Future Improvements

- Implement CNN model for gaze direction classification
- Add LSTM for temporal pattern recognition
- Integrate with PyAutoGUI for mouse control
- Add calibration functionality
- Improve accuracy and reduce latency
- Add support for multiple users

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 