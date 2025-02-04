# ECE Cup - Pi-Car Autonomous Robot

## 🏆 ECE Cup Project - Pi-Car Automation

This project was carried out as part of the ECE Cup. The goal was to automate a Pi-Car robot so that it could navigate autonomously by combining several features: line following, shape and color recognition using OpenCV, and obstacle detection through ultrasonic sensors.

---

## 🚀 Features

### 🔹 Line Following
- Use of an onboard camera to detect and follow a line on the ground.
- Image processing with OpenCV to extract contours and adjust the robot's trajectory.

### 🎨 Shape and Color Recognition
- Detection of geometric shapes and specific colors using OpenCV.
- Decision-making based on detected objects (e.g., changing direction according to encountered colors or shapes).

### 📡 Communication and Script Execution
- SSH connection to the Raspberry Pi of the Pi-Car to execute control scripts.
- Python interface allowing real-time command sending to the robot.

### 🏁 Maze Navigation and Obstacle Avoidance
- Integration of ultrasonic sensors to detect obstacles and adjust trajectory.
- Navigation algorithm to avoid collisions and efficiently move through a maze.

---

## 🔧 Technologies and Hardware Used

- **Hardware:** Pi-Car (Raspberry Pi-based robot), onboard camera, ultrasonic sensors.
- **Software & Frameworks:**
  - OpenCV (Image processing)
  - Python (Control and data processing)
  - SSH (Communication and script execution)
  - NumPy and SciPy (Mathematical processing)

---

## 🛠️ Installation and Execution

### 📥 Prerequisites
- A Raspberry Pi with Raspbian installed
- Python 3 and the following libraries:
  ```bash
  pip install opencv-python numpy scipy paramiko
  ```
- Configured SSH connection to execute remote commands

### 🚀 Execution
1. **SSH Connection:**
   ```bash
   ssh pi@robot_ip_address
   ```
2. **Run the main script:**
   ```bash
   sudo python3 soutenance.py
   ```
3. Observe the robot analyze its environment and move intelligently 🚗💨

---

## 📌 Possible Improvements
- Optimization of image processing algorithms for faster recognition.
- Addition of a graphical interface to facilitate robot control.
- Implementation of artificial intelligence to enhance maze navigation.

---

## 👥 Authors
- **Shaima Derouich** & Team 🎯
