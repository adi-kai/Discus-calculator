# Discus-calculator
A computer vision tool that analyzes discus throw videos to track flight trajectory, classify throw phases, and predict landing distance.
Built with OpenCV and MediaPipe.

What it does:
* Detects and tracks the discus across video frames using background subtraction and circularity filtering
* Overlays pose landmarks and skeleton tracking on the thrower in real time
* Classifies throw phase (backswing, approach, release, follow-through) frame by frame
* Draws velocity vectors on key joints
* Predicts landing distance from the tracked trajectory arc
* Generates 6-panel analysis graphs including bird's eye view, side arc, 3D perspective, and velocity field

How to use:
* Install dependencies (see below)
* Run main.py
* Click Open Video and select your throw footage
* Draw a box around the disc flight zone when prompted
* Use the playback controls to step through the annotated video
* Click Show Graphs for full trajectory analysis

Dependencies:
pip install opencv-python mediapipe==0.10.35 numpy Pillow matplotlib
Notes

Helpful tips:
1. The pose model (pose_landmarker.task) downloads automatically on first run (~20MB)
2. Works best with footage where the disc is clearly visible against the background
