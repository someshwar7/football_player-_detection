import cv2
import numpy as np
from ultralytics import YOLO
from scipy.spatial.distance import cdist
from collections import deque
import colorsys
import os

# PlayerTracker class definition
class PlayerTracker:
    def __init__(self, player_id, initial_position, team):
        self.id = player_id
        self.positions = deque(maxlen=30)  # Store last 30 positions
        self.positions.append(initial_position)
        self.step_count = 0
        self.team = team
        self.color = self.generate_color()
        self.velocity = [0, 0]
        self.last_step_time = 0

    def generate_color(self):
        hue = self.id * 0.1 % 1.0
        return tuple(int(x * 255) for x in colorsys.hsv_to_rgb(hue, 0.8, 1.0))

    def update_position(self, new_position, current_time):
        if self.positions:
            self.velocity = [new_position[0] - self.positions[-1][0],
                             new_position[1] - self.positions[-1][1]]
        self.positions.append(new_position)
        self.update_step_count(current_time)

    def update_step_count(self, current_time):
        if len(self.positions) < 2:
            return

        # Calculate acceleration
        if len(self.positions) >= 3:
            prev_velocity = [self.positions[-2][0] - self.positions[-3][0],
                             self.positions[-2][1] - self.positions[-3][1]]
            acceleration = [self.velocity[0] - prev_velocity[0],
                            self.velocity[1] - prev_velocity[1]]
            acceleration_magnitude = np.linalg.norm(acceleration)

            # Detect step based on acceleration peak
            if acceleration_magnitude > 0.5 and (current_time - self.last_step_time) > 0.2:  # Adjust thresholds as needed
                self.step_count += 1
                self.last_step_time = current_time

# Detect team function
def detect_team(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox[:4])
    roi = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Define color ranges for team jerseys (adjust these based on actual jersey colors)
    team1_lower = np.array([0, 100, 100])  # Red for team 1
    team1_upper = np.array([10, 255, 255])
    team2_lower = np.array([100, 100, 100])  # Blue for team 2
    team2_upper = np.array([130, 255, 255])
    
    mask1 = cv2.inRange(hsv, team1_lower, team1_upper)
    mask2 = cv2.inRange(hsv, team2_lower, team2_upper)
    
    return 1 if np.sum(mask1) > np.sum(mask2) else 2

def main():
    # Specify the input video path directly in the code
    input_video_path = "E:/DS project/WhatsApp Video 2024-09-17 at 11.29.00_fbaf2aaf.mp4"
    
    # Validate the input video path
    if not os.path.exists(input_video_path):
        print(f"Error: The file {input_video_path} does not exist.")
        return

    # Specify the output video path on disk E
    output_video_path = "E:/output_video.mp4"

    model = YOLO('yolov8n.pt')
    video = cv2.VideoCapture(input_video_path)
    
    if not video.isOpened():
        print(f"Error: Unable to open video file {input_video_path}")
        return

    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Error: Unable to create output video file {output_video_path}")
        video.release()
        return

    players = {}
    next_id = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break

        current_time = video.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Current time in seconds
        results = model(frame)

        # Process detections
        detections = results[0].boxes.data
        for detection in detections:
            bbox = detection[:4].tolist()
            conf, class_id = detection[4:6]
            if int(class_id) == 0:  # Assuming 0 is the class ID for person
                center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                team = detect_team(frame, bbox)

                # Initialize new player or update existing player
                if not players:
                    players[next_id] = PlayerTracker(next_id, center, team)
                    next_id += 1
                else:
                    distances = cdist([center], [p.positions[-1] for p in players.values()])
                    closest_player_id = min(players.keys(), key=lambda i: distances[0][i])
                    
                    if distances[0][closest_player_id] < 50:  # Adjust threshold as needed
                        players[closest_player_id].update_position(center, current_time)
                    else:
                        players[next_id] = PlayerTracker(next_id, center, team)
                        next_id += 1

        # Visualize results
        for player in players.values():
            if player.positions:
                cv2.putText(frame, f"Player {player.id} (Team {player.team}): {player.step_count} steps", 
                            (int(player.positions[-1][0]), int(player.positions[-1][1])), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, player.color, 2)

        # Write the frame to the output video
        out.write(frame)

        # Display the frame
        cv2.imshow('Football Player Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Output video saved as: {output_video_path}")

    # Print final step counts
    for player_id, player in players.items():
        print(f"Player {player_id} (Team {player.team}): {player.step_count} steps")

if __name__ == "__main__":
    main()
