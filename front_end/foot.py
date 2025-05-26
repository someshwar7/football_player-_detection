import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
from scipy.spatial.distance import cdist
from moviepy.video.io.VideoFileClip import VideoFileClip
from football_player_step_1 import PlayerTracker  # Import your custom tracker class
import base64


# Load YOLO model
@st.cache_resource
def load_yolo_model(model_path="football_player_detector.pt"):
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None


# Function for object detection
def detect_objects(frame, model):
    frame_copy = frame.copy()
    results = model(frame_copy)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().numpy()
            confidence = box.conf[0]
            class_id = box.cls[0].int().item()
            class_name = model.names[class_id]

            label = f"{class_name} {confidence:.2f}"
            color = (0, 255, 0) if class_name == "Ball" else \
                    (255, 0, 0) if class_name == "Player" else (0, 0, 255)

            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame_copy


# Function for step tracking
def track_player_steps(frame, model, players, next_id, current_time):
    frame_copy = frame.copy()
    results = model(frame_copy)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().numpy()
            class_id = box.cls[0].int().item()
            class_name = model.names[class_id]

            if class_name == "Player":
                center = ((x1 + x2) / 2, (y1 + y2) / 2)

                # If no players are being tracked, add a new player
                if not players:
                    players[next_id] = PlayerTracker(next_id, center, "Unknown")
                    next_id += 1
                else:
                    # Calculate distances between detected center and tracked players' last positions
                    distances = cdist([center], [p.positions[-1] for p in players.values()])
                    closest_id = min(players.keys(), key=lambda i: distances[0][list(players.keys()).index(i)])

                    # Update the closest player's position or add a new player if too far
                    if distances[0][list(players.keys()).index(closest_id)] < 50:
                        players[closest_id].update_position(center, current_time)
                    else:
                        players[next_id] = PlayerTracker(next_id, center, "Unknown")
                        next_id += 1

                # Display the tracked player's ID and step count
                tracked_id = list(players.keys())[-1]
                cv2.putText(
                    frame_copy,
                    f"Player {players[tracked_id].id}: {players[tracked_id].step_count} steps",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    2,
                )

    return frame_copy, players, next_id



# Process video and return paths for generated videos
def process_video(video_path, model):
    players = {}
    next_id = 0
    detection_video_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    steps_video_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name

    with VideoFileClip(video_path) as clip:
        def detection_frame(frame):
            return detect_objects(frame, model)

        def steps_frame(frame):
            nonlocal players, next_id
            current_time = clip.reader.pos / clip.fps
            return track_player_steps(frame, model, players, next_id, current_time)[0]

        detection_clip = clip.fl_image(detection_frame)
        detection_clip.write_videofile(detection_video_path, codec="libx264")

        steps_clip = clip.fl_image(steps_frame)
        steps_clip.write_videofile(steps_video_path, codec="libx264")

    return detection_video_path, steps_video_path


# Generate video HTML with unified controls
def generate_video_html(original_path, detection_path, steps_path):
    def video_to_base64(video_path):
        with open(video_path, "rb") as file:
            return base64.b64encode(file.read()).decode()

    original_base64 = video_to_base64(original_path)
    detection_base64 = video_to_base64(detection_path)
    steps_base64 = video_to_base64(steps_path)

    return f"""
    <div style="display: flex; flex-direction: row; justify-content: space-around;">
        <video id="original" width="500" height="500" controls>
            <source src="data:video/mp4;base64,{original_base64}" type="video/mp4">
        </video>
        <video id="detection" width="500" height="500" controls>
            <source src="data:video/mp4;base64,{detection_base64}" type="video/mp4">
        </video>
        <video id="steps" width="500" height="500" controls>
            <source src="data:video/mp4;base64,{steps_base64}" type="video/mp4">
        </video>
    </div>
    <div style="margin-top: 20px;">
        <button onclick="playAll()">Play All</button>
        <button onclick="pauseAll()">Pause All</button>
    </div>
    <script>
        const videos = document.querySelectorAll('video');

        function playAll() {{
            videos.forEach(video => video.play());
        }}

        function pauseAll() {{
            videos.forEach(video => video.pause());
        }}
    </script>
    """

# Streamlit App
def main():
    st.title("Football Detection System")

    model = load_yolo_model()
    video_file = st.file_uploader("Upload a football match video", type=["mp4", "mov", "avi"])

    if video_file and model:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(video_file.read())

        st.info("Processing video. Please wait...")

        detection_video_path, steps_video_path = process_video(temp_file.name, model)

        st.success("Processing complete! View results below.")

        video_html = generate_video_html(temp_file.name, detection_video_path, steps_video_path)
        st.components.v1.html(video_html, height=700)


if __name__ == "__main__":
    main()
