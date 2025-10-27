import cv2
import mediapipe as mp
import csv
import numpy as np

LANDMARK_NAMES = {
    11: "LEFT_SHOULDER", 12: "RIGHT_SHOULDER", 13: "LEFT_ELBOW",
    14: "RIGHT_ELBOW", 15: "LEFT_WRIST", 16: "RIGHT_WRIST",
    23: "LEFT_HIP", 24: "RIGHT_HIP", 25: "LEFT_KNEE", 26: "RIGHT_KNEE",
}

VIEW_DEFINITIONS = {
    'left': [11, 13, 23, 25],
    'right': [12, 14, 24, 26],
    'front': [11, 12, 13, 14, 23, 24, 25, 26]
}

def detect_view_from_landmarks(landmarks, threshold):
    try:
        left_vis = [landmarks[idx][3] for idx in VIEW_DEFINITIONS['left']]
        right_vis = [landmarks[idx][3] for idx in VIEW_DEFINITIONS['right']]
        avg_left = sum(left_vis) / len(left_vis)
        avg_right = sum(right_vis) / len(right_vis)
        
        is_left_clear = avg_left > threshold
        is_right_clear = avg_right > threshold
        
        if is_left_clear and is_right_clear: return 'front'
        elif is_left_clear and not is_right_clear: return 'left'
        elif not is_left_clear and is_right_clear: return 'right'
        else: return None
            
    except (IndexError, ZeroDivisionError, TypeError):
        return None

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# --- 1. SETUP INPUT AND OUTPUT ---
video_path = 'data/latteral_raise_raw/test.webm'
visibility_threshold = 0.70 

# <<< NEW: Đặt chiều cao tối đa cho cửa sổ hiển thị (tính bằng pixel)
MAX_DISPLAY_HEIGHT = 800
# --- ---

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Cannot open video file {video_path}")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
aspect_ratio = frame_width / frame_height # <<< NEW: Tính tỉ lệ khung hình

# --- 2. SETUP VIDEO WRITER ---
output_video_path = 'output_video_debug.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# VideoWriter vẫn dùng kích thước GỐC
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height)) 

# --- 3. SETUP CSV FILE ---
csv_output_path = 'landmarks_debug.csv'
header = ['frame']
for i in range(33):
    header += [f'lmk_{i}_x', f'lmk_{i}_y', f'lmk_{i}_z', f'lmk_{i}_visibility']

with open(csv_output_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(header) 

    frame_count = 0
    detected_view = None 
    
    print(f"Processing video... Press 'q' to stop.")
    print(f"DEBUG MODE: Video sẽ tự động DỪNG LẠI nếu visibility < {visibility_threshold}")
    print("Nhấn phím bất kỳ trên cửa sổ video để tiếp tục.")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Video finished or frame could not be read.")
            break

        frame_count += 1
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        # Đây là frame gốc, có độ phân giải đầy đủ
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR) 

        # --- 4. EXTRACT LANDMARKS & WRITE TO CSV ---
        row = [frame_count]
        landmarks_list_for_checking = [] 
        
        if results.pose_landmarks:
            landmarks_pb2 = results.pose_landmarks.landmark
            for lmk in landmarks_pb2:
                landmarks_list_for_checking.append((lmk.x, lmk.y, lmk.z, lmk.visibility))
                row.extend([lmk.x, lmk.y, lmk.z, lmk.visibility])
        else:
            landmarks_list_for_checking = [(0,0,0,0)] * 33
            row.extend([0.0] * (33 * 4)) 
            
        csv_writer.writerow(row) 

        # --- 5. DEBUGGING & PAUSE LOGIC ---
        key_wait_time = 1 
        failing_landmarks_info = [] 

        if not results.pose_landmarks or len(landmarks_list_for_checking) < 33:
            pass 
        else:
            if detected_view is None:
                detected_view = detect_view_from_landmarks(landmarks_list_for_checking, visibility_threshold)
                if detected_view:
                    print(f"--- Frame {frame_count}: View detected and locked: '{detected_view}' ---")
            
            if detected_view:
                required_indices = VIEW_DEFINITIONS[detected_view]
                for idx in required_indices:
                    vis_score = landmarks_list_for_checking[idx][3] 
                    if vis_score < visibility_threshold:
                        name = LANDMARK_NAMES.get(idx, f"INDEX_{idx}")
                        failing_landmarks_info.append(f"{name} (Vis: {vis_score:.2f})")
        
        
        # --- 6. DRAW ON VIDEO FRAME (TRÊN FRAME GỐC) ---
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image_bgr,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS)
        
        # Thêm text... (vẫn vẽ trên frame gốc)
        cv2.putText(image_bgr, f"Frame: {frame_count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image_bgr, f"View: {detected_view}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Xử lý tạm dừng
        if failing_landmarks_info:
            key_wait_time = 0 
            
            print(f"--- Frame {frame_count}: DEBUG PAUSE ---")
            print(f"  Reason: Low visibility for view '{detected_view}'")
            print(f"  Failing landmarks: {', '.join(failing_landmarks_info)}")
            print("  Press any key in the video window to continue...")

            # Vẽ thông báo lỗi lên Video (vẫn là frame gốc)
            cv2.putText(image_bgr, "LOW VISIBILITY! PAUSED.", (frame_width // 2 - 300, frame_height // 2 - 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
            for i, info in enumerate(failing_landmarks_info):
                cv2.putText(image_bgr, info, (frame_width // 2 - 300, frame_height // 2 + (i * 40)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # --- 7. HANDLE DISPLAY & SAVING (MODIFIED) ---
        
        # 7a. LƯU FRAME GỐC, ĐẦY ĐỦ KÍCH THƯỚC
        out.write(image_bgr) 
        
        # 7b. TẠO FRAME HIỂN THỊ ĐÃ RESIZE
        # Tạo một bản copy để resize, giữ nguyên `image_bgr`
        display_frame = image_bgr.copy() 
        
        if frame_height > MAX_DISPLAY_HEIGHT:
            # Tính chiều rộng mới dựa trên tỉ lệ
            display_width = int(MAX_DISPLAY_HEIGHT * aspect_ratio)
            # Resize frame
            display_frame = cv2.resize(display_frame, (display_width, MAX_DISPLAY_HEIGHT))
        
        # 7c. HIỂN THỊ FRAME ĐÃ RESIZE
        cv2.imshow('MediaPipe Pose Debugger', display_frame) 

        # 7d. CHỜ PHÍM BẤM
        key = cv2.waitKey(key_wait_time)
        if key & 0xFF == ord('q'):
            print("User pressed 'q', stopping...")
            break
        elif key_wait_time == 0:
            print("  Resuming...")

# --- 8. CLEANUP ---
print(f"Done. Processed {frame_count} frames.")
print(f"Landmark data saved to: {csv_output_path}")
print(f"Annotated video saved to: {output_video_path}")

pose.close()
cap.release()
out.release()
cv2.destroyAllWindows()