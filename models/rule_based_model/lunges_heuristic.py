import cv2
import mediapipe as mp
import numpy as np
import csv
import os

# Khởi tạo giải pháp MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_angle(a, b, c):
    """
    Tính góc giữa 3 điểm (landmark)
    a, b, c: Tọa độ (x, y, z) của các landmark
    """
    a = np.array(a) # Điểm đầu
    b = np.array(b) # Điểm giữa (đỉnh góc)
    c = np.array(c) # Điểm cuối
    
    # Tính toán vector
    ba = a - b
    bc = c - b
    
    # Tính góc (tính bằng radian)
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0)) # Giới hạn giá trị trong [-1, 1]
    
    # Chuyển sang độ
    return np.degrees(angle)

def get_landmark_coords(landmarks, landmark_name):
    """Trích xuất tọa độ [x, y, z] từ tên landmark"""
    lm = landmarks[mp_pose.PoseLandmark[landmark_name].value]
    return [lm.x, lm.y, lm.z]

def check_lunge_errors(landmarks):
    """
    Hàm Heuristic để kiểm tra lỗi.
    GIẢ ĐỊNH: Quay ngang, người quay mặt sang PHẢI (chân phải ở trước).
    Nếu người quay mặt sang TRÁI, bạn cần đảo ngược logic (ví dụ: dùng chân TRÁI).
    """
    errors = []
    
    # Lấy tọa độ các điểm mốc cần thiết
    # Giả định chân phải (RIGHT) là chân trước
    try:
        r_shoulder = get_landmark_coords(landmarks, 'RIGHT_SHOULDER')
        r_hip = get_landmark_coords(landmarks, 'RIGHT_HIP')
        r_knee = get_landmark_coords(landmarks, 'RIGHT_KNEE')
        r_ankle = get_landmark_coords(landmarks, 'RIGHT_ANKLE')
        r_foot_index = get_landmark_coords(landmarks, 'RIGHT_FOOT_INDEX')
        
        # 1. Lỗi: Thân người đổ về phía trước
        # Tính góc của thân người (Vai - Hông - Đầu gối)
        # Góc này nên giữ tương đối thẳng (> 70 độ)
        torso_angle = calculate_angle(r_shoulder, r_hip, r_knee)
        if torso_angle < 70:
            errors.append("Loi: Gap lung/Do nguoi ve phia truoc")
            
        # 2. Lỗi: Đầu gối trước vượt quá mũi chân
        # So sánh tọa độ x (chiều ngang) của đầu gối và mũi chân
        # Nếu x của đầu gối > x của mũi chân (với 1 khoảng đệm) -> Lỗi
        knee_x = r_knee[0]
        foot_index_x = r_foot_index[0]
        
        if knee_x > foot_index_x + 0.03: # 0.03 là khoảng đệm nhỏ
            errors.append("Loi: Dau goi truoc vuot mui chan")
            
    except Exception as e:
        # Bỏ qua nếu không tìm thấy landmark
        pass
        
    return ", ".join(errors)

def process_video(video_path, csv_writer):
    """
    Xử lý video, đếm reps, kiểm tra lỗi và ghi CSV.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Khong the mo file video {video_path}")
        return

    # Biến trạng thái để đếm rep
    rep_count = 0
    state = "up"  # Trạng thái hiện tại (up hoặc down)
    frame_id = 0
    
    # Biến để lưu trữ dữ liệu cho CSV
    current_rep_data = [] # List để lưu (frame_id, landmark_data)
    rep_error_frame_count = 0
    start_frame = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_id += 1
        delay = 1 # Mặc định là 1ms
        
        # Chuyển đổi màu BGR sang RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Xử lý pose
        results = pose.process(image)
        
        # Chuyển đổi màu RGB sang BGR để vẽ
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Trích xuất và vẽ landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            # --- Logic đếm Rep ---
            # Lấy góc của cả hai đầu gối
            # (Giả định một trong hai sẽ là chân trụ)
            l_knee = get_landmark_coords(landmarks, 'LEFT_KNEE')
            l_hip = get_landmark_coords(landmarks, 'LEFT_HIP')
            l_ankle = get_landmark_coords(landmarks, 'LEFT_ANKLE')
            left_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
            
            r_knee = get_landmark_coords(landmarks, 'RIGHT_KNEE')
            r_hip = get_landmark_coords(landmarks, 'RIGHT_HIP')
            r_ankle = get_landmark_coords(landmarks, 'RIGHT_ANKLE')
            right_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
            
            # Kiểm tra trạng thái "down" (xuống)
            # Nếu một trong hai gối gập dưới 110 độ
            if (left_knee_angle < 110 or right_knee_angle < 110) and state == "up":
                state = "down"
                start_frame = frame_id
                current_rep_data = []
                rep_error_frame_count = 0
            
            # Kiểm tra trạng thái "up" (đứng lên)
            # Nếu cả hai gối duỗi (trên 160 độ)
            if (left_knee_angle > 160 and right_knee_angle > 160) and state == "down":
                state = "up"
                rep_count += 1
                end_frame = frame_id
                
                # --- Xử lý dữ liệu Rep và ghi CSV ---
                total_frames_in_rep = len(current_rep_data)
                
                # Kiểm tra tỷ lệ lỗi
                label = "DUNG"
                if total_frames_in_rep > 0:
                    error_ratio = rep_error_frame_count / total_frames_in_rep
                    if error_ratio > 0.20:
                        label = "SAI"
                        
                # Ghi tất cả các frame của rep này vào CSV
                for frame_data in current_rep_data:
                    # Dữ liệu: [rep_count, start_frame, end_frame, ...99 coords..., label]
                    row = [rep_count, start_frame, end_frame] + frame_data + [label]
                    csv_writer.writerow(row)
                
                current_rep_data = [] # Xóa bộ đệm
            
            # --- Kiểm tra lỗi và lưu dữ liệu khi ở trạng thái "down" ---
            error_message = ""
            if state == "down":
                # 1. Kiểm tra lỗi
                error_message = check_lunge_errors(landmarks)
                
                # 2. Lưu trữ landmark data cho frame này
                frame_landmarks_flat = []
                for lm in landmarks:
                    frame_landmarks_flat.extend([lm.x, lm.y, lm.z])
                current_rep_data.append(frame_landmarks_flat)
                
                # 3. Xử lý hiển thị lỗi
                if error_message:
                    rep_error_frame_count += 1
                    cv2.putText(image, error_message, (50, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    delay = 300 # Giữ frame 0.3 giây
            
            # Vẽ landmarks lên frame
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                      mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
            
            # Hiển thị số Rep
            cv2.putText(image, 'REPS: ' + str(rep_count), (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, 'STATE: ' + state, (50, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        except Exception as e:
            # Bỏ qua nếu không phát hiện được pose
            # print(f"Frame {frame_id}: Khong phat hien duoc pose - {e}")
            pass
            
        # Hiển thị
        cv2.imshow('MediaPipe Pose - Lunge Analysis', image)
        
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

def main():
    VIDEO_SOURCE = "data/lunges/your_lunge_video.mp4" 
    
    CSV_OUTPUT_FILE = "lunge_analysis_results.csv"

    # Tạo header cho file CSV
    # 33 landmarks * 3 (x, y, z) = 99 cột
    landmarks_header = []
    for i in range(33):
        landmarks_header.extend([f'lm_{i}_x', f'lm_{i}_y', f'lm_{i}_z'])
        
    csv_header = ['rep', 'start_frame', 'end_frame'] + landmarks_header + ['label']

    # Mở file CSV để ghi
    try:
        with open(CSV_OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(csv_header)
            
            print(f"Bat dau xu ly video: {VIDEO_SOURCE}")
            print(f"Ghi ket qua vao: {CSV_OUTPUT_FILE}")
            
            # Kiểm tra xem VIDEO_SOURCE có tồn tại không
            if not os.path.exists(VIDEO_SOURCE):
                print(f"LOI: Khong tim thay file video tai '{VIDEO_SOURCE}'.")
                print("Vui long cap nhat bien 'VIDEO_SOURCE' trong ham main().")
                return

            process_video(VIDEO_SOURCE, csv_writer)
            
            print("Xu ly hoan tat.")
            
    except IOError as e:
        print(f"LOI: Khong the mo file CSV. {e}")
    except Exception as e:
        print(f"Da xay ra loi: {e}")

if __name__ == "__main__":
    main()