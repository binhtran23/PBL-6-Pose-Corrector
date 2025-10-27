import cv2
import mediapipe as mp
import numpy as np
import math
import csv
import time

# Khởi tạo các giải pháp MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    """(Đã sửa) Tính góc (0-180 độ) với b là đỉnh."""
    try:
        ang_rad = math.atan2(c['py'] - b['py'], c['px'] - b['px']) - \
                    math.atan2(a['py'] - b['py'], a['px'] - b['px'])
        ang_deg = math.degrees(ang_rad)
        ang_deg = ang_deg % 360
        if ang_deg > 180:
            ang_deg = 360 - ang_deg
        return abs(ang_deg)
    except:
        return 0

def calculate_distance(a, b):
    """Tính khoảng cách Euclidean 2D giữa 2 điểm."""
    try:
        return math.sqrt((a['px'] - b['px'])**2 + (a['py'] - b['py'])**2)
    except:
        return 0

def get_view(landmarks):
    """Xác định góc nhìn của camera dựa trên tọa độ Z của vai."""
    try:
        l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    except:
        return "Unknown"

    l_vis, r_vis = l_shoulder['vis'], r_shoulder['vis']
    l_z, r_z = l_shoulder['z'], r_shoulder['z']
    vis_thresh = 0.7
    z_diff = l_z - r_z
    front_thresh = 0.08
    diag_thresh = 0.25
    view = "Unknown"

    if l_vis > vis_thresh and r_vis > vis_thresh:
        if abs(z_diff) < front_thresh:
            view = "Front"
        elif z_diff > diag_thresh:
            view = "Left"
        elif z_diff < -diag_thresh:
            view = "Right"
        elif z_diff > front_thresh:
            view = "Diagonal Left"
        elif z_diff < -front_thresh:
            view = "Diagonal Right"
    elif l_vis > vis_thresh:
        view = "Right"
    elif r_vis > vis_thresh:
        view = "Left"
        
    return view

def process_video(video_path, output_csv_path='pose_log.csv'):
    
    # === 1. KHỞI TẠO ===
    cap = cv2.VideoCapture(video_path) 
    if not cap.isOpened():
        print(f"Error: Khong a mo duoc video tai '{video_path}'")
        return

    TARGET_WIDTH = 1200
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Chuẩn bị file CSV
    landmarks_header = []
    for i in range(33):
        landmarks_header.extend([f'lm_{i}_x', f'lm_{i}_y', f'lm_{i}_z', f'lm_{i}_vis'])
    csv_header = ['frame_id'] + landmarks_header + ['rep_label']
    
    csv_file = open(output_csv_path, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(csv_header)

    # Biến theo dõi trạng thái
    frame_id = 0
    rep_count = 0
    state = "down"
    view = "Initializing"
    
    # Biến logic chấm điểm rep
    current_rep_frames = 0
    current_rep_error_frames = 0
    rep_label = "N/A"
    rep_data_log = [] # Bộ đệm

    # Biến heuristic vận tốc
    last_frame_time = time.time()
    last_wrist_pos = {'left': None, 'right': None}

    # Biến debug lỗi
    error_display_counter = 0
    last_image_with_error = None
    error_message = ""

    # === 2. VÒNG LẶP CHÍNH XỬ LÝ VIDEO ===
    
    while cap.isOpened():
        
        # --- 2.1 Logic giữ frame lỗi ---
        if error_display_counter > 0:
            cv2.imshow('MediaPipe Pose Debug', last_image_with_error)
            cv2.waitKey(100)
            error_display_counter -= 1
            continue
            
        # --- 2.2 Đọc frame mới ---
        success, image = cap.read()
        if not success:
            break
        frame_id += 1

        # --- 2.3 Resize ---
        h, w = image.shape[:2]
        scale = TARGET_WIDTH / w
        dim = (TARGET_WIDTH, int(h * scale))
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        h, w = image.shape[:2]

        # --- 2.4 Xử lý MediaPipe ---
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        is_error_frame = False
        error_message = ""
        lm_coords = {}
        lm_data_row = [frame_id]
        avg_shoulder_angle = 0 # Khởi tạo

        # --- 2.5 Trích xuất Keypoints ---
        if results.pose_landmarks:
            landmarks_raw = results.pose_landmarks.landmark
            
            for id, lm in enumerate(landmarks_raw):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_coords[id] = {
                    'x': lm.x, 'y': lm.y, 'z': lm.z, 'vis': lm.visibility,
                    'px': cx, 'py': cy
                }
                lm_data_row.extend([lm.x, lm.y, lm.z, lm.visibility])

            # --- Task 1: Xác định View ---
            view = get_view(lm_coords)

            # --- Tính góc vai (cần cho cả Task 3 và Task 2) ---
            try:
                left_shoulder_angle = calculate_angle(lm_coords[13], lm_coords[11], lm_coords[23])
                right_shoulder_angle = calculate_angle(lm_coords[14], lm_coords[12], lm_coords[24])
                avg_shoulder_angle = (left_shoulder_angle + right_shoulder_angle) / 2
            except Exception as e:
                pass # Bỏ qua nếu thiếu keypoint

            # --- Task 3: Đếm Rep (State Machine - ĐÃ SỬA LOGIC) ---
            try:
                up_threshold = 75
                down_threshold = 30
                new_state = state

                # Xác định trạng thái mới dựa trên góc
                if avg_shoulder_angle > up_threshold:
                    new_state = "up"
                elif avg_shoulder_angle < down_threshold:
                    new_state = "down"
                
                # THAY ĐỔI 1: Bắt đầu rep (down -> up)
                # Bắt đầu ghi log, nhưng CHƯA đếm rep
                if new_state == "up" and state == "down":
                    rep_label = "In Progress"
                    rep_data_log = [] # Xóa bộ đệm cũ
                    current_rep_frames = 0
                    current_rep_error_frames = 0
                
                # THAY ĐỔI 2: Kết thúc rep (up -> down)
                # Đếm rep và xử lý log (ghi CSV)
                elif new_state == "down" and state == "up":
                    rep_count += 1 # ĐẾM REP KHI KẾT THÚC
                    rep_label = f"Rep {rep_count} Done"
                    
                    # --- Task 4 & 5: Xử lý Rep VỪA HOÀN THÀNH ---
                    if current_rep_frames > 0 and rep_data_log:
                        # (Ghi nốt frame cuối cùng này vào log)
                        current_rep_frames += 1
                        if is_error_frame: # (Kiểm tra lỗi của frame cuối)
                            current_rep_error_frames += 1
                        rep_data_log.append(lm_data_row)
                        
                        # Xử lý
                        error_ratio = current_rep_error_frames / current_rep_frames
                        final_label = "Incorrect" if error_ratio > 0.20 else "Correct"
                        
                        for frame_data in rep_data_log:
                            frame_data.append(final_label)
                            csv_writer.writerow(frame_data)
                    
                    rep_data_log = [] # Dọn dẹp bộ đệm

                state = new_state
            
            except Exception as e:
                pass

            # --- Task 2: Heuristics kiểm tra lỗi (theo từng frame) ---
            
            # Chỉ kiểm tra lỗi NẾU rep đang diễn ra
            if rep_label == "In Progress":
                
                # Lỗi 1: Lưng thẳng
                if view != "Front":
                    try:
                        side_idx = 23 if "Left" in view else 24
                        ear = lm_coords[7] if "Left" in view else 8
                        shoulder = lm_coords[11] if "Left" in view else 12
                        hip = lm_coords[side_idx]
                        back_angle = calculate_angle(ear, shoulder, hip)
                        if back_angle > 165:
                            is_error_frame = True
                            error_message = "LOI: Lung thang"
                    except: pass

                # Lỗi 2: Tay lên quá cao (Bất kể view) - (ĐÃ SỬA NGƯỠỠNG)
                try:
                    # THAY ĐỔI CHÍNH: Tăng ngưỡng từ 105 -> 115
                    if avg_shoulder_angle > 115: 
                        is_error_frame = True
                        error_message = "LOI: Tay len qua cao"
                except: pass

                # Lỗi 3: Nhún vai
                try:
                    l_ear_y, l_shoulder_y = lm_coords[7]['py'], lm_coords[11]['py']
                    r_ear_y, r_shoulder_y = lm_coords[8]['py'], lm_coords[12]['py']
                    left_shrug_dist = abs(l_ear_y - l_shoulder_y)
                    right_shrug_dist = abs(r_ear_y - r_shoulder_y)
                    shoulder_width = calculate_distance(lm_coords[11], lm_coords[12])
                    if shoulder_width > 0:
                        shrug_ratio = (left_shrug_dist + right_shrug_dist) / (2 * shoulder_width)
                        if shrug_ratio < 0.15:
                            is_error_frame = True
                            error_message = "LOI: Nhun vai (Vai qua cao)"
                except: pass

                # Lỗi 4: Vung tay quá nhanh
                try:
                    current_time = time.time()
                    delta_time = current_time - last_frame_time
                    if delta_time > 0.01: 
                        left_wrist, right_wrist = lm_coords[15], lm_coords[14]
                        if last_wrist_pos['left'] is not None:
                            left_dist = calculate_distance(left_wrist, last_wrist_pos['left'])
                            right_dist = calculate_distance(right_wrist, last_wrist_pos['right'])
                            left_velocity = left_dist / delta_time
                            right_velocity = right_dist / delta_time
                            if left_velocity > 2000 or right_velocity > 2000:
                                is_error_frame = True
                                error_message = "LOI: Vung tay qua nhanh"
                        last_wrist_pos['left'], last_wrist_pos['right'] = left_wrist, right_wrist
                    last_frame_time = current_time
                except: pass
            
            # --- Cập nhật bộ đếm/log (Data Logging) ---
            
            # Chỉ ghi log KHI rep đang diễn ra
            if rep_label == "In Progress":
                current_rep_frames += 1
                if is_error_frame:
                    current_rep_error_frames += 1
                rep_data_log.append(lm_data_row)


            # --- Task 6: Vẽ Keypoints (với threshold > 75%) ---
            connections = mp_pose.POSE_CONNECTIONS
            if connections:
                for connection in connections:
                    try:
                        start_idx, end_idx = connection[0], connection[1]
                        if lm_coords[start_idx]['vis'] > 0.75 and lm_coords[end_idx]['vis'] > 0.75:
                            start_point = (lm_coords[start_idx]['px'], lm_coords[start_idx]['py'])
                            end_point = (lm_coords[end_idx]['px'], lm_coords[end_idx]['py'])
                            cv2.line(image, start_point, end_point, (245, 117, 66), 2)
                    except: pass

            for id, lm in lm_coords.items():
                if lm['vis'] > 0.75:
                    cv2.circle(image, (lm['px'], lm['py']), 5, (245, 66, 230), -1)
                    cv2.circle(image, (lm['px'], lm['py']), 6, (255, 255, 255), 1)

            # --- Task 7: Kích hoạt Debug Frame Hold ---
            if is_error_frame:
                error_display_counter = 5
                cv2.putText(image, error_message, (50, h - 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                last_image_with_error = image.copy()

        
        # --- 2.6 Vẽ giao diện (UI) ---
        # (Phần này đã có frame_id, rep, view như bạn yêu cầu)
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (450, 220), (50, 50, 50), -1)
        alpha = 0.7
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        cv2.putText(image, f"VIEW: {view}", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image, f"REPS: {rep_count}", (10, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image, f"STATE: {state.upper()}", (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image, f"FRAME: {frame_id}", (10, 160), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image, f"LABEL: {rep_label}", (10, 200), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('MediaPipe Pose Debug', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # === 3. DỌN DẸP ===
    
    # Xử lý rep cuối cùng (nếu video kết thúc khi đang "In Progress")
    if rep_data_log:
        print("Xu ly rep cuoi cung (dang do)...")
        error_ratio = current_rep_error_frames / current_rep_frames if current_rep_frames > 0 else 0
        # Coi rep dở là 'Incorrect'
        final_label = "Incorrect"
        for frame_data in rep_data_log:
            frame_data.append(final_label)
            csv_writer.writerow(frame_data)

    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()
    print(f"Da xu ly xong. Du lieu duoc luu tai: {output_csv_path}")


# === CÁCH SỬ DỤNG ===
if __name__ == "__main__":
    # VIDEO_SOURCE = "video_lateral_raises.mp4" 
    VIDEO_SOURCE = "data/lateral raise/811033b9-636c-46d1-81dd-fea5337bf615.mp4"
    OUTPUT_CSV = "exercise_log.csv"
    
    process_video(VIDEO_SOURCE, OUTPUT_CSV)