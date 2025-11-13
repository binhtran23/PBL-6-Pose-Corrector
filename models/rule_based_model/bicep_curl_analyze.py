import cv2
import mediapipe as mp
import numpy as np
import math
import pandas as pd
import time
from PIL import Image, ImageDraw, ImageFont
import os

# Khởi tạo các giải pháp MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# --- Kích thước Video Cố định ---
OUTPUT_DIM = (1280, 720) # (Width, Height)

# --- Nạp Font Unicode (Tiếng Việt) ---
FONT_PATH = "C:/Windows/Fonts/arial.ttf" 
if not os.path.exists(FONT_PATH):
    print(f"Cảnh báo: Không tìm thấy font tại {FONT_PATH}. Sẽ dùng font mặc định (có thể lỗi Tiếng Việt).")
    try:
        font_sm = ImageFont.load_default()
        font_md = ImageFont.load_default()
        font_lg = ImageFont.load_default()
        font_reason = ImageFont.load_default()
    except Exception as e:
        print(f"Không thể nạp font mặc định: {e}")
        raise SystemExit("Lỗi nghiêm trọng: Không thể nạp font chữ.")
else:
    try:
        font_sm = ImageFont.truetype(FONT_PATH, 20)
        font_md = ImageFont.truetype(FONT_PATH, 25)
        font_lg = ImageFont.truetype(FONT_PATH, 30)
        font_reason = ImageFont.truetype(FONT_PATH, 40)
    except IOError:
        print(f"Lỗi khi nạp font từ {FONT_PATH}. Sử dụng font mặc định.")
        font_sm = ImageFont.load_default()
        font_md = ImageFont.load_default()
        font_lg = ImageFont.load_default()
        font_reason = ImageFont.load_default()

# --- Hàm trợ giúp Vẽ Unicode ---
def draw_text_unicode(image_pil, text, position, font, fill=(255, 255, 255)):
    """Vẽ văn bản Unicode lên ảnh PIL."""
    draw = ImageDraw.Draw(image_pil)
    draw.text(position, text, font=font, fill=fill)

# --- Hàm trợ giúp Tính toán ---
def calculate_distance(a, b):
    try:
        return math.sqrt((a['px'] - b['px'])**2 + (a['py'] - b['py'])**2)
    except:
        return 0

def calculate_angle(a, b, c):
    # Tính toán góc 2D (x, y)
    a_np = np.array([a['x'], a['y']])
    b_np = np.array([b['x'], b['y']])
    c_np = np.array([c['x'], c['y']])
    
    radians = np.arctan2(c_np[1]-b_np[1], c_np[0]-b_np[0]) - np.arctan2(a_np[1]-b_np[1], a_np[0]-b_np[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def get_view(lm_coords):
    """Xác định góc nhìn dựa trên visibility và vị trí."""
    try:
        vis_left_hip = lm_coords.get(23, {}).get('vis', 0)
        vis_right_hip = lm_coords.get(24, {}).get('vis', 0)
        vis_left_shoulder = lm_coords.get(11, {}).get('vis', 0)
        vis_right_shoulder = lm_coords.get(12, {}).get('vis', 0)
        
        nose_x = lm_coords.get(0, {}).get('x', 0.5)
        left_shoulder_x = lm_coords.get(11, {}).get('x', 0)
        right_shoulder_x = lm_coords.get(12, {}).get('x', 0)

        if (vis_left_hip > 0.8 and vis_right_hip < 0.2) or (vis_left_shoulder > 0.8 and vis_right_shoulder < 0.2):
            return "Side (Right View)"
        if (vis_right_hip > 0.8 and vis_left_hip < 0.2) or (vis_right_shoulder > 0.8 and vis_left_shoulder < 0.2):
            return "Side (Left View)"

        if vis_left_hip > 0.8 and vis_right_hip > 0.8:
            return "Front"
            
        if right_shoulder_x > nose_x and left_shoulder_x > nose_x:
             return "Side (Left View)"
        if right_shoulder_x < nose_x and left_shoulder_x < nose_x:
             return "Side (Right View)"

        return "Front"
    except Exception:
        return "Unknown"

# === LƯỢT 1: HÀM PHÂN TÍCH ===
def analyze_video_to_dataframe(video_path):
    """
    Xử lý video, phát hiện lỗi rule-based CỦA BICEP CURL và trả về DataFrame thô.
    """
    print(f"[Analyze Bicep] Bắt đầu phân tích: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Analyze Bicep] Lỗi: Không thể mở video '{video_path}'")
        return pd.DataFrame(), 0, {}

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("[Analyze Bicep] Cảnh báo: FPS = 0, đặt mặc định là 30.")
        fps = 30
        
    w_origin = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_origin = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Analyze Bicep] Video gốc: {w_origin}x{h_origin} @ {fps:.2f} FPS")
    print(f"[Analyze Bicep] Resize video về: {OUTPUT_DIM}")

    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    all_frames_data = []
    landmarks_header = []
    for i in range(33):
        landmarks_header.extend([f'lm_{i}_x', f'lm_{i}_y', f'lm_{i}_z', f'lm_{i}_vis'])

    frame_id = 0
    rep_count = 0
    state = "down" # Trạng thái của Bicep curl
    view = "Initializing"
    
    rep_label = "N/A"
    
    last_frame_time = time.time()
    last_wrist_pos = {'left': None, 'right': None}

    rep_time_boundaries = {} 

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        frame_id += 1
        
        try:
            image = cv2.resize(image, OUTPUT_DIM, interpolation=cv2.INTER_AREA)
        except Exception as e:
            print(f"Lỗi resize frame {frame_id}: {e}")
            continue
            
        h, w = OUTPUT_DIM[1], OUTPUT_DIM[0]

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        current_frame_error = "" 
        lm_coords = {}
        lm_data_row = [0.0] * (33 * 4) 

        if results.pose_landmarks:
            landmarks_raw = results.pose_landmarks.landmark
            lm_data_row_temp = []
            
            for id, lm in enumerate(landmarks_raw):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_coords[id] = {
                    'x': lm.x, 'y': lm.y, 'z': lm.z, 'vis': lm.visibility,
                    'px': cx, 'py': cy
                }
                lm_data_row_temp.extend([lm.x, lm.y, lm.z, lm.visibility])
            
            if len(lm_data_row_temp) == 132:
                 lm_data_row = lm_data_row_temp

            view = get_view(lm_coords)

            # === THAY ĐỔI LOGIC: BICEP CURL ===
            
            # Tính góc CÙI CHỎ (Elbow)
            avg_elbow_angle = 0
            try:
                left_elbow_angle = calculate_angle(lm_coords[11], lm_coords[13], lm_coords[15]) # Shoulder-Elbow-Wrist
                right_elbow_angle = calculate_angle(lm_coords[12], lm_coords[14], lm_coords[16])
                avg_elbow_angle = (left_elbow_angle + right_elbow_angle) / 2
            except Exception:
                pass 

            # Task 3: Đếm Rep (State Machine - DỰA TRÊN GÓC CÙI CHỎ)
            try:
                up_threshold = 70   # Gập tay (góc nhỏ)
                down_threshold = 150 # Duỗi tay (góc lớn)
                new_state = state

                if avg_elbow_angle < up_threshold:
                    new_state = "up"
                elif avg_elbow_angle > down_threshold:
                    new_state = "down"
                
                if new_state == "up" and state == "down":
                    rep_label = "In Progress"
                    rep_count += 1
                    print(f"[Analyze Bicep] Bat dau Rep {rep_count} tai frame {frame_id}")
                    rep_time_boundaries[rep_count] = {"start_frame": frame_id, "end_frame": frame_id}
                
                elif new_state == "down" and state == "up":
                    rep_label = f"Rep {rep_count} Done"
                    print(f"[Analyze Bicep] Ket thuc Rep {rep_count} tai frame {frame_id}")
                    if rep_count in rep_time_boundaries:
                        rep_time_boundaries[rep_count]["end_frame"] = frame_id

                state = new_state
            
            except Exception:
                pass

            # Task 2: Heuristics kiểm tra lỗi BICEP CURL
            if rep_label == "In Progress":
                
                # Lỗi 1: Dùng Lưng (Swinging) - Chỉ check side view
                if "Side" in view:
                    try:
                        side_idx = 23 if "Left View" in view else 24
                        ear = lm_coords[7] if "Left View" in view else 8
                        shoulder = lm_coords[11] if "Left View" in view else 12
                        hip = lm_coords[side_idx]
                        back_angle = calculate_angle(ear, shoulder, hip)
                        
                        # Nếu lưng không thẳng (ngả về sau HOẶC cúi về trước)
                        if back_angle < 170: # 180 là thẳng tuyệt đối
                            current_frame_error = "Lỗi: Dùng lưng (Swinging)"
                    except: pass

                # Lỗi 2: Đưa cùi chỏ ra trước
                try:
                    # Tính góc vai (Wrist-Shoulder-Hip)
                    l_shoulder_angle = calculate_angle(lm_coords[15], lm_coords[11], lm_coords[23])
                    r_shoulder_angle = calculate_angle(lm_coords[16], lm_coords[12], lm_coords[24])
                    avg_shoulder_angle = (l_shoulder_angle + r_shoulder_angle) / 2
                    
                    if avg_shoulder_angle > 50: # Nếu nhấc vai quá 50 độ
                        current_frame_error = "Lỗi: Đưa cùi chỏ ra trước"
                except: pass
                
                # Lỗi 3: Vung tạ quá nhanh (Giữ nguyên)
                try:
                    current_time = time.time()
                    delta_time = current_time - last_frame_time
                    if delta_time > 0.01: 
                        left_wrist, right_wrist = lm_coords[15], lm_coords[16] # Chú ý: 15 và 16
                        if last_wrist_pos['left'] is not None:
                            left_dist = calculate_distance(left_wrist, last_wrist_pos['left'])
                            right_dist = calculate_distance(right_wrist, last_wrist_pos['right'])
                            left_velocity = left_dist / delta_time
                            right_velocity = right_dist / delta_time
                            if left_velocity > 2000 or right_velocity > 2000:
                                current_frame_error = "Lỗi: Vung tạ quá nhanh"
                        last_wrist_pos['left'], last_wrist_pos['right'] = left_wrist, right_wrist
                    last_frame_time = current_time
                except: pass
            
            if rep_label == "In Progress":
                if rep_count in rep_time_boundaries:
                    rep_time_boundaries[rep_count]["end_frame"] = frame_id

        # --- Ghi dữ liệu frame này ---
        frame_data = {
            'frame_id': frame_id,
            'view': view,
            'state': state,
            'rep_id': rep_count if "In Progress" in rep_label or state == "up" else 0,
            'rule_error': current_frame_error,
        }
        for i, col_name in enumerate(landmarks_header):
            frame_data[col_name] = lm_data_row[i]
            
        all_frames_data.append(frame_data)

    cap.release()
    print(f"[Analyze Bicep] Đã phân tích {frame_id} frames.")
    
    df = pd.DataFrame(all_frames_data)
    
    rep_time_boundaries_sec = {}
    for rep_id, times in rep_time_boundaries.items():
        rep_time_boundaries_sec[rep_id] = {
            "start_time": times['start_frame'] / fps,
            "end_time": times['end_frame'] / fps
        }
    
    print(f"[Analyze Bicep] Hoàn thành. Tìm thấy {rep_count} reps.")
    return df, fps, rep_time_boundaries_sec

# === LƯỢT 3: HÀM TẠO VIDEO ===
# HÀM NÀY GIỐNG HỆT VỚI LATERAL RAISE
# Nó chỉ đọc từ DataFrame và vẽ lên, nên không cần sửa
def create_annotated_video(input_path, output_path, df_summary, rep_results_ml, fps, rep_time_boundaries_original):
    """
    Đọc lại video, vẽ chú thích (bao gồm lỗi động và slow-motion)
    và tính toán lại mốc thời gian.
    (Hàm này mang tính tổng quát, dùng chung cho cả hai bài tập)
    """
    print(f"[CreateVideo Bicep] Bắt đầu tạo video: {output_path}")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[CreateVideo Bicep] Lỗi: Không thể mở video gốc '{input_path}'")
        return {}

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, OUTPUT_DIM)

    if not out.isOpened():
        print("[CreateVideo Bicep] CẢNH BÁO: Không thể mở VideoWriter với codec 'avc1'.")
        print("[CreateVideo Bicep] Thử lại với codec 'mp4v' (có thể không phát được trên web).")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, OUTPUT_DIM)
        if not out.isOpened():
            print("[CreateVideo Bicep] LỖI NGHIÊM TRỌNG: Không thể khởi tạo VideoWriter.")
            cap.release()
            return {}

    frame_id = 0
    current_output_frame_count = 0
    new_rep_boundaries = {}
    
    rep_start_frames_original = {}
    for rep_id, times in rep_time_boundaries_original.items():
        rep_start_frames_original[rep_id] = int(times['start_time'] * fps)
    
    print("[CreateVideo Bicep] Bắt đầu vòng lặp ghi video...")
    
    df_dict = df_summary.set_index('frame_id').to_dict('index')

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        frame_id += 1

        try:
            image = cv2.resize(image, OUTPUT_DIM, interpolation=cv2.INTER_AREA)
        except Exception as e:
            print(f"Lỗi resize frame {frame_id}: {e}")
            continue

        frame_data = df_dict.get(frame_id, {})
        
        rep_id = frame_data.get('rep_id', 0)
        state = frame_data.get('state', 'down')
        view = frame_data.get('view', 'Unknown')
        
        reason_display = frame_data.get('rule_error', '')
        
        ml_label = ""
        if rep_id > 0:
            ml_label = rep_results_ml.get(rep_id, {}).get('label', '')
            if ml_label == "Correct":
                ml_label = "Đúng"
            elif ml_label == "Incorrect":
                ml_label = "Sai"

        # === VẼ LANDMARKS VÀ CONNECTIONS ===
        lm_pixels = {}
        for i in range(33):
            try:
                x = frame_data.get(f'lm_{i}_x', 0)
                y = frame_data.get(f'lm_{i}_y', 0)
                vis = frame_data.get(f'lm_{i}_vis', 0)
                
                if vis > 0.5:
                    lm_pixels[i] = {
                        'px': int(x * OUTPUT_DIM[0]),
                        'py': int(y * OUTPUT_DIM[1]),
                        'vis': vis
                    }
            except Exception:
                pass

        if mp_pose.POSE_CONNECTIONS:
            for connection in mp_pose.POSE_CONNECTIONS:
                try:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    if start_idx in lm_pixels and end_idx in lm_pixels:
                        start_point = (lm_pixels[start_idx]['px'], lm_pixels[start_idx]['py'])
                        end_point = (lm_pixels[end_idx]['px'], lm_pixels[end_idx]['py'])
                        cv2.line(image, start_point, end_point, (255, 255, 255), 2)
                except Exception:
                    pass
        
        for i, data in lm_pixels.items():
            try:
                cv2.circle(image, (data['px'], data['py']), 5, (230, 66, 245), -1) 
                cv2.circle(image, (data['px'], data['py']), 6, (255, 255, 255), 1)
            except Exception:
                pass
        
        # --- Chuyển sang PIL để vẽ ---
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        
        # --- Vẽ giao diện (UI) ---
        image_pil = image_pil.convert('RGBA')
        overlay_pil = Image.new('RGBA', image_pil.size, (255, 255, 255, 0))
        draw_overlay = ImageDraw.Draw(overlay_pil)
        
        rect_pos = [0, 0, 450, 230] 
        draw_overlay.rectangle(rect_pos, fill=(0, 0, 0, 150))
        
        image_pil = Image.alpha_composite(image_pil, overlay_pil)
        image_pil = image_pil.convert('RGB')
        
        draw_text_unicode(image_pil, f"FRAME: {frame_id}", (20, 20), font_md)
        draw_text_unicode(image_pil, f"VIEW: {view}", (20, 60), font_md)
        draw_text_unicode(image_pil, f"STATE: {state.upper()}", (20, 100), font_md)
        draw_text_unicode(image_pil, f"REP: {rep_id}", (20, 140), font_md)
        
        if ml_label:
            color = (0, 255, 0) if ml_label == "Đúng" else (255, 0, 0)
            draw_text_unicode(image_pil, f"ĐÁNH GIÁ: {ml_label}", (20, 180), font_lg, fill=color)

        if reason_display:
            try:
                text_bbox = font_reason.getbbox(reason_display)
                text_w = text_bbox[2] - text_bbox[0]
                text_h = text_bbox[3] - text_bbox[1]
                pos_x = (OUTPUT_DIM[0] - text_w) // 2
                pos_y = (OUTPUT_DIM[1] - text_h) // 2
                
                draw = ImageDraw.Draw(image_pil, 'RGBA')
                rect_pos = [pos_x - 10, pos_y - 10, pos_x + text_w + 10, pos_y + text_h + 10]
                draw.rectangle(rect_pos, fill=(0, 0, 0, 150))
                
                draw_text_unicode(image_pil, reason_display, (pos_x, pos_y), font_reason, fill=(255, 0, 0))
            except Exception as e:
                print(f"Lỗi khi vẽ text: {e}")

        image_annotated = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

        # --- Ghi frame ---
        out.write(image_annotated)
        current_output_frame_count += 1
        
        # --- Tính toán mốc thời gian ---
        for rep_id_orig, start_frame_orig in rep_start_frames_original.items():
            if frame_id == start_frame_orig:
                new_start_time_sec = (current_output_frame_count - 1) / fps
                new_rep_boundaries[rep_id_orig] = {"start_time": new_start_time_sec}
                print(f"[CreateVideo Bicep] Rep {rep_id_orig}: Mốc thời gian gốc {start_frame_orig} (frame) -> Mốc thời gian mới {new_start_time_sec:.2f} (giây)")

        # --- Slow-motion ---
        if reason_display:
            slow_mo_frames = int(fps * 0.5) 
            for _ in range(slow_mo_frames):
                out.write(image_annotated) 
                current_output_frame_count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"[CreateVideo Bicep] Đã tạo video thành công tại: {output_path}")
    
    return new_rep_boundaries