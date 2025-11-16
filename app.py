import os
import uuid
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder 
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash

# === SỬA ĐỔI: Import CẢ HAI analyzer ===
# Import tệp của Lateral Raise và đặt bí danh (alias) là 'lr'
from models.rule_based_model.latter_raise_analyze import (
    analyze_video_to_dataframe as lr_analyze_video_to_dataframe, 
    create_annotated_video as lr_create_annotated_video
)
# Import tệp của Biceps Curl và đặt bí danh (alias) là 'bc'
from models.rule_based_model.bicep_curl_analyze import (
    analyze_video_to_dataframe as bc_analyze_video_to_dataframe, 
    create_annotated_video as bc_create_annotated_video
)
# ==========================================

app = Flask(__name__)

# Cấu hình đường dẫn
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
PROCESSED_FOLDER = os.path.join(BASE_DIR, 'static', 'processed')
MODEL_FOLDER = os.path.join(BASE_DIR, 'models', 'classification_model')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER


# --- Vô hiệu hóa cache ---
@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response
# --------------------------

# === SỬA ĐỔI: Hàm này giờ nhận model_path ===
def perform_ml_prediction(df_summary, model_path):
    """
    Thực hiện tổng hợp và dự đoán ML trên
    DataFrame chứa dữ liệu thô của các frame.
    """
    if df_summary.empty:
        return {}, df_summary

    df_summary = df_summary[df_summary['rep_id'] > 0].copy()
    if df_summary.empty:
        print("Không tìm thấy rep nào trong video.")
        return {}, df_summary

    # --- 1. Xác định các cột đặc trưng ---
    feature_columns = [col for col in df_summary.columns if col.startswith('lm_')]
    
    # --- 2. Tổng hợp (Aggregate) dữ liệu theo Rep ---
    agg_funcs = ['mean', 'std', 'min', 'max']
    agg_dict = {col: agg_funcs for col in feature_columns}
    
    print(f"Đang tổng hợp dữ liệu cho {df_summary['rep_id'].nunique()} reps...")
    df_reps = df_summary.groupby('rep_id').agg(agg_dict)
    
    df_reps.columns = ['_'.join(col).strip() for col in df_reps.columns.values]

    # --- 3. Xử lý NaNs ---
    X_reps = df_reps.fillna(0)
    
    # --- 4. Tải Mô hình và Dự đoán (Sử dụng model_path) ---
    # model_path đã được cung cấp
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy tệp mô hình tại: {model_path}")
        return {}, df_summary

    print(f"Dự đoán trên X có shape: {X_reps.shape}")
    ml_predictions_numeric = model.predict(X_reps) 
    
    ml_labels = ['Correct' if p == 0 else 'Incorrect' for p in ml_predictions_numeric]
    
    # --- 5. Xây dựng Kết quả Chi tiết (VỚI LOGIC GHI ĐÈ) ---
    rep_results = {}
    label_map = {}
    
    print("Đang xây dựng kết quả chi tiết...")
    for rep_id, label_from_ml in zip(df_reps.index, ml_labels):
        rep_data = {}
        final_label = label_from_ml 
        reason = ""

        rep_frames = df_summary[df_summary['rep_id'] == rep_id]
        
        # --- LOGIC GHI ĐÈ MỚI ---
        error_counts = rep_frames[rep_frames['rule_error'] != '']['rule_error'].value_counts()
        
        if not error_counts.empty:
            final_label = 'Incorrect'
            reason = error_counts.idxmax() 
        else:
            if final_label == 'Incorrect':
                reason = "Lỗi (không rõ nguyên nhân)" 
            else:
                reason = "" 
            
        rep_data['label'] = final_label
        rep_data['reason'] = reason
        rep_results[rep_id] = rep_data
        label_map[rep_id] = final_label 

    df_summary['ml_label'] = df_summary['rep_id'].map(label_map)
    df_summary['ml_label'] = df_summary['ml_label'].fillna('N/A')

    return rep_results, df_summary
# ==========================================


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    
    # --- Kiểm tra tệp và bài tập ---
    if 'video' not in request.files:
        print("Lỗi: 'video' không có trong request.files")
        flash("Không tìm thấy tệp video. Hãy thử lại.")
        return redirect(url_for('index'))
    
    file = request.files['video']
    
    if file.filename == '':
        print("Lỗi: Tên tệp trống.")
        flash("Bạn chưa chọn tệp video nào.")
        return redirect(url_for('index'))

    exercise = request.form.get('exercise')
    
    if not exercise or exercise == "": # Kiểm tra nếu người dùng chưa chọn
        print("Lỗi: Chưa chọn bài tập.")
        flash("Bạn chưa chọn bài tập.")
        return redirect(url_for('index'))
    # --- Kết thúc kiểm tra ---
    
    
    # === CẬP NHẬT: Logic IF/ELIF ===
    if file:
        unique_id = str(uuid.uuid4())
        input_filename = f"{unique_id}_input.mp4"
        output_filename = f"{unique_id}_processed.mp4"
        csv_filename = f"{unique_id}_log.csv"

        input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
        csv_path = os.path.join(app.config['PROCESSED_FOLDER'], csv_filename)

        file.save(input_path)
        print(f"File saved to {input_path}")
        
        # Khai báo biến
        analyze_func = None
        create_video_func = None
        model_name = ""

        if exercise == 'lateral_raise':
            print("--- Chọn bài tập: Lateral Raise ---")
            analyze_func = lr_analyze_video_to_dataframe
            create_video_func = lr_create_annotated_video
            model_name = 'latteral_raise_best_model.joblib'
        
        elif exercise == 'bicep_curl':
            print("--- Chọn bài tập: Biceps Curl ---")
            analyze_func = bc_analyze_video_to_dataframe
            create_video_func = bc_create_annotated_video
            model_name = 'bicep_curl_best_model.joblib'
            
        else:
            flash("Bài tập không hợp lệ.")
            return redirect(url_for('index'))

        # Xây dựng đường dẫn mô hình
        model_path = os.path.join(app.config['MODEL_FOLDER'], model_name)

        try:
            # === LƯỢT 1: Phân tích Video -> DataFrame ===
            print("Bắt đầu Lượt 1: Phân tích video...")
            (df_summary, fps, rep_time_boundaries_original) = analyze_func(input_path)
            print("Lượt 1 Hoàn thành.")

            # === LƯỢT 2: Dự đoán ML ===
            print("Bắt đầu Lượt 2: Dự đoán ML...")
            (rep_results_ml, df_summary_with_labels) = perform_ml_prediction(df_summary, model_path)
            print(f"Lượt 2 Hoàn thành. Kết quả ML (đã ghi đè): {rep_results_ml}")

            df_summary_with_labels.to_csv(csv_path, index=False, encoding='utf-8')
            print(f"File CSV đã lưu tại: {csv_path}")

            # === LƯỢT 3: Tạo Video Chú Thích ===
            print("Bắt đầu Lượt 3: Tạo video chú thích...")
            
            processed_rep_times = create_video_func(
                input_path, 
                output_path, 
                df_summary_with_labels, 
                rep_results_ml,
                fps,
                rep_time_boundaries_original 
            )
            print(f"Lượt 3 Hoàn thành. Mốc thời gian đã xử lý: {processed_rep_times}")

            # === XÂY DỰNG KẾT QUẢ CUỐI CÙNG ===
            final_rep_results = {}
            for rep_id, ml_data in rep_results_ml.items():
                final_rep_results[rep_id] = ml_data
                
                if rep_id in processed_rep_times:
                    final_rep_results[rep_id]['start_time'] = processed_rep_times[rep_id]['start_time']
                else:
                    final_rep_results[rep_id]['start_time'] = 0 

        except Exception as e:
            print(f"Error processing video: {e}")
            import traceback
            traceback.print_exc()
            flash(f"Đã xảy ra lỗi khi xử lý video: {e}")
            return redirect(url_for('index'))
        finally:
            if os.path.exists(input_path):
                os.remove(input_path)
                print(f"Removed input file: {input_path}")
        
        sorted_rep_results = dict(sorted(final_rep_results.items()))
        
        return render_template('result.html', 
                               video_file=output_filename, 
                               csv_file=csv_filename, 
                               rep_results=sorted_rep_results) 

    return redirect(url_for('index'))

@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)