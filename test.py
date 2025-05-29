import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd

def load_model_and_scalers():
    # Tải mô hình đã huấn luyện
    model = tf.keras.models.load_model('lung_cancer_model.h5')
    
    # Tải dữ liệu gốc để fit scaler
    data = pd.read_csv('Lung_cancer/survey lung cancer.csv')
    
    # Chuẩn bị scaler
    le = LabelEncoder()
    scaler = StandardScaler()
    
    # Fit LabelEncoder cho cột GENDER
    data['GENDER'] = le.fit_transform(data['GENDER'])
    
    # Fit StandardScaler
    X = data.drop('LUNG_CANCER', axis=1)
    scaler.fit(X)
    
    return model, le, scaler

def predict_lung_cancer():
    # Nhập thông tin từ người dùng
    print("\n=== NHẬP THÔNG TIN BỆNH NHÂN ===")
    while True:
        gender = input("Giới tính (M/F): ").upper()
        if gender in ['M', 'F']:
            break
        print("Lỗi: Vui lòng nhập 'M' hoặc 'F'")
    
    while True:
        try:
            age = int(input("Tuổi: "))
            if 0 <= age <= 120:
                break
            print("Lỗi: Tuổi phải từ 0 đến 120")
        except ValueError:
            print("Lỗi: Vui lòng nhập số nguyên")

    def get_binary_input(prompt):
        while True:
            try:
                value = int(input(prompt))
                if value in [1, 2]:
                    return value
                print("Lỗi: Vui lòng nhập 1 (Không) hoặc 2 (Có)")
            except ValueError:
                print("Lỗi: Vui lòng nhập số 1 hoặc 2")

    smoking = get_binary_input("Hút thuốc (1-Không, 2-Có): ")
    yellow_fingers = get_binary_input("Ngón tay vàng (1-Không, 2-Có): ")
    anxiety = get_binary_input("Lo âu (1-Không, 2-Có): ")
    peer_pressure = get_binary_input("Áp lực từ bạn bè (1-Không, 2-Có): ")
    chronic_disease = get_binary_input("Bệnh mãn tính (1-Không, 2-Có): ")
    fatigue = get_binary_input("Mệt mỏi (1-Không, 2-Có): ")
    allergy = get_binary_input("Dị ứng (1-Không, 2-Có): ")
    wheezing = get_binary_input("Thở khò khè (1-Không, 2-Có): ")
    alcohol = get_binary_input("Uống rượu (1-Không, 2-Có): ")
    coughing = get_binary_input("Ho (1-Không, 2-Có): ")
    shortness_breath = get_binary_input("Khó thở (1-Không, 2-Có): ")
    swallow_diff = get_binary_input("Khó nuốt (1-Không, 2-Có): ")
    chest_pain = get_binary_input("Đau ngực (1-Không, 2-Có): ")
    
    # Tải mô hình và các scaler
    model, le, scaler = load_model_and_scalers()
    
    # Chuyển đổi giới tính
    gender_encoded = le.transform([gender])[0]
    
    # Tạo mảng đầu vào
    input_data = np.array([[gender_encoded, age, smoking, yellow_fingers, anxiety,
                           peer_pressure, chronic_disease, fatigue, allergy,
                           wheezing, alcohol, coughing, shortness_breath,
                           swallow_diff, chest_pain]])
    
    # Chuẩn hóa dữ liệu
    input_scaled = scaler.transform(input_data)
    
    # Reshape dữ liệu cho CNN
    input_reshaped = input_scaled.reshape(-1, input_scaled.shape[1], 1)
    
    # Dự đoán
    prediction = model.predict(input_reshaped, verbose=0)[0][0]
    
    # In kết quả
    print("\n=== KẾT QUẢ CHẨN ĐOÁN ===")
    print(f"Xác suất mắc ung thư phổi: {prediction*100:.2f}%")
    print(f"Chẩn đoán: {'Có nguy cơ cao' if prediction >= 0.5 else 'Nguy cơ thấp'}")
    
    if prediction >= 0.5:
        print("\nKHUYẾN NGHỊ: Vui lòng đến gặp bác sĩ để kiểm tra chi tiết!")
    else:
        print("\nKHUYẾN NGHỊ: Tiếp tục theo dõi sức khỏe và kiểm tra định kỳ.")

if __name__ == "__main__":
    print("=== CHƯƠNG TRÌNH CHẨN ĐOÁN NGUY CƠ UNG THƯ PHỔI ===")
    while True:
        predict_lung_cancer()
        while True:
            again = input("\nBạn có muốn chẩn đoán cho bệnh nhân khác không? (y/n): ").lower()
            if again in ['y', 'n']:
                break
            print("Lỗi: Vui lòng nhập 'y' hoặc 'n'")
        if again != 'y':
            break
    print("\nCảm ơn bạn đã sử dụng chương trình!")
