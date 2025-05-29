# Hệ Thống Chẩn Đoán Ung Thư Phổi Qua Ảnh X-ray

Hệ thống sử dụng Deep Learning (CNN) để phân tích ảnh X-ray phổi và dự đoán khả năng mắc ung thư phổi.

## Yêu Cầu Hệ Thống

- Python 3.8 trở lên
- GPU (không bắt buộc nhưng khuyến khích để tăng tốc độ huấn luyện)

## Cài Đặt

1. Clone repository này về máy:
```bash
git clone https://github.com/noqokhxnh/Lung_Cancer.git
cd <Lung_Cancer>
```

2. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

## Cấu Trúc Dữ Liệu

Để huấn luyện mô hình, bạn cần chuẩn bị dữ liệu theo cấu trúc sau:

```
Lung_xray/
└── chest_xray_lung/
    ├── train/
    │   ├── normal/         # Ảnh X-ray phổi bình thường
    │   └── cancer/         # Ảnh X-ray phổi có dấu hiệu ung thư
    ├── val/
    │   ├── normal/
    │   └── cancer/
    └── test/
        ├── normal/
        └── cancer/
```
Link chi tiết :
```
https://www.kaggle.com/datasets/quynhlecl/lung-cancer-x-ray
```
## Huấn Luyện Mô Hình

1. Đảm bảo dữ liệu đã được tổ chức đúng cấu trúc như trên
2. Chạy script huấn luyện:
```bash
python train.py
```

Quá trình huấn luyện sẽ:
- Tự động chia dữ liệu thành các tập train/validation/test
- Áp dụng data augmentation để tăng cường dữ liệu
- Lưu mô hình tốt nhất vào file `lung_xray_model.h5`
- Hiển thị đồ thị accuracy và loss

## Sử Dụng Giao Diện Đồ Họa

1. Khởi động ứng dụng:
```bash
python GUI.py
```

2. Các bước sử dụng:
   - Nhấn nút "Chọn ảnh X-ray" để tải ảnh cần chẩn đoán
   - Ảnh sẽ được hiển thị trên giao diện
   - Nhấn "Chẩn đoán" để xem kết quả
   - Kết quả sẽ hiển thị:
     - Xác suất mắc ung thư phổi
     - Đánh giá và khuyến nghị

## Thông Số Kỹ Thuật

- Kích thước ảnh đầu vào: 124x124 pixels
- Mô hình: CNN với nhiều lớp Conv2D và BatchNormalization
- Data augmentation: Xoay, zoom, lật ảnh để tăng cường dữ liệu
- Early stopping để tránh overfitting

## Lưu Ý

- Đây chỉ là công cụ hỗ trợ chẩn đoán ban đầu
- Kết quả cần được xác nhận bởi bác sĩ chuyên khoa
- Độ chính xác phụ thuộc vào chất lượng ảnh X-ray đầu vào
- Nên sử dụng ảnh X-ray có độ phân giải tốt để có kết quả chính xác nhất

## Xử Lý Sự Cố

1. Lỗi "Không thể tải mô hình":
   - Kiểm tra file `lung_xray_model.h5` đã tồn tại
   - Chạy lại `train.py` nếu cần

2. Lỗi "Không thể mở ảnh":
   - Kiểm tra định dạng ảnh (hỗ trợ: jpg, png, bmp)
   - Đảm bảo ảnh không bị hỏng

3. Lỗi GPU:
   - Cài đặt CUDA và cuDNN nếu muốn sử dụng GPU
   - Hoặc sử dụng CPU (chậm hơn nhưng vẫn hoạt động)

