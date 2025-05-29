import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import os
from train import IMG_SIZE
class LungXrayApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Chẩn Đoán Ung Thư Phổi qua X-ray")
        
        # Lấy kích thước màn hình
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        
        # Thiết lập kích thước cửa sổ
        window_width = 800
        window_height = 600
        
        # Tính toán vị trí để cửa sổ xuất hiện giữa màn hình
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        
        # Đặt kích thước và vị trí cửa sổ
        self.root.geometry(f'{window_width}x{window_height}+{x}+{y}')
        self.root.configure(bg='white')
        
        # Tạo style
        style = ttk.Style()
        style.configure("Title.TLabel", font=("Helvetica", 16, "bold"), background="white")
        style.configure("Result.TLabel", font=("Helvetica", 12), background="white")
        
        # Main Frame
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(expand=True, fill="both", padx=20, pady=20)
        
        # Title
        title = ttk.Label(
            self.main_frame,
            text="CHẨN ĐOÁN UNG THƯ PHỔI QUA ẢNH X-RAY",
            style="Title.TLabel"
        )
        title.pack(pady=20)
        
        # Frame cho ảnh
        self.image_frame = ttk.Frame(self.main_frame)
        self.image_frame.pack(pady=20)
        
        # Label để hiển thị ảnh
        self.image_label = ttk.Label(self.image_frame)
        self.image_label.pack()
        
        # Frame cho các nút
        button_frame = ttk.Frame(self.main_frame)
        button_frame.pack(pady=20)
        
        # Nút chọn ảnh
        self.select_button = ttk.Button(
            button_frame,
            text="Chọn ảnh X-ray",
            command=self.select_image
        )
        self.select_button.pack(side=tk.LEFT, padx=10)
        
        # Nút dự đoán
        self.predict_button = ttk.Button(
            button_frame,
            text="Chẩn đoán",
            command=self.predict,
            state=tk.DISABLED
        )
        self.predict_button.pack(side=tk.LEFT, padx=10)
        
        # Frame cho kết quả
        self.result_frame = ttk.Frame(self.main_frame)
        self.result_frame.pack(pady=20)
        
        # Label hiển thị kết quả
        self.result_label = ttk.Label(
            self.result_frame,
            text="",
            style="Result.TLabel",
            wraplength=600
        )
        self.result_label.pack()
        
        # Khởi tạo các biến
        self.model = None
        self.current_image = None
        self.image_path = None
        
        # Load model
        try:
            self.model = tf.keras.models.load_model('lung_xray_model.h5')
            print("Đã tải mô hình thành công!")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể tải mô hình: {str(e)}")
    
    def select_image(self):
        # Mở hộp thoại chọn file
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh X-ray",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Lưu đường dẫn ảnh
                self.image_path = file_path
                
                # Đọc và hiển thị ảnh
                image = Image.open(file_path)
                
                # Resize ảnh để vừa với giao diện
                display_size = (400, 400)
                image.thumbnail(display_size, Image.Resampling.LANCZOS)
                
                # Chuyển đổi sang PhotoImage
                photo = ImageTk.PhotoImage(image)
                
                # Hiển thị ảnh
                self.image_label.configure(image=photo)
                self.image_label.image = photo  # Giữ reference
                
                # Kích hoạt nút dự đoán
                self.predict_button.configure(state=tk.NORMAL)
                
                # Xóa kết quả cũ
                self.result_label.configure(text="")
                
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể mở ảnh: {str(e)}")
    
    def predict(self):
        if self.image_path and self.model:
            try:
                # Đọc và tiền xử lý ảnh
                img = tf.keras.preprocessing.image.load_img(
                    self.image_path,
                    target_size=(IMG_SIZE, IMG_SIZE)
                )
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = img_array / 255.0
                
                # Dự đoán
                prediction = self.model.predict(img_array, verbose=0)[0][0]
                
                # Hiển thị kết quả
                result_text = f"Xác suất ung thư phổi: {prediction*100:.2f}%\n\n"
                if prediction >= 0.5:
                    result_text += "CẢNH BÁO: Có dấu hiệu bất thường!\n"
                    result_text += "Vui lòng đến gặp bác sĩ để kiểm tra chi tiết."
                else:
                    result_text += "Không phát hiện dấu hiệu bất thường.\n"
                    result_text += "Tiếp tục theo dõi sức khỏe định kỳ."
                
                self.result_label.configure(text=result_text)
                
            except Exception as e:
                messagebox.showerror("Lỗi", f"Lỗi khi dự đoán: {str(e)}")
        else:
            messagebox.showerror("Lỗi", "Vui lòng chọn ảnh và đảm bảo mô hình đã được tải.")

if __name__ == "__main__":
    root = tk.Tk()
    app = LungXrayApp(root)
    root.mainloop()
