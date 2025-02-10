import re
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import easyocr
import pickle
import numpy as np


# Tải model EasyOCR
with open('reader.pkl', 'rb') as f:
    
    reader = pickle.load(f)

# # Hàm tính độ tương đồng giữa hai chuỗi
# def similarity_score(str1, str2):
#     # Code tính độ tương đồng giữa hai chuỗi (ví dụ: sử dụng Levenshtein distance)
#     pass

def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        # Xử lý ảnh và hiển thị kết quả
        process_and_display_image(file_path)

def process_and_display_image(file_path):
    img = cv2.imread(file_path)
    if img is not None:
        process_frame(img)
    else:
        label_text.config(text="Không thể đọc ảnh")


def load_video():
    file_path = filedialog.askopenfilename()
    if file_path:
      
        process_and_display_video(file_path)    
def process_and_display_video(file_path):
    cap = cv2.VideoCapture(file_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        process_frame(frame)
        cv2.imshow('License Plate Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def process_frame(frame):
    # Đọc ảnh sử dụng OpenCV
    img = frame
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17) #Noise reduction
    edged = cv2.Canny(bfilter, 30, 200) #Edge detection

    # Tìm contours
    # Tìm contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break
    
    if location is None:
        label_text.config(text="Không tìm thấy biển số xe")
        return

    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], -1, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)
    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    # Thêm phần đệm (padding) xung quanh vùng cắt
    padding = 10  # Kích thước phần đệm (pixels)
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(gray.shape[0], x2 + padding)
    y2 = min(gray.shape[1], y2 + padding)
    # Kiểm tra giới hạn tọa độ
    if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > gray.shape[0] or y2 > gray.shape[1]:
        return img
    cropped_image = gray[x1:x2+1, y1:y2+1]
    # Kiểm tra nếu vùng cắt rỗng hoặc không hợp lệ
    if cropped_image.size == 0 or cropped_image is None:
        return img
    # Chuyển đổi ảnh biển số sang định dạng PIL để hiển thị trên Tkinter
    img_pil = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    img_tk = ImageTk.PhotoImage(img_pil)

    # Hiển thị ảnh biển số
    label_image.config(image=img_tk)
    label_image.image = img_tk

    # Sử dụng EasyOCR để đọc ký tự trên biển số
    corrected_text = "Không tìm thấy biển số xe"
    ocr_result = reader.readtext(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB), detail=1)
    if ocr_result:
        char_to_num_dict = {'A': '4', 'B': '3','@':'3', 'D': '0', 'G': '6', 'T': '1', 'S': '5', 'Z': '2','J':'3'}
        def correct_first_nums(first_nums):
            corrected = ''
            for char in first_nums:
                corrected += char_to_num_dict.get(char, char)  # Sử dụng ký tự tương ứng hoặc giữ nguyên nếu không có trong từ điển
            return corrected
        def correct_chars(chars):
            corrected = ''
            for char in chars:
                corrected += char_to_num_dict.get(char, char)  # Sử dụng ký tự tương ứng hoặc giữ nguyên nếu không có trong từ điển
            return corrected
        # Split the OCR results
        full_text = ''.join([result[1] for result in ocr_result])
        first_nums= full_text[:2]
        # Bỏ qua dấu chấm hoặc gạch ngang, đảm bảo lấy đúng 5 ký tự cuối
        last_part = re.sub(r'[.\-]', '', full_text[-6:])
        last_nums = last_part[-5:]
        chars=full_text[2:len(full_text)-6]
        # Sửa lỗi cho các ký tự đầu nếu cần
        first_nums = correct_first_nums(first_nums)
        last_nums = correct_chars(last_nums)
        # Correct the middle part
        final_chars = ''
        for char in chars:
            if char == '4':
                final_chars += 'A'
            elif char == '6':
                final_chars += 'G'
            elif char == '8':
                final_chars += 'B'
            elif char == 'DI':
                final_chars += 'A'
            elif char == '1':
                final_chars += 'T'
            elif char == '0':
                final_chars += 'D'
            elif char == '5':
                final_chars += 'S'
            elif char == '2':
                final_chars += 'Z'
            else:
                final_chars += char
        corrected_text = first_nums + final_chars + last_nums

        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, text=corrected_text, org=(location[0][0][0], location[1][0][1] + 60), fontFace=font, fontScale=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.rectangle(img, tuple(location[0][0]), tuple(location[2][0]), (0, 255, 0), 3)
       # Tính toán độ chính xác trung bình
        confidences = [result[2] for result in ocr_result]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        # Hiển thị kết quả và độ chính xác
        license_text = corrected_text + f"\nĐộ chính xác: {avg_confidence:.2f}"
        label_text.config(text=license_text)

# Thiết lập giao diện người dùng
root = tk.Tk()
root.title("License Plate Recognition")

# Thiết lập các widget
btn_load_image = tk.Button(root, text="Load Image", command=load_image)
btn_load_image.pack()

btn_load_video = tk.Button(root, text="Load Video", command=load_video)
btn_load_video.pack()

label_image = tk.Label(root)
label_image.pack()

label_text = tk.Label(root, text="Biển số xe sẽ hiển thị ở đây", font=('Helvetica', 16))
label_text.pack()
root.mainloop()
