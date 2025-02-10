import easyocr
import pickle

# Tải đối tượng reader
reader = easyocr.Reader(['en'],gpu=False);
# Lưu đối tượng reader vào tệp
with open('reader.pkl', 'wb') as f:
    pickle.dump(reader, f)