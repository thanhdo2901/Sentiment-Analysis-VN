import streamlit as st
import joblib
import re
from underthesea import word_tokenize

# Đảm bảo 2 tên file này khớp Y HỆT tên file Đô đã upload lên GitHub
model = joblib.load('best_sentiment_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# Hàm tiền xử lý (Mục tiêu cụ thể số 3 trong đề cương)
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = word_tokenize(text, format="text")
    return text

st.title("😊 Hệ thống Phân loại Cảm xúc Tiếng Việt")
st.subheader("Đồ án tốt nghiệp - SV: Nguyễn Thành Đô")

input_text = st.text_input("Nhập câu văn cần phân tích:", "Giảng viên nhiệt tình quá!")

if st.button("Phân tích ngay"):
    clean_input = preprocess_text(input_text)
    vectorized_input = tfidf.transform([clean_input])
    prediction = model.predict(vectorized_input)[0]
    
    # Hiển thị kết quả (0: Tiêu cực, 1: Trung lập, 2: Tích cực)
    if prediction == 2 or prediction == 'positive':
        st.success("Kết quả: TÍCH CỰC 😊")
    elif prediction == 0 or prediction == 'negative':
        st.error("Kết quả: TIÊU CỰC 😡")
    else:
        st.warning("Kết quả: TRUNG LẬP 😐")
