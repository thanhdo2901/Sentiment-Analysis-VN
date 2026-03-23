{
  "cells": [
    {
      "cell_type": "code",
      "execution_count": 20,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "GZfR4Qiy09D4",
        "outputId": "024663a7-6158-41d8-ad45-0cda3ffd148e"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Writing app.py\n"
          ]
        }
      ],
      "source": [
        "%%writefile app.py\n",
        "import streamlit as st\n",
        "import joblib\n",
        "import re\n",
        "from underthesea import word_tokenize\n",
        "\n",
        "# 1. Load mô hình và bộ biến đổi đã lưu ở bước trước\n",
        "model = joblib.load('best_sentiment_model.pkl')\n",
        "tfidf = joblib.load('tfidf_vectorizer.pkl')\n",
        "\n",
        "# 2. Hàm tiền xử lý (phải giống hệt lúc huấn luyện)\n",
        "def preprocess_text(text):\n",
        "    text = str(text).lower()\n",
        "    text = re.sub(r'[^\\w\\s]', '', text)\n",
        "    text = re.sub(r'\\d+', '', text)\n",
        "    text = word_tokenize(text, format=\"text\")\n",
        "    return text\n",
        "\n",
        "# 3. Giao diện Web\n",
        "st.title(\"😊 Hệ thống Phân loại Cảm xúc Tiếng Việt\")\n",
        "st.write(\"Đồ án tốt nghiệp - SV thực hiện: Nguyễn Thành Đô\")\n",
        "\n",
        "input_text = st.text_input(\"Nhập câu văn cần phân tích:\", \"Giảng viên nhiệt tình, bài giảng hay.\")\n",
        "\n",
        "if st.button(\"Phân tích ngay\"):\n",
        "    # Tiền xử lý câu nhập vào\n",
        "    clean_input = preprocess_text(input_text)\n",
        "    # Biến đổi TF-IDF\n",
        "    vectorized_input = tfidf.transform([clean_input])\n",
        "    # Dự đoán\n",
        "    prediction = model.predict(vectorized_input)[0]\n",
        "\n",
        "    # Hiển thị kết quả kèm icon\n",
        "    if prediction == 'positive' or prediction == 2:\n",
        "        st.success(f\"Kết quả: TÍCH CỰC 😊\")\n",
        "    elif prediction == 'negative' or prediction == 0:\n",
        "        st.error(f\"Kết quả: TIÊU CỰC 😡\")\n",
        "    else:\n",
        "        st.warning(f\"Kết quả: TRUNG LẬP 😐\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 23,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "CBfjYezl1AXV",
        "outputId": "f8b6d6a9-abcb-4d03-b514-7d4ac411ad13"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "136.111.248.9\n",
            "\u001b[1G\u001b[0K⠙\n",
            "Collecting usage statistics. To deactivate, set browser.gatherUsageStats to false.\n",
            "\u001b[0m\n",
            "\u001b[1G\u001b[0K⠹\u001b[1G\u001b[0K⠸\u001b[1G\u001b[0K⠼\u001b[1G\u001b[0K⠴\u001b[1G\u001b[0K⠦\u001b[1G\u001b[0K⠧\u001b[1G\u001b[0K⠇\u001b[1G\u001b[0K⠏\u001b[1G\u001b[0K⠋\u001b[1G\u001b[0K⠙\u001b[1G\u001b[0K⠹\u001b[1G\u001b[0K⠸\u001b[1G\u001b[0K⠼\u001b[1G\u001b[0K⠴\u001b[1G\u001b[0Kyour url is: https://thick-llamas-tease.loca.lt\n",
            "\u001b[0m\n",
            "\u001b[34m\u001b[1m  You can now view your Streamlit app in your browser.\u001b[0m\n",
            "\u001b[0m\n",
            "\u001b[34m  Local URL: \u001b[0m\u001b[1mhttp://localhost:8501\u001b[0m\n",
            "\u001b[34m  Network URL: \u001b[0m\u001b[1mhttp://172.28.0.12:8501\u001b[0m\n",
            "\u001b[34m  External URL: \u001b[0m\u001b[1mhttp://136.111.248.9:8501\u001b[0m\n",
            "\u001b[0m\n",
            "\u001b[34m  Stopping...\u001b[0m\n",
            "^C\n"
          ]
        }
      ],
      "source": [
        "# Lấy địa chỉ IP công cộng của Colab (bạn sẽ cần nó để dán vào trang web)\n",
        "!wget -q -O - ipv4.icanhazip.com\n",
        "\n",
        "# Chạy Streamlit ở chế độ nền và mở tunnel\n",
        "!streamlit run app.py & npx localtunnel --port 8501"
      ]
    }
  ],
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}
