import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 제목
st.title("AI 생성 이미지 판별")
st.write("이미지를 업로드하면 Real / Fake 여부를 판별합니다.")

# 모델 불러오기
model = tf.keras.models.load_model("best_model.keras")

# 이미지 업로드
file = st.file_uploader(
    "이미지를 선택해주세요.",
    type=["jpg", "jpeg", "png"]
)

if file is not None:

    # 이미지 열기
    image = Image.open(file).convert("RGB")

    # 이미지 출력
    st.image(image, caption="업로드 이미지", width=300)

    if st.button("판별하기"):
    
        img = image.resize((128, 128))
        # numpy 배열로 변환
        img = np.array(img).astype("float32")

        # 학습 데이터와 동일하게 0~1 정규화
        img = img / 255.0

        # 배치 차원 추가
        img = np.expand_dims(img, axis=0)

        # 예측
        prediction = model.predict(img)

        score = float(prediction[0][0])
        if score >= 0.56:
            real_prob = score * 100

            st.success("판별 결과 : Real 이미지")
            st.write("Real 확률 :", round(real_prob, 2), "%")

        else:
            fake_prob = (1 - score) * 100

            st.error("판별 결과 : Fake 이미지")
            st.write("Fake 확률 :", round(fake_prob, 2), "%")