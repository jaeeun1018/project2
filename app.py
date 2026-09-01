import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd

st.set_page_config(
    page_title="AI 생성 이미지 판별",
    page_icon="🤖",
    layout="wide"
)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model.keras")

model = load_model()



tab1, tab2 = st.tabs([
    "🎮 AI 위조도 시뮬레이터",
    "📊 데이터 인사이트 대시보드"
])


#
with tab1:

    st.title("🤖 AI 생성 이미지 판별")

    st.write(
        "얼굴 이미지를 업로드하면 CNN 모델이 "
        "Real / Fake 여부와 예측 확률을 판별합니다."
    )

    st.divider()

    file = st.file_uploader(
        "이미지를 선택해주세요.",
        type=["jpg", "jpeg", "png"]
    )

    if file is not None:

        image = Image.open(file).convert("RGB")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(
                image,
                caption="업로드 이미지",
                width=300
            )

        with col2:

            st.subheader("AI 이미지 판별")

            if st.button(
                "🔍 판별하기",
                use_container_width=True
            ):

                # 이미지 전처리
                img = image.resize((128, 128))

                img = np.array(
                    img
                ).astype("float32")

                # 0~1 정규화
                img = img / 255.0

                # 배치 차원 추가
                img = np.expand_dims(
                    img,
                    axis=0
                )

                # 예측
                prediction = model.predict(
                    img,
                    verbose=0
                )

                score = float(
                    prediction[0][0]
                )

                # Fake = 0 / Real = 1
                if score >= 0.56:

                    real_prob = score * 100
                    fake_prob = (1 - score) * 100

                    st.success(
                        "✅ 판별 결과 : Real 이미지"
                    )

                    st.metric(
                        "Real 확률",
                        f"{real_prob:.2f}%"
                    )

                else:

                    fake_prob = (1 - score) * 100
                    real_prob = score * 100

                    st.error(
                        "⚠️ 판별 결과 : Fake 이미지"
                    )

                    st.metric(
                        "Fake 확률",
                        f"{fake_prob:.2f}%"
                    )

                # 두 클래스 확률
                st.subheader("예측 확률")

                probability_df = pd.DataFrame({
                    "분류": ["Real", "Fake"],
                    "확률": [real_prob, fake_prob]
                })

                st.bar_chart(
                    probability_df,
                    x="분류",
                    y="확률"
                )



with tab2:

    st.title("📊 데이터 인사이트 대시보드")

    st.write(
        "CNN 모델의 학습 결과와 성능 평가 지표를 "
        "확인할 수 있습니다."
    )

    st.divider()


    st.subheader("📌 모델 주요 성능")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Train Accuracy",
            "83.63%"
        )

    with col2:
        st.metric(
            "Validation Accuracy",
            "82.37%"
        )

    with col3:
        st.metric(
            "ROC-AUC",
            "0.9293"
        )

    with col4:
        st.metric(
            "판별 Threshold",
            "0.56"
        )

    st.divider()

    st.subheader("🗂️ 데이터셋 구성")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "전체 이미지",
            "161,972장"
        )

    with col2:
        st.metric(
            "Real",
            "81,000장"
        )

    with col3:
        st.metric(
            "Fake",
            "80,972장"
        )

    st.caption(
        "원본 이미지를 모델 학습 시 "
        "128 × 128 크기로 리사이즈하여 사용했습니다."
    )

    st.divider()

 
    st.subheader("🧩 Confusion Matrix")

    confusion_matrix = pd.DataFrame(
        [
            [13712, 2563],
            [2438, 13681]
        ],
        columns=[
            "예측 Fake",
            "예측 Real"
        ],
        index=[
            "실제 Fake",
            "실제 Real"
        ]
    )

    st.dataframe(
        confusion_matrix,
        use_container_width=True
    )

    st.info(
        "Fake와 Real 모두 약 1만 3천 장 이상을 "
        "정확하게 분류했으며, 특정 클래스에 "
        "크게 편향되지 않은 결과를 확인했습니다."
    )

    st.divider()

    st.subheader("🧠 CNN 모델 구성")

    st.write("""
    **Conv2D(32) → MaxPooling2D →  
    Conv2D(64) → MaxPooling2D →  
    Conv2D(128) → MaxPooling2D →  
    GlobalAveragePooling2D → Dense(128) →  
    Dropout(0.5) → Sigmoid**
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**Optimizer**")
        st.write("Adam")

    with col2:
        st.write("**Loss Function**")
        st.write("Binary Crossentropy")

    with col3:
        st.write("**Learning Rate**")
        st.write("0.0001")

    st.divider()

    st.caption(
        "※ 현재 모델은 Real/Fake 얼굴 이미지 데이터셋을 "
        "기반으로 학습된 이진 분류 모델입니다."
    )