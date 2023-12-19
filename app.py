import base64
import json
import time
from glob import glob
import wave
import random
from PIL import Image
import streamlit as st
import speech_recognition as sr
import requests
import torch
import google.generativeai as gmn
from transformers import pipeline, AutoModelForSequenceClassification, BertJapaneseTokenizer

# 音声録音を行う関数
def record():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)
    output = r.recognize_google(audio, language='ja-JP')
    return output

host = "127.0.0.1"
port = 50021

# 音声ファイルを再生する関数
def replay_audio(audio_path):
    file_ = open(audio_path, "rb")
    contents = file_.read()
    file_.close()
    audio_placeholder = st.empty()
    audio_str = "data:audio/ogg;base64,%s" % (base64.b64encode(contents).decode())
    audio_html = f"""
        <audio autoplay=True>
        <source src="{audio_str}" type="audio/ogg" autoplay=True>
        Your browser does not support the audio element.
        </audio>
    """
    audio_placeholder.empty()
    time.sleep(0.5)
    audio_placeholder.markdown(audio_html, unsafe_allow_html=True)

# 音声クエリを行う関数
def audio_query(text, speaker, max_retry):
    query_payload = {"text": text, "speaker": speaker}
    for query_i in range(max_retry):
        r = requests.post(f"http://{host}:{port}/audio_query", params=query_payload, timeout=(10.0, 300.0))
        if r.status_code == 200:
            query_data = r.json()
            break
        time.sleep(0.1)
    else:
        raise ConnectionError("リトライ回数が上限に到達しました。 audio_query : ", "/", text[:30], r.text)
    return query_data

# 音声合成を行う関数
def synthesis(speaker, query_data, max_retry):
    synth_payload = {"speaker": speaker}
    for synth_i in range(max_retry):
        r = requests.post(f"http://{host}:{port}/synthesis", params=synth_payload, data=json.dumps(query_data), timeout=(10.0, 300.0))
        if r.status_code == 200:
            return r.content
        time.sleep(0.1)
    else:
        raise ConnectionError("音声エラー：リトライ回数が上限に到達しました。 synthesis : ", r)

# テキストを音声に変換する関数
def text_to_speech(texts, speaker, output_file, max_retry=20):
    if not texts:
        texts = "ちょっと、通信状態悪いかも？"
    
    query_data = audio_query(texts, speaker, max_retry)
    voice_data = synthesis(speaker, query_data, max_retry)
    
    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(voice_data)

# モデルを初期化する関数
def initialize_model(chat_model):
    if chat_model is None:
        model = gmn.GenerativeModel("gemini-pro")
        chat_model = model.start_chat(history=[])
    return chat_model

# 感情分析を行う関数
def output_emotion(text, model, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        model = AutoModelForSequenceClassification.from_pretrained("koheiduck/bert-japanese-finetuned-sentiment")
    if tokenizer is None:    
        tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
    pl = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device)

    emotion = pl(text)[0]["label"]
    return emotion

# 感情に応じた画像パスを取得する関数
def output_face_path(emotion):
    if emotion == "POSITIVE":
        path_list = glob("face/positive/talk/*.png")
    elif emotion == "NEGATIVE":
        path_list = glob("face/negative/talk/*.png")
    else:
        path_list = glob("face/neutral/talk/*.png")

    return random.choice(path_list)

# 感情に応じた話し手の番号を取得する関数
def output_speaker(emotion):
    if emotion == "POSITIVE":
        speaker = 48
    elif emotion == "NEGATIVE":
        speaker = 49
    else:
        speaker = 47

    return speaker

# 画像を重ね合わせる関数
def image_overlay(back_path, face_path):
    back = Image.open(back_path)
    face = Image.open(face_path)

    back_w, back_h = back.size
    back = back.resize((389, int(389/(back_w) * back_h)))
    face = face.resize((400, 550))
    back.paste(face, (95, 228), face)
    
    return back

#modelに渡す内容を出力する関数
def output_contents(prompt, uploaded_image):
    contents = [prompt]
    for image in uploaded_image:
        image = Image.open(image)
        contents.append(image)

    return contents
#画像と返答を表示する関数
def display_Images_and_response(prompt, uploaded_image):
    if "pro_vision_model" not in st.session_state:
        st.session_state.pro_vision_model = gmn.GenerativeModel('gemini-pro-vision')
    model = st.session_state.pro_vision_model

    num_images = len(uploaded_image)
    columns = st.columns(num_images)
    for i, image in enumerate(uploaded_image):
        columns[i].image(image, use_column_width=True)

    contents = output_contents(prompt, uploaded_image)
    
    response = model.generate_content(
            contents=contents).text
    
    st.write(response)

#説明を記載する関数
def displayExplanation():
    chat_explaination = """Chat:
                    \n1. Press the 'Chat' button.
                    \n2. Click on the three dots in the upper-right corner, and from the 'Settings' menu, turn on 'Wide mode'. 
                    \n3. After pressing the 'Exec' button, please speak into your audio input device,such as a microphone.
                    \n4. After a few seconds, the avatar's expression will change, and a response will be returned."""
    
    vision_explanation = """Vision:
                    \n1. Press the 'Vision' button.
                    \n2. Enter text related to the image into the 'Prompt' text box.
                    \n3. Drag and drop the image or click the 'Browse file' button to upload the image.
                    \n4. After pressing the 'Exec' button, a response will be returned."""
    
    api_explanation = """API:
                    \nPlease create or obtain an API key from the following
                    \nURL: https://makersuite.google.com/app/apikey"""
    
    credit = "VOICEVOX:ナースロボ＿タイプＴ"

    character = """立ち絵:【MYCOEIROINK】ナースロボ_タイプT + 立ち絵PSD
              \nhttps://booth.pm/ja/items/3876977"""

    if st.session_state.chat == True and st.session_state.vision == False:
        st.sidebar.info(chat_explaination)

    elif st.session_state.chat == False and st.session_state.vision == True:
        st.sidebar.info(vision_explanation)

    else:
        st.sidebar.info(chat_explaination)
        st.sidebar.info(vision_explanation)

    st.sidebar.info(api_explanation)
    st.sidebar.info(credit)
    st.sidebar.info(character)
        
# Streamlitアプリのメイン関数
def app():
    st.sidebar.title("Gemini: Chat & VQA app.")

    if "chat" not in st.session_state:
        st.session_state.chat = None

    if "vision" not in st.session_state:
        st.session_state.vision = None

    side_col1, side_col2 = st.sidebar.columns(2)

    if side_col1.button("Chat"):
        st.session_state.chat = True
        st.session_state.vision = False

    if side_col2.button("Vision"):
        st.session_state.chat = False
        st.session_state.vision = True

    if st.session_state.chat:
        back_path = "back/back.jpg"
        input1 = st.sidebar.text_input("API_KEY", key="textbox1")
        input2 = st.sidebar.text_input("Constraint ", key="textbox2")
        gmn.configure(api_key=input1)
        audio_path = "output.wav"
        user_text = None
        is_talk = False

    # セッションステートの初期化
        if "chat_model" not in st.session_state:
            st.session_state.chat_model = None

        if "emotional_model" not in st.session_state:
            st.session_state.emotional_model = None

        if "emotional_tokenizer" not in st.session_state:
            st.session_state.emotional_tokenizer = None

        col1, col2 = st.columns(2)

        if st.sidebar.button("exec"):
            operation = """命令:
            あなたはとあるキャラクターとして、以下の制約条件を厳密に守って、会話に対する人間らしい応答を生成してください。\n
            """
            constraint = "制約条件:" + input2 + "\n"
            operation += constraint
            
            user_text = record()

            chat_model = st.session_state.chat_model
            emotional_model = st.session_state.emotional_model
            emotional_tokenizer = st.session_state.emotional_tokenizer

            if user_text is not None:
                text = operation + constraint + user_text
                chat_model = initialize_model(chat_model)
                st.session_state.chat_model = chat_model
                output_texts = chat_model.send_message(text).text
                with col2:
                    st.write("Your response:")
                    st.write(">" + user_text)
                emotion = output_emotion(output_texts, emotional_model, emotional_tokenizer)
                speaker = output_speaker(emotion)
                text_to_speech(output_texts, speaker, audio_path)
                face_path = output_face_path(emotion)
                is_talk = True

        if is_talk:
            replay_audio(audio_path)
            image = image_overlay(back_path, face_path)
            with col1:
                st.image(image)
            with col2:
                st.write("My response:")
                st.write(">" + output_texts.replace("\n", ""))

        else:
            image = image_overlay(back_path, face_path="face/neutral/wait/1.png")
            with col1:
                st.image(image)

    elif st.session_state.vision:
        input1 = st.sidebar.text_input("API_KEY", key="textbox1")
        gmn.configure(api_key=input1)
        text_box = st.sidebar.text_input("Prompt")
        uploaded_image = st.sidebar.file_uploader("Image file", accept_multiple_files=True)
        if st.sidebar.button("exec"): 
            prompt = text_box
            display_Images_and_response(prompt, uploaded_image)
    displayExplanation()

if __name__ == "__main__":
    app()
