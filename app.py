import pyaudio  
import wave  
import numpy as np  
import time
import requests  
import json
from dotenv import load_dotenv
import os  
from qwen_agent.agents import Assistant
from qwen_agent.utils.output_beautify import typewriter_print

# 加载已有的 .env 文件（如果存在）  
load_dotenv("./1.env")  

# 配置  
FORMAT = pyaudio.paInt16  # 音频格式  
CHANNELS = 1               # 单声道  
RATE = 44100               # 采样率  
CHUNK = 1024               # 每次读取的音频帧数  
THRESHOLD = 200            # 音量阈值  
RECORD_SECONDS = 20        # 最大录音时间  
SILENCE_DURATION = 1      # 静音持续时间（秒） 
sk_key=os.getenv('key') 

llm_cfg = {
    # Use the model service provided by DashScope:
    'model': 'Qwen/Qwen3-235B-A22B',
    'model_server': 'https://api.siliconflow.cn/v1',
    'api_key': sk_key,
    'generate_cfg': {
        'top_p': 0.8,
        'thought_in_content': False,
    }
}
system_instruction = '''你是小新，一个AI智能体。'''
with open("mcp_server_config.json", "r") as f:
    config = json.load(f)
    
tools = [config,'code_interpreter']
bot = Assistant(llm=llm_cfg,
                system_message=system_instruction,
                function_list=tools)  

def get_volume(data):  
    """计算音量"""  
    return np.frombuffer(data, dtype=np.int16).astype(np.float32).mean()  
  
def record_audio():  
    """录制音频并保存为 WAV 文件"""  
    audio = pyaudio.PyAudio()  
      
    # 开始流  
    stream = audio.open(format=FORMAT, channels=CHANNELS,  
                        rate=RATE, input=True,  
                        frames_per_buffer=CHUNK)  
      
    print("\n(请说话...)")  
  
    frames = []  
    recording = False  
    silence_start_time = None  # 用于记录静音开始时间  
    start_time = time.time()    # 记录开始时间  
  
    while True:  
        # 读取音频数据  
        data = stream.read(CHUNK)  
        volume = get_volume(data)  
          
        if volume > THRESHOLD and not recording:  
            print("(持续聆听中...)")  
            recording = True  
            start_time = time.time()  # 记录开始时间  
            silence_start_time = None  # 重置静音计时器  
          
        if recording:  
            frames.append(data)  
            #print(f"录音中... 音量: {volume}")  
            
            # 检查录音时间  
            if time.time() - start_time > RECORD_SECONDS:  
                print("(处理中，请稍后...)")  
                break  
  
            # 检查静音  
            if volume < THRESHOLD:  
                if silence_start_time is None:  
                    silence_start_time = time.time()  # 记录静音开始时间  
                elif time.time() - silence_start_time > SILENCE_DURATION:  
                    print("(处理中，请稍后...)")  
                    break  
            else:  
                silence_start_time = None  # 重置静音计时器  
  
    # 停止流  
    stream.stop_stream()  
    stream.close()  
    audio.terminate()  
  
    # 保存录音  
    if frames:  
        with wave.open("output.wav", 'wb') as wf:  
            wf.setnchannels(CHANNELS)  
            wf.setsampwidth(audio.get_sample_size(FORMAT))  
            wf.setframerate(RATE)  
            wf.writeframes(b''.join(frames))  
        #print("录音已保存为 output.wav")  
    else:  
        print("(没有发现声音！)")  


def transcribe_audio(file_path):  
    """将音频文件发送到 API 并返回转录文本"""  
    url = "https://api.siliconflow.cn/v1/audio/transcriptions"

    payload = {'model': 'FunAudioLLM/SenseVoiceSmall'}
    f=open('output.wav','rb')
    files=[
      ('file',('1.wav',f,'audio/wav'))
    ]
    headers = {
      'Authorization': f"Bearer {sk_key}"
    }
    try:  
        response = requests.request("POST", url, headers=headers, data=payload, files=files)
        response.raise_for_status()  # 检查请求是否成功  
  
        # 解析 JSON 响应  
        json_response = response.json()  
        return json_response.get('text', '').strip()  # 返回 text 字段的值，如果没有则返回空字符串  
  
    except requests.exceptions.RequestException as e:  
        print(f"请求失败: {e}")  
        return None  
    except Exception as e:  
        print(f"发生错误: {e}")  
        return None  
    finally:  
        f.close()  # 确保文件在结束时关闭  

def text2speech(text,display=False):
    # API 请求
    url = "https://api.siliconflow.cn/v1/audio/speech"
    
    payload = {
        "model": "FunAudioLLM/CosyVoice2-0.5B",
        "voice": "FunAudioLLM/CosyVoice2-0.5B:david",
        "input": text,
        "response_format": "pcm",  # 直接请求 PCM 格式
        "sample_rate": 44100
    }
    
    headers = {
        "Authorization": f"Bearer {sk_key}"
    }
    
    response = requests.post(url, json=payload, headers=headers)
    
    # 检查请求是否成功
    if response.status_code == 200:
        # 获取 PCM 数据
        pcm_data = response.content
    
        # 初始化 PyAudio
        p = pyaudio.PyAudio()
    
        # 打开音频流
        stream = p.open(format=pyaudio.paInt16,  # PCM 数据通常为 16 位
                        channels=1,             # 单声道
                        rate=44100,             # 采样率（根据 API 返回的采样率调整）
                        output=True)            # 输出流
    
        # 播放 PCM 数据
        if display:
            print(f"{text}")
        stream.write(pcm_data)
        #print("播放完成！")
    
        # 关闭流和 PyAudio
        stream.stop_stream()
        stream.close()
        p.terminate()
    else:
        print("❤️")

messages=[]
def generate():  
    global messages  
    i=0
    result_buffer = ""  
    result_all=""
    print("AI:")
    for response in bot.run(messages=messages):
        #print(response)
        if len(response)>0 and  "content" in response[-1]:
            result = response[-1]["content"]
            message = result[i:]
            i=len(result)
            
            if len(result_all)>0 or (message!="\n" and message!="\n\n"):
                result_buffer += message 
                result_all+=message
                print(message,end="",flush=True)
            #if len(result_all)==0:
            #    print(f".",end="",flush=True)
            if message.endswith(('!', '?','。','！','？','\n\n')) and len(result_buffer)>10: 
                #print(message)
                text2speech(result_buffer.strip().replace("*","").replace("-"," "))  
                result_buffer = ""  # 清空缓冲区  
                
    if result_buffer:  
        text2speech(result_buffer.strip().replace("*","").replace("#",""))  
  
    messages.append({"role": "assistant", "content": result_all.strip()})  
    return result_all        
    
        
if __name__ == "__main__": 
    text2speech("很高兴为您服务，我在听请讲！",display=True)
    while True:
        record_audio()
        question = transcribe_audio("output.wav")
        
        if question=="退出。":
            break
        if question and len(question)>0:
            print(f"你: {question}",flush=True)
        else:
            text2speech("我没听清，您可以大点声吗？")
            continue
        messages=messages+[{"role":"user","content":question}]
        answer = generate()
        