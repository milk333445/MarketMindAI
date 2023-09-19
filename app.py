from flask import Flask, request, jsonify
from flask import Flask, request, render_template
from demo import CauseAnalysisWebAPI, draw_diagram, get_event_and_related_event, get_my_agent, point_of_view_analysis, cut_result, get_related_history_events, event_time_inquiry, history_event
#api和格式整理
from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from serpapi import GoogleSearch
from langchain.document_loaders import WebBaseLoader
import re
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
import numpy as np
from sklearn.cluster import KMeans
import json
from kor import create_extraction_chain, Object, Text ,Number
from langchain import LLMChain
import os
from config import OPEN_API_KEY, serpapi
#初始化
os.environ['OPENAI_API_KEY'] = OPEN_API_KEY
os.environ['SERPAPI_API_KEY'] = serpapi

app = Flask(__name__)


def read_txt_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    
def delete_images():
    current_dir = os.getcwd()
    image_names = ['accounting_and_stock.png', 'bollinger.png', 'financials.png', 'MACD.png', 'PRICE.png', 'RSI.png', 'stock_valuation1.png', 'stock_valuation2.png']

    for image_name in image_names:
        save_path = os.path.join(current_dir, 'static', image_name)
        if os.path.exists(save_path):
            os.remove(save_path)
            print(f"Deleted {image_name}")
        else:
            print(f"File {image_name} not found")

@app.route('/')
def index():
    return render_template('index.html')


#設定全局變量(事件時間)
event_data = {'event': None}
casualresult_data = {'casual_result': None}
history_keyword_data = {'history_keyword': None}

@app.route('/get_event', methods=['POST'])
def get_event():
    user_input = request.json['user_input_event']
    user_input_company_name = request.json['user_input_company']
    user_input_hint = request.json['user_input_hint']
    user_input_history_keyword = request.json['user_input_history_keyword']
    
    print(user_input)
    print(user_input_company_name)
    print(user_input_hint)
    print(user_input_history_keyword)
    try:
        event = event_time_inquiry(user_input)
    except Exception as e:
        print(f"可能該事件無確切時間:", e)
        event = [user_input, "無確切時間"]
    print(event)
    event_data['event'] = event
    return jsonify({'eventname': event[0],
                    'eventtime': event[1]})


@app.route('/get_history_keyword', methods=['POST'])
def get_history_keyword():
    user_input = request.json['user_input_event']
    user_input_company_name = request.json['user_input_company']
    user_input_hint = request.json['user_input_hint']
    user_input_history_keyword = request.json['user_input_history_keyword']

    
    key_word = history_event(user_input)
    key_word = key_word['keyword_extract'][0]['keyword']
    return jsonify({'history_keyword': key_word})
    
 

@app.route('/process', methods=['POST'])
def process():
    #獲得全局變量(事件時間)
    global event_data, casualresult_data
    event = event_data['event']
    print('event:', event)
    
    delete_images()
    user_input = request.json['user_input_event']
    user_input_company_name = request.json['user_input_company']
    user_input_hint = request.json['user_input_hint']
    user_input_history_keyword = request.json['user_input_history_keyword']
    print(user_input)
    print(user_input_company_name)
    print(user_input_hint)
    #歷史事件
    
    while True:
        key_word = user_input_history_keyword
        related_history_events = get_related_history_events(key_word)
        if related_history_events is not None and any(related_history_events.values()):
            print("取得歷史事件資料成功！")
            break
        else:
            print("取得歷史事件資料失敗！ 正在重新取得...")
    
    
        
    #因果分析
    processed_result , result, wiki_search_summary, cause_and_effect_input, news_reference = CauseAnalysisWebAPI(user_input, related_history_events, event)
    news_reference = json.dumps(news_reference)
    casualresult_data['casual_result'] = processed_result
    #切字
    try:
        background, cause, effect, pointer = cut_result(processed_result)
    except Exception as e:
        print(f"Error cutting result: {e}")
        background, cause, effect, pointer = None, None, None, None
    #觀點分析
    if user_input_hint.strip() != '':
        point_of_view_analysis_result = point_of_view_analysis(result, processed_result, event, user_input_hint)
    else:
        point_of_view_analysis_result = None
    #因果流程圖
    try:
        draw_diagram(f'事件名稱:{event[0]}\n {processed_result}')
    except Exception as e:
        print(f"Error drawing diagram: {e}")
        draw_diagram(f'事件名稱:{event[0]}\n {processed_result}')
    #投資建議
    if user_input_company_name.strip() != '':
        my_agent = get_my_agent(processed_result, event, user_input_company_name)
        max_retries = 3
        retries = 0
        while retries < max_retries:
            try:   
                invest_recommendation = my_agent.run(f'請給我{user_input_company_name}(股票代碼或公司名稱)的投資建議')
                break
            except Exception as e:
                print("出現錯誤")
                retries += 1
                if retries < max_retries:
                    print(f"嘗試重新執行... (第{retries}次)...")
                else:
                    print(f"已達到最大重試次數，無法繼續嘗試")
                    invest_recommendation = None

        file_path = os.path.join(os.getcwd(),'investment.txt')
        investment_result = read_txt_file(file_path)  
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write('')
    else:
        investment_result = None
        
    if user_input_company_name.strip() != '':
        #股價變化分析
        file_path_stock_price = os.path.join(os.getcwd(),'price_plus_casual_result.txt')
        price_plus_casual_result = read_txt_file(file_path_stock_price)
        with open(file_path_stock_price, 'w', encoding='utf-8') as file:
            file.write('')
        file_path_reference_data = os.path.join(os.getcwd(),'reference.txt')
        reference_data = read_txt_file(file_path_reference_data)
        try:
            reference_data = json.loads(reference_data)
            reference_data = json.dumps(reference_data)
        except Exception as e:
            print(f"Error reading reference data: {e}")
            reference_data = None
        with open (file_path_reference_data, 'w', encoding='utf-8') as file:
            file.write('')
    else:
        price_plus_casual_result = None
        reference_data = None
        
    #處理參考資料
        

    related_history_events = json.dumps(related_history_events)
    return jsonify({'result': processed_result, 
                    'background': background,
                    'cause': cause,
                    'effect': effect,
                    'pointer': pointer,
                    'investment_result': investment_result,
                    'related_history_events': related_history_events,
                    'price_plus_casual_result': price_plus_casual_result,
                    'point_of_view_analysis':point_of_view_analysis_result,
                    'reference_data': reference_data,
                    'news' : result,
                    'news_reference': news_reference})


@app.route('/regenerate', methods=['POST'])
def regenerate():
    global event_data, casualresult_data
    event = event_data['event']
    processed_result = casualresult_data['casual_result']
    print('event:', event)
    print('processed_result:', processed_result)
    user_input = request.json['user_input_event']
    user_input_company_name = request.json['user_input_company']
    user_input_hint = request.json['user_input_hint']
    print(user_input)
    print(user_input_company_name)
    print(user_input_hint)
    
    try:
        draw_diagram(f'事件名稱:{event[0]}\n {processed_result}')
        confirm = '重新產生流程圖成功'
    except Exception as e:
        print(f"Error drawing diagram: {e}")
        draw_diagram(f'事件名稱:{event[0]}\n {processed_result}') 
    return jsonify({'confirm': confirm})

@app.route('/news_reference.html')
def news_reference():
    return render_template('news_reference.html')


if __name__ == '__main__':
    app.run()
