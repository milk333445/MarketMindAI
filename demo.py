#api和格式整理

from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from serpapi import GoogleSearch
from langchain.document_loaders import WebBaseLoader #serp api爬下來的內容只有網址，使用這個功能解析網址內的內容
import re #正則表達式
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings #將資料轉成向量
import pinecone
from langchain.chat_models import ChatOpenAI 
import numpy as np
from sklearn.cluster import KMeans
from kor import create_extraction_chain, Object, Text  #字詞擷取結構化抽取工具
from langchain import PromptTemplate, LLMChain
import os #連接其他資料夾
from config import OPEN_API_KEY, serpapi, pinecone_api, pinecone_env, pinecone_index_name
import ast #把string轉成list
from langchain.agents import initialize_agent 
from langchain.agents import AgentType
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from langchain.agents import load_tools
import subprocess
from pathlib import Path
import json
import yfinance as yf
from yahooquery import Ticker
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from typing import Type
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from typing import Any
import math, random, csv, getpass
from math import gamma
from tabnanny import verbose #可以把結果在終端顯示出來的顯示工具
# log and save
import json, logging, pickle, sys, shutil, copy
from argparse import ArgumentParser, Namespace
from copy import copy
# %matplotlib inline
from PIL import Image
import torch
import shutil
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.agents import initialize_agent
from langchain.document_loaders import UnstructuredFileLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import UnstructuredExcelLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
import nltk
nltk.download('punkt')
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List, Union
import zipfile
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import BaseChatPromptTemplate, ChatPromptTemplate
from langchain.schema import AgentAction, AgentFinish, HumanMessage, SystemMessage
from text2vec import SentenceModel
import matplotlib
os.environ['SERPAPI_API_KEY'] = serpapi
#初始化
os.environ['OPENAI_API_KEY'] = OPEN_API_KEY

def increase_docs_diversity(docs, num):
    embeddings = OpenAIEmbeddings() #呼叫建立向量資料庫
    #做文本分群
    vectors = embeddings.embed_documents([x.page_content for x in docs]) #把文本轉為向量
    #分群
    num_clusters = num
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10).fit(vectors)
    #找到最近的embedding
    closest_indices = []

    for i in range(num_clusters):
        distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)
    
        close_index = np.argmin(distances)
        closest_indices.append(close_index)
    
    selected_indices = sorted(closest_indices)
    selected_docs = [docs[doc] for doc in selected_indices]
    return selected_docs



def extract_text(document):
        content = document.page_content.strip()
        content = re.sub(r"[\n\t]+", "", content)
        text = re.sub(r"[^\w\s.]", "", content)
        return text

def remove_duplicate_document(documents):
        unique_documents = []
        for doc in documents: #doc為從向量資料庫中抽取出來的文件資料集
            if doc not in unique_documents:
                unique_documents.append(doc)
        return unique_documents
def remove_unnessary_word(split_doce): #處理爬文下來的東西，將斜線跟其他怪符號刪掉
        for i in range(len(split_doce)):
            text = split_doce[i].page_content
            text = re.sub(r"[^\w\s\n]+", "", text)
            text = text.replace('\n', '')
            text = text.replace('\t', '')
            split_doce[i].page_content = text
        return split_doce
def remove_short_documents(split_doce, min_word_count): #刪掉太短的參考文本
        filtered_documents = []
        for i in range(len(split_doce)):
            if len(split_doce[i].page_content) > min_word_count:
                filtered_documents.append(split_doce[i])
        return filtered_documents


#處理維基百科需要資料
#轉字典
def parse_date(date_str):
    # 以"年"和"月"為分隔符號拆分日期字符串
    #parts = date_str.replace('年', '-').replace('月', '-').split('至')
    
    # 第一部分是起始年分
    start_year = date_str[:4]

    
    return start_year

def extract_info(data_str):
    name_list = []
    date_list = []
    description_list = []
    causal_analysis_list = []
    economic_situation_list = []

    lines = data_str.split('\n')

    for line in lines:
        if "類似事件名稱:" in line or "類似事件名稱：" in line:
            name = line.split('：', 1)[1] if '：' in line else line.split(':', 1)[1]
            name_list.append(name.strip())
        elif "類似事件起始結束日期:" in line or "類似事件起始結束日期：" in line:
            date = line.split('：', 1)[1] if '：' in line else line.split(':', 1)[1]
            date_list.append(date.strip())
        elif "類似事件描述:" in line or "類似事件描述：" in line:
            description = line.split('：', 1)[1] if '：' in line else line.split(':', 1)[1]
            description_list.append(description.strip())
        elif "類似事件因果分析:" in line or "類似事件因果分析：" in line:
            causal_analysis = line.split('：', 1)[1] if '：' in line else line.split(':', 1)[1]
            causal_analysis_list.append(causal_analysis.strip())
        elif "類似事件期間的經濟狀況:" in line or "類似事件期間的經濟狀況：" in line:
            economic_situation = line.split('：', 1)[1] if '：' in line else line.split(':', 1)[1]
            economic_situation_list.append(economic_situation.strip())
        
    info_dict = {
        '類似事件名稱': name_list,
        '類似事件起始結束日期': date_list,
        '類似事件描述': description_list,
        '類似事件因果分析': causal_analysis_list,
        '類似事件期間的經濟狀況': economic_situation_list
    }
    
    parsed_dates = [parse_date(date) for date in date_list]
    info_dict['類似事件年'] = [date for date in parsed_dates]
    return info_dict



#抽取字
#進到chain中去抽取字詞
def history_event(doc): 
    keyword_schema = Object(
    id="keyword_extract",
    description="keyword extraction",
    attributes=[
        Text(
            id='keyword',
            description='keyword extraction from a sentence'
        ),
        
    ],
    examples=[
        (
            '''請問台灣未來升息方向?''',
            [{'keyword': '升息影響',
              }]
        ),
        (
            '''請問為什麼今年美國SVB銀行會倒閉?''',
            [{'keyword': '銀行倒閉',
              }]
        ),
        (
            '''請問台灣能源產業的現況及未來發展?''',
            [{
                'keyword': '能源產業發展'
            }]
        ),
        (
            '''請問為什麼今年美國失業率下降影響''',
            [{
                'keyword': '美國失業率影響'
            }]
        ),
        (
            '''微軟併購暴雪''',
            [{
                'keyword':'微軟併購'
            }]
        ),
        (
            '''IBM收購紅帽''',
            [{
                'keyword':'IBM收購'
            }]
        )        
    ]
    )
    llm = ChatOpenAI(temperature=0.5, model_name="gpt-3.5-turbo")
    extraction_chain = create_extraction_chain(llm, keyword_schema)
    data = extraction_chain.run((doc))['data']
    return data

def get_related_history_events(key_word): #到LLM提問抓出相似歷史事件
    print(key_word)
    llm = ChatOpenAI(temperature  =0.5, model_name = "gpt-3.5-turbo")
    prompt_template = """
    I would like you to provide a comprehensive and detailed response when you are asked about any historical event.(more is better)
    Please list the events from old to new of their occurrence.你給的資料越新越好
    -----------------------
    Here is a question:
    {input}
    -----------------------
    in each similar event you mentioned, please provide the following information(please follwing the format)):
    here is the format that you have to follow and provide the answer in:
    - 類似事件名稱: 
    - 類似事件起始結束日期: 
    - 類似事件描述: 
    - 類似事件因果分析: 
    - 類似事件期間的經濟狀況:
    your final response should always in traditional Chinese
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["input"])
    chain = LLMChain(llm=llm, prompt=prompt)
    out = chain.run(f'我想知道過去是否曾經發生過類似{key_word}的近期歷史事件，若不只一件請全部告訴我，近期歷史事件請與{key_word}中的公司或產業有關。')
    print(out)
    final_answer = extract_info(out)
    print(final_answer)
    return final_answer
   


def get_wiki_search_results(query, history_events):
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    out = wikipedia.run(query)
    related_event_output_string = ""
    for i in range(len(history_events['類似事件名稱'])):
        related_event_output_string += f"事件名稱：{history_events['類似事件名稱'][i]}\n"
        related_event_output_string += f"起始結束日期：{history_events['類似事件起始結束日期'][i]}\n"
        related_event_output_string += f"描述：{history_events['類似事件描述'][i]}\n"
        related_event_output_string += f"因果分析：{history_events['類似事件因果分析'][i]}\n"
        related_event_output_string += f"期間的經濟狀況：{history_events['類似事件期間的經濟狀況'][i]}\n"
    
    
    #歷史資料跟維基百科結合
    related_history_and_wiki_events = f'過去發生相似歷史事件:{related_event_output_string}\n + 維基百科相關資料:{out}'
    
    
    llm = ChatOpenAI(temperature  =0.5, model_name = "gpt-3.5-turbo-16k")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200, separators=["\n\n", "\n", "", " "])  
    docs = text_splitter.create_documents([related_history_and_wiki_events])
    map_prompt = """
    As an expert in data organization, I kindly request that you provide a comprehensive and highly detailed summary based on the article I am about to share with you. I expect your response to be precise, thorough, and of the highest quality.
    "{text}"
    """
    map_prompt_templete = PromptTemplate(template=map_prompt, input_variables=['text'])
    combine_prompt = """
    As an economist and financial expert, I would like you to create a comprehensive summary by incorporating all the articles I have provided.
    Please utilize your extensive economic knowledge and database to determine the complete cause and effect of the event. The summary should address the following aspects: the government's involvement and assistance in the event, the actions taken by the company in response to the event, the perspective of the market, the impact on the industry, and the future implications for the company, the country, and even the world.
    '''{text}'''
    The response must always use "Traditional Chinese"
    """

    combine_prompt_templete = PromptTemplate(template=combine_prompt, input_variables=['text'])
    summarize_chain = load_summarize_chain(llm=llm,
                                       chain_type='map_reduce', #如何處理的方式
                                       verbose=True, 
                                       map_prompt=map_prompt_templete, 
                                       combine_prompt=combine_prompt_templete)
    out = summarize_chain.run(docs)
    return out


def merge_document(document):
    merge_content = ""
    for i in range(len(document)):
        merge_content += document[i].page_content
    document[0].page_content = merge_content
    return [document[0]]

def get_related_info(query):
    prompt_template = '''Tell me a summary of the following paragraph, the shorter the better, no more than 5 words, and do not mention any dates:
    ======
    {text}  
    ======
    pleease use English to answer the question:
    Here are some examples:
    human:請問下個月美國經濟走勢如何?
    AI: The U.S. Economy 

    human:請問台灣能源產業的現況及未來發展?
    Energy Industry Development

    human:請問2023下半年美國會升息嗎?
    Interest Rate impact

    human:請問為什麼今年美國SVB銀行會倒閉?
    Bank failures

    human:請問台灣未來升息方向?
    Interest Rate impact

    human:請問為什麼今年美國失業率下降影響
    Interest U.S. Unemployment Rate imapct

    '''
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
    llm = ChatOpenAI(temperature  =0.5, model_name = "gpt-3.5-turbo")

    llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=False)

    out = llm_chain(query)['text']
    return out

#抓事件名稱跟日期
#重寫搜尋函數
def event_time_inquiry(question):
    llm = ChatOpenAI(temperature=0, model_name="gpt-4")
    tools = load_tools(["serpapi", "llm-math"], llm=llm)
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, max_iterations=2)
    query = f"可以告訴我{question}事件的開始時間(年-月-日)嗎?"
    raw_answer = agent.run(query)
    final_input = f'問題: {query}\n 回答:{raw_answer}'
    prompt_template = """
    As an expert in data analysis, you will receive a query and a related narrative containing information about an event's occurrence time. 
    I would like you to provide a response in a list format, including the mentioned event and its corresponding 
    time (please use the standard format). For example, your response should be ['事件名稱', 'year-month-day']. Please ensure your response is clear, 
    structured, and detailed to provide the most precise, comprehensive, and high-quality information.
    -----------------------
    Here is a question:
    {input}
    -----------------------
    here is the example of the response:
    your response should only be:['Event Name', '2023-04-01']
    your final response should only include the event name and the time of the event, and the format should be the same as the example.
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["input"])
    chain = LLMChain(llm=llm, prompt=prompt)
    final_answer = chain.run(final_input)
    print(final_answer)
    final_answer_list = ast.literal_eval(final_answer)
    return final_answer_list
    
def get_event_and_related_event(query):
    related_event = get_related_history_events(query)
    event = event_time_inquiry(query)
    return event, related_event

def CauseAnalysisWebAPI(input_query: str, history_events: dict, event):
    llm = ChatOpenAI(temperature  =0.5, 
                     model_name = "gpt-3.5-turbo-16k")
    
    '''依照關鍵字搜尋google前n個網址並總結'''
    num_news = 10
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""])
    
    #refine方法中的template
    prompt_template = """As a master of data organization, please write a summary based on the article I give you.The response must always use "Traditional Chinese"
    
    {text}
    """
    web_prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
    
    
    refine_templete = """
    As an economist and financial expert, I would like you to create a comprehensive summary by incorporating all the articles I have provided. 
    Please utilize your extensive economic knowledge and database to determine the complete cause and effect of the event. The summary should address the following aspects: the government's involvement and assistance in the event, the actions taken by the company in response to the event, the perspective of the market, the impact on the industry, and the future implications for the company, the country, and even the world.
    ------------
    {text}
    -----------
    The response must always use "Traditional Chinese"
    please provide Your opnions and similar historical events.
    """

    
    refine_prompt = PromptTemplate(
        template = refine_templete, 
        input_variables = ["text"]
    )
    
    #google search api params
    params = {
        "q": f"{input_query}",
        "tbm": "nws",
        "google_domain": "google.com",
        "api_key": serpapi,
        "num": f"{num_news}",
    }
    
   
    search = GoogleSearch(params)
    results = search.get_dict()
    
    #get website title
    title_news, link_news, source, date = [], [], [], []
    for i in range(len(results['news_results'])):
        title_news.append(results['news_results'][i]['title'])
        link_news.append(results['news_results'][i]['link'])
        source.append(results['news_results'][i]['source'])
        date.append(results['news_results'][i]['date'])
    print(f"related top {num_news} title: {title_news}")
    print(f"related top {num_news} link: {link_news}")
    print(f"related top {num_news} source: {source}")
    print(f"related top {num_news} date: {date}")
    
    ## 新聞存成字典回傳
    news_data = {
                'title': title_news,
                'link': link_news,
            }
    documents = []
    for link in link_news:
        try:
            loader = WebBaseLoader(link)
            document = loader.load()
            documents += document
        except Exception as e:
            print(f"Error loading document: {e}")
            continue
    
    split_doce = text_splitter.split_documents(documents)
    split_doce = remove_unnessary_word(split_doce)
    
    PINECONE_API_KEY = pinecone_api
    PINECONE_API_ENV = pinecone_env
    embeddings = OpenAIEmbeddings()
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
    index_name = pinecone_index_name
    docsearch = Pinecone.from_texts([t.page_content for t in split_doce], embeddings, index_name=index_name)
    
    query =  f"{input_query}"
    docs = docsearch.similarity_search(query, 100)
    in_word_count = 50  # 最低字數閾值
    
    docs = remove_duplicate_document(docs)
    docs = remove_short_documents(docs, in_word_count)
    docs = increase_docs_diversity(docs, 4)
    
    for doc in docs:
        doc = extract_text(doc)
        print("here is the doc: ", doc)
    
        
    web_sum_chain = load_summarize_chain(llm, chain_type="refine", 
                                         question_prompt=web_prompt, 
                                         refine_prompt=refine_prompt, 
                                         verbose=True)
    
    result = web_sum_chain.run(docs)

    out = get_related_info(query)
    print("關鍵字所屬類別", out)
    #維基百科搜尋
    wiki_search_summary = get_wiki_search_results(out, history_events)
    #結合wiki_search和result結果，並在跑一次語言模型
    cause_and_effect_input = f"需要因果分析事件名稱:{event[0]}\n該事件發生時間:{event[1]}\n\n過去發生相似歷史事件以及相關維基資料:\n{wiki_search_summary} \n\n相關新聞摘要:\n{result}"
    
    prompt_template = """
    You are a macro international economist as well as a financial scientist, please consider the general economy, individual economic orientations, and Wikipedia based on general economic and financial principles, using all of your specialized information as well as insight into past historical information and links to intergovernmental relations.
    -----------------------
    {input}
    -----------------------
    Please provide a comprehensive cause-and-effect analysis in the specified format. Utilize your extensive economic knowledge and the information available in your database to infer the complete cause and effect.
    The analysis should cover the government's role and assistance, the company's response to the event, the market's perspective, the impact on the industry, and the future implications for the company, the country, and the world.
    The following is the format you must provide to analyze the results of the cause and effect of the event(at least 3 causes and effects, more details is better):
    - 背景: [描述事件]
    原因:
    - 原因1: [描述第一個原因]
    - 原因2: [描述第二個原因]
    - 原因3: [描述第三個原因]
    未來影響:
    - 未來影響1: [描述第一個有可能的未來影響]
    - 未來影響2: [描述第二個有可能的未來影響]
    - 未來影響3: [描述第三個有可能的未來影響]
    我的觀點: 
    - [你的意見和觀點](The answer must be specific.)
    The response must always use "Traditional Chinese"
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["input"])
    chain = LLMChain(llm=llm, prompt=prompt)
    final_cause_and_effect_output = chain.run(cause_and_effect_input)
    
    return final_cause_and_effect_output, result, wiki_search_summary, cause_and_effect_input, news_data

#前置函數
def parse_code(text: str, lang: str = "") -> str:
    pattern = rf'```{lang}.*?\s+(.*?)```'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        code = match.group(1)
    else:
        print(f"{pattern} not match following text:")
        raise Exception
    return code

def mermaid_to_svg(mermaid_code, output_file):
    # Write the Mermaid code to a temporary file
    tmp = Path(f'{output_file}.mmd')
    tmp.write_text(mermaid_code, encoding='utf-8')
    current_dir = os.getcwd()
    # 構建儲存圖片的路徑
    output_file = f'{output_file}.svg'
    save_path = os.path.join(current_dir, 'static', output_file)

    # Call the mmdc command to convert the Mermaid code to a SVG
    mmdc_path = shutil.which('mmdc.cmd')
    subprocess.run([mmdc_path, '-i', str(tmp), '-o', save_path]) 
    ##路徑要放mmdc的位置，不然有時候會找不到，可以用print(subprocess.run(['where', 'mmdc'], capture_output=True).stdout)找
    
    
def draw_diagram(casual_result):
    llm = ChatOpenAI(temperature  =0.5, model_name = "gpt-3.5-turbo-16k")
    prompt_template = """
    ## Format example that you should follow(do not use this as your answer), and please use "" to wrap the text
    {format_example}

    ### Implementation: provided as plain text
    You can add different colors to make the flowchart look better.
    according to the following article, provide causal analysis mermaid diagram code in markdown codeblock, the diagram should detail the cause and effect sequence, and be complete and detailed.:
    -----------------------
    # Context
    {input}
    -----------------------
    according to the above article, provide causal analysis mermaid diagram code in markdown codeblock, the diagram should detail the cause and effect sequence, and be complete and detailed.:
    ## Rules
    You should add different colors to make the flowchart look better.
    Guidelines when creating the graph diagram in any diagram language:
    - Avoid linear diagrams when possible, diagrams should be hierarchical and have multiple branches when applicable.
    - Don't add the label if its the same as the destination node.

    Important rules when creating the graph diagram in mermaid syntax:
    - Prefer using graph TB types of diagrams.
    - Never use the ampersand (&) symbol in the diagram, it will break the diagram. Use the word "and" instead. For example use "User and Admin" instead of "User & Admin".
    - Never use round brackets () in the node identifiers, node labels and edge labels, it will break the diagram. Use a coma instead. For example use "User, Admin" instead of "User (Admin)".
    - Don't use empty labels "" for edges, instead don't label the edge at all. For example U["User"] --> A["Admin"].
    - Avoid using semicolon as line separator, prefer using new-lines instead. For example use "graph LR\n  A --> B" instead of "graph LR;  A --> B"

    Rules when using graph diagrams in mermaid syntax:
    - Use short node identifiers, for example U for User or FS for File System.
    - Always use double quotes for node labels, for example U["User"].
    - Never create edges that connect to only one node; each edge should always link two nodes together. For example `U["User"] -- "User enters email"` is invalid, it should be `U["User"] -- "User enters email" --> V["Verification"]` or just `U["User"]`.
    - Always use double quotes for edge labels, for example U["User"] -- "User enters email" --> V["Verification"].
    - Indentation is very important, always indent according to the examples below.
    mermaid themes:
    - default - This is the default theme for all diagrams.
    - neutral - This theme is great for black and white documents that will be printed.
    - dark - This theme goes well with dark-colored elements or dark-mode.
    - forest - This theme contains shades of green.
    - base - This is the only theme that can be modified. Use this theme as the base for customizations.
    Rules when using graph diagrams with subgraphs in mermaid syntax:
    Never refer to the subgraph root node from within the subgraph itself.
    """
    FORMAT_EXAMPLE ="""
    ## Data structures and interface definitions
    ```mermaid
    graph TB

    subgraph diagram
        Background["背景：富邦金控併購日盛事件"]
        Cause1["原因1：擴大市場份額"]
        Cause2["原因2：提高競爭力"]
        Cause3["原因3：實現經濟規模效益"]
        Future1["未來影響1：金融業結構調整"]
        Future2["未來影響2：金融市場穩定性"]
        Future3["未來影響3：經濟發展和就業機會"]
        
        Background -->|"富邦金控可以獲得日盛金控的客戶和業務，\n進一步擴大其在金融業的影響力和市場份額"| Cause1
        Background -->|"富邦金控可以獲得日盛金控的資源和專業知識，\n提高自身的綜合實力和競爭力"| Cause2
        Background -->|"富邦金控可以整合兩家公司的資源和業務，實現成本節約和效率提升"| Cause3
        Cause1 -->|"可能導致"| Future1
        Cause2 -->|"可能導致"| Future2
        Cause3 -->|"可能導致"| Future3

    end

    subgraph Impact
        Impact1["金融業結構調整"]
        Impact2["金融市場穩定性"]
        Impact3["經濟發展和就業機會"]
        
        Future1 -->|"可能引發其他金融機構之間的併購和整合"| Impact1
        Future2 -->|"可能引發市場不確定性和投資者的擔憂，導致市場波動和風險增加"| Impact2
        Future3 -->|"可能帶來新的商機和投資，促進經濟增長和就業機會的增加"| Impact3
    end

    Background:::neutral
    Cause1:::forest
    Cause2:::forest
    Cause3:::forest
    Future1:::dark
    Future2:::dark
    Future3:::dark
    Impact1:::base
    Impact2:::base
    Impact3:::base
    ```
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["input", "format_example"])
    chain = LLMChain(llm=llm, prompt=prompt)
    mermmaid_text = chain.run(input = casual_result, format_example = FORMAT_EXAMPLE)
    mermaid_code = parse_code(mermmaid_text, 'mermaid')
    mermaid_to_svg(mermaid_code, 'my_graph')
      
    
    
    #財務分析
    #api和格式整理


#前置作業   
#抓新聞  
def get_company_news(company_name, num_news):
    # 从google新闻获取公司新闻
    params = {
        "engine": "google",
        "q": f'{company_name}',
        "tbm": "nws",
        "api_key": serpapi,
        "num":  f"{num_news}"
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    title_news, link_news, source, date = [], [], [], []
    for i in range(len(results['news_results'])):
        title_news.append(results['news_results'][i]['title'])
        link_news.append(results['news_results'][i]['link'])
        source.append(results['news_results'][i]['source'])
        date.append(results['news_results'][i]['date'])
    print(f"related top {num_news} title: {title_news}")
    print(f"related top {num_news} link: {link_news}")
    print(f"related top {num_news} source: {source}")
    print(f"related top {num_news} date: {date}")
    
    return title_news, link_news, source, date

import matplotlib
#畫技術、財務圖表
font_path = 'C:/Windows/Fonts/msjh.ttc' 
font_prop = matplotlib.font_manager.FontProperties(fname=font_path)
matplotlib.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['axes.unicode_minus'] =False
#插入歷史事件到圖表中
def plot_stock_event_analysis(ticker_symbol, event, extra_events=None):
    extra_events = {'類似事件名稱': extra_events['類似事件名稱'], '類似事件開始時間':extra_events['類似事件日期']}
    ticker = yf.Ticker(ticker_symbol)
    event_date = pd.to_datetime(event[1]).date()
    if extra_events is not None:
        extra_events_dates = [pd.to_datetime(date).date() for date in extra_events['類似事件開始時間']]
    else:
        extra_events_dates = [] 
        
    earliest_date = min(event_date,  *extra_events_dates)
    earliest_date = pd.to_datetime(earliest_date).date()
    
    #計算最早日期
    start_date = earliest_date - pd.DateOffset(months=6)
    
    end_date = pd.Timestamp.today().date()
    
    #畫圖
    df = ticker.history(start=start_date, end=end_date)
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date']).dt.date

    # 計算均線
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()

    # 股價跟均線圖表
    plt.figure(figsize=(16, 8))
    plt.plot(df['Date'], df['Close'], label=f'{ticker_symbol}收盤價', color='#5e94b7')
    plt.plot(df['Date'], df['MA_50'], label='50日均線', color='#FFB01C')
    ##74bf76
    #添加事件箭頭和文字
    event_date = pd.to_datetime(event[1]).date()
    event_text = f'{event[0]}\n( {event_date} )'
    event_price = df[df['Date'] == event_date]['Close'].values[0]
    plt.annotate(event_text, xy=(event_date, event_price), xytext=(event_date, event_price+(max(df['Close'])-min(df['Close'])) *0.3),
                 arrowprops=dict(arrowstyle='-|>', color='red'), fontproperties=font_prop, fontsize=15, 
                 ha='center', va='top')
    
    #添加額外事件
    if extra_events is not None:
        for i, event_info in enumerate(extra_events['類似事件開始時間']):
            event_date = pd.to_datetime(event_info).date()
            event_name = extra_events['類似事件名稱'][i]
            event_text = f'{event_name}\n( {event_date} )'
            event_price = df[df['Date'] == event_date]['Close'].values[0]
            plt.annotate(event_text, xy=(event_date, event_price), xytext=(event_date, event_price+(max(df['Close'])-min(df['Close'])) *0.3),
                         arrowprops=dict(arrowstyle='-|>', color='red'), fontproperties=font_prop,
                         fontsize=12, ha='center', va='top')
    plt.xlabel('日期', fontsize=15)
    plt.ylabel('價格', fontsize=15)
    plt.gca().xaxis.set_tick_params(labelsize=12)
    plt.gca().yaxis.set_tick_params(labelsize=12)
    plt.title(f'{ticker_symbol} 股價與事件表', fontsize=15)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()
    
#RSI中插入事件
#RSI
def plot_RSI_event_analysis(ticker_symbol, event):
    # 
    ticker = yf.Ticker(ticker_symbol)
    
    try:
        # 股價歷史數據
        event_date = pd.to_datetime(event[1]).date()
        start_date = event_date - pd.DateOffset(months=6)
        end_date = pd.Timestamp.today().date()
        
        df = ticker.history(start=start_date, end=end_date)
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date']).dt.date

        # 計算均線
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        # 計算RSI
        window = 14
        delta = df['Close'].diff()
        gain = delta.mask(delta < 0, 0)
        loss = -delta.mask(delta > 0, 0)
        average_gain = gain.rolling(window).mean()
        average_loss = loss.rolling(window).mean()
        rs = average_gain / average_loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # 畫股價
        plt.figure(figsize=(16, 8))
        plt.plot(df['Date'], df['RSI'], label=f'{ticker_symbol} RSI', color='#5e94b7')
        plt.axhline(y=70, color='red', linestyle='--', label='Overbought')
        plt.axhline(y=30, color='green', linestyle='--', label='Oversold')
        

        # 事件和文字
        closest_date = df[df['Date'] <= event_date]['Date'].max()
        event_date = pd.to_datetime(event[1]).date()
        event_text = f'{event[0]}\n( {event_date} )'
        event_price = df[df['Date'] == closest_date]['RSI'].values[0]
        plt.annotate(event_text, xy=(closest_date, event_price), xytext=(closest_date, event_price+15),
                    arrowprops=dict(arrowstyle='-|>', color='red'), fontproperties=font_prop, fontsize=15, 
                    ha='center', va='top')

        plt.xlabel('日期', fontsize=15)
        plt.ylabel('RSI', fontsize=15)
        plt.gca().xaxis.set_tick_params(labelsize=12)
        plt.gca().yaxis.set_tick_params(labelsize=12)
        plt.title(f'{ticker_symbol} RSI與事件表', fontsize=15)
        plt.legend(fontsize=12)
        plt.grid(True)
        current_dir = os.getcwd()
        # 構建儲存圖片的路徑
        save_path = os.path.join(current_dir, 'static', 'RSI')
        plt.savefig(save_path)
    except Exception as e:
        print(f"Error: {e}")
        end_date = pd.Timestamp.today().date()
        start_date = end_date - timedelta(days=365)

        df = ticker.history(start=start_date, end=end_date)
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        # 計算均線
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        # 計算RSI
        window = 14
        delta = df['Close'].diff()
        gain = delta.mask(delta < 0, 0)
        loss = -delta.mask(delta > 0, 0)
        average_gain = gain.rolling(window).mean()
        average_loss = loss.rolling(window).mean()
        rs = average_gain / average_loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # 畫股價
        plt.figure(figsize=(16, 8))
        plt.plot(df['Date'], df['RSI'], label=f'{ticker_symbol} RSI', color='#5e94b7')
        plt.axhline(y=70, color='red', linestyle='--', label='Overbought')
        plt.axhline(y=30, color='green', linestyle='--', label='Oversold')
        plt.xlabel('日期', fontsize=15)
        plt.ylabel('RSI', fontsize=15)
        plt.gca().xaxis.set_tick_params(labelsize=12)
        plt.gca().yaxis.set_tick_params(labelsize=12)
        plt.title(f'{ticker_symbol} Last year RSI', fontsize=15)
        plt.legend(fontsize=12)
        plt.grid(True)
        current_dir = os.getcwd()
        # 構建儲存圖片的路徑
        save_path = os.path.join(current_dir, 'static', 'RSI')
        plt.savefig(save_path)
        
        
        
    
#MACD插入事件
def plot_MACD_event_analysis(ticker_symbol, event):
    ticker = yf.Ticker(ticker_symbol)
    
    try:
        # 股價歷史數據
        event_date = pd.to_datetime(event[1]).date()
        start_date = event_date - pd.DateOffset(months=6)
        end_date = pd.Timestamp.today().date()
        
        df = ticker.history(start=start_date, end=end_date)
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        # 計算均線
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        # Calculate MACD
        short_ema = df['Close'].ewm(span=12, adjust=False).mean()
        long_ema = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = short_ema - long_ema
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        # 畫股價
        plt.figure(figsize=(16, 8))
        plt.plot(df['Date'], df['MACD'], label='MACD', color='#5e94b7')
        plt.plot(df['Date'], df['Signal_Line'], label='Signal_Line', color='#FFB01C')
        

        # 事件和文字
        closest_date = df[df['Date'] <= event_date]['Date'].max()
        event_date = pd.to_datetime(event[1]).date()
        event_text = f'{event[0]}\n( {event_date} )'
        event_price = df[df['Date'] == closest_date]['MACD'].values[0]
        plt.annotate(event_text, xy=(closest_date, event_price), xytext=(closest_date, event_price+(max(df['MACD'])-min(df['MACD'])) *0.3),
                    arrowprops=dict(arrowstyle='-|>', color='red'), fontproperties=font_prop, fontsize=15, 
                    ha='center', va='top')
    
        plt.xlabel('日期', fontsize=15)
        plt.ylabel('Price', fontsize=15)
        plt.gca().xaxis.set_tick_params(labelsize=12)
        plt.gca().yaxis.set_tick_params(labelsize=12)
        plt.title(f'{ticker_symbol} MACD與事件表', fontsize=15)
        plt.legend(fontsize=12)
        plt.grid(True)
        current_dir = os.getcwd()
        # 構建儲存圖片的路徑
        save_path = os.path.join(current_dir, 'static', 'MACD')
        plt.savefig(save_path)
    except Exception as e:
        print(f"Error: {e}")
        end_date = pd.Timestamp.today().date()
        start_date = end_date - timedelta(days=365)

        df = ticker.history(start=start_date, end=end_date)
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        # 計算均線
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        # Calculate MACD
        short_ema = df['Close'].ewm(span=12, adjust=False).mean()
        long_ema = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = short_ema - long_ema
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        # 畫股價
        plt.figure(figsize=(16, 8))
        plt.plot(df['Date'], df['MACD'], label='MACD', color='#5e94b7')
        plt.plot(df['Date'], df['Signal_Line'], label='Signal_Line', color='#FFB01C')
        plt.xlabel('日期', fontsize=15)
        plt.ylabel('Price', fontsize=15)
        plt.gca().xaxis.set_tick_params(labelsize=12)
        plt.gca().yaxis.set_tick_params(labelsize=12)
        plt.title(f'{ticker_symbol} Last year MACD ', fontsize=15)
        plt.legend(fontsize=12)
        plt.grid(True)
        current_dir = os.getcwd()
        # 構建儲存圖片的路徑
        save_path = os.path.join(current_dir, 'static', 'MACD')
        plt.savefig(save_path)
        
    
#布林通道(單一事件)
def plot_bollinger_event_analysis(ticker_symbol, event):
    # 
    ticker = yf.Ticker(ticker_symbol)

    try:
        # 股價歷史數據
        event_date = pd.to_datetime(event[1]).date()
        start_date = event_date - pd.DateOffset(months=6)
        end_date = pd.Timestamp.today().date()
        
        df = ticker.history(start=start_date, end=end_date)
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        # 計算均線
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        # 計算布林通道指標
        rolling_mean = df['Close'].rolling(window=20).mean()
        rolling_std = df['Close'].rolling(window=20).std()
        upper_band = rolling_mean + 2 * rolling_std
        lower_band = rolling_mean - 2 * rolling_std
        # 畫股價
        plt.figure(figsize=(16, 8))
        plt.plot(df['Date'], df['Close'], label=f'{ticker_symbol}收盤價', color='#5e94b7')
        plt.plot(df['Date'], df['MA_20'], label='20日均線', color='#E47F5E')
        plt.plot(df['Date'], upper_band, label='upper_band', color='#d4d4d4')
        plt.plot(df['Date'], lower_band, label='lower_band', color='#d4d4d4')
        ##6ebf77

        # 事件和文字
        closest_date = df[df['Date'] <= event_date]['Date'].max()
        event_date = pd.to_datetime(event[1]).date()
        event_text = f'{event[0]}\n( {event_date} )'
        event_price = df[df['Date'] == closest_date]['Close'].values[0]
        plt.annotate(event_text, xy=(closest_date, event_price), xytext=(closest_date, event_price+(max(df['Close'])-min(df['Close'])) *0.3),
                    arrowprops=dict(arrowstyle='-|>', color='red'), fontproperties=font_prop, fontsize=15, 
                    ha='center', va='top')
    
        plt.xlabel('日期', fontsize=15)
        plt.ylabel('Price', fontsize=15)
        plt.gca().xaxis.set_tick_params(labelsize=12)
        plt.gca().yaxis.set_tick_params(labelsize=12)
        plt.title(f'{ticker_symbol} 布林通道與事件表', fontsize=15)
        plt.legend(fontsize=12)
        plt.grid(True)
        current_dir = os.getcwd()
        # 構建儲存圖片的路徑
        save_path = os.path.join(current_dir, 'static', 'bollinger')
        plt.savefig(save_path)
    except Exception as e:
        print(f"Error: {e}")
        end_date = pd.Timestamp.today().date()
        start_date = end_date - timedelta(days=365)
        df = ticker.history(start=start_date, end=end_date)
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        # 計算均線
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        # 計算布林通道指標
        rolling_mean = df['Close'].rolling(window=20).mean()
        rolling_std = df['Close'].rolling(window=20).std()
        upper_band = rolling_mean + 2 * rolling_std
        lower_band = rolling_mean - 2 * rolling_std
        # 畫股價
        plt.figure(figsize=(16, 8))
        plt.plot(df['Date'], df['Close'], label=f'{ticker_symbol}收盤價', color='#5e94b7')
        plt.plot(df['Date'], df['MA_20'], label='20日均線', color='#E47F5E')
        plt.plot(df['Date'], upper_band, label='upper_band', color='#d4d4d4')
        plt.plot(df['Date'], lower_band, label='lower_band', color='#d4d4d4')
        plt.xlabel('日期', fontsize=15)
        plt.ylabel('Price', fontsize=15)
        plt.gca().xaxis.set_tick_params(labelsize=12)
        plt.gca().yaxis.set_tick_params(labelsize=12)
        plt.title(f'{ticker_symbol} last year bollinger', fontsize=15)
        plt.legend(fontsize=12)
        plt.grid(True)
        current_dir = os.getcwd()
        # 構建儲存圖片的路徑
        save_path = os.path.join(current_dir, 'static', 'bollinger')
        plt.savefig(save_path)
        
        

#單一事件   
def plot_only_stock_event_analysis(ticker_symbol, event):
    ticker = yf.Ticker(ticker_symbol)
    
    try:
        # 股價歷史數據
        event_date = pd.to_datetime(event[1]).date()
        start_date = event_date - pd.DateOffset(months=6)
        end_date = pd.Timestamp.today().date()
        
        df = ticker.history(start=start_date, end=end_date)
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date']).dt.date

        
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()

        # 股價跟均線圖表
        plt.figure(figsize=(16, 8))
        plt.plot(df['Date'], df['Close'], label=f'{ticker_symbol}收盤價', color='C0')
        plt.plot(df['Date'], df['MA_20'], label='20日均線', color='green')
        plt.plot(df['Date'], df['MA_50'], label='50日均線', color='red')

        # 事件和文字
        closest_date = df[df['Date'] <= event_date]['Date'].max()
        event_text = f'{event[0]}\n( {closest_date} )'
        event_price = df[df['Date'] == closest_date]['Close'].values[0]
        plt.annotate(event_text, xy=(closest_date, event_price), xytext=(closest_date, event_price+(max(df['Close'])-min(df['Close'])) *0.6),
                    arrowprops=dict(arrowstyle='-|>', color='red'), fontproperties=font_prop, fontsize=15, 
                    ha='center', va='top')

        plt.xlabel('日期', fontsize=15)
        plt.ylabel('價格', fontsize=15)
        plt.gca().xaxis.set_tick_params(labelsize=12)
        plt.gca().yaxis.set_tick_params(labelsize=12)
        plt.title(f'{ticker_symbol} 股價與事件表', fontsize=15)
        plt.legend()
        plt.grid(True)
        current_dir = os.getcwd()
        # 構建儲存圖片的路徑
        save_path = os.path.join(current_dir, 'static', 'PRICE')
        plt.savefig(save_path)
    except Exception as e:
        print(f"Error: {e}")
        end_date = pd.Timestamp.today().date()
        start_date = end_date - timedelta(days=365)
        df = ticker.history(start=start_date, end=end_date)
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()

        # 股價跟均線圖表
        plt.figure(figsize=(16, 8))
        plt.plot(df['Date'], df['Close'], label=f'{ticker_symbol}收盤價', color='C0')
        plt.plot(df['Date'], df['MA_20'], label='20日均線', color='green')
        plt.plot(df['Date'], df['MA_50'], label='50日均線', color='red')
        plt.xlabel('日期', fontsize=15)
        plt.ylabel('價格', fontsize=15)
        plt.gca().xaxis.set_tick_params(labelsize=12)
        plt.gca().yaxis.set_tick_params(labelsize=12)
        plt.title(f'{ticker_symbol} last year price', fontsize=15)
        plt.legend()
        plt.grid(True)
        current_dir = os.getcwd()
        # 構建儲存圖片的路徑
        save_path = os.path.join(current_dir, 'static', 'PRICE')
        plt.savefig(save_path)
        
        
    
#損益表
def plot_financials(stock_code):
    # 讀取財務數據
    income_statement_path = os.path.join(os.getcwd(), f'{stock_code}_financial_income_statement.csv')
    data = pd.read_csv(income_statement_path)
    data['asOfDate'] = pd.to_datetime(data['asOfDate'])
    new_column_names = {
        'asOfDate': 'Date'
    }
    data.rename(columns=new_column_names, inplace=True)
    data = data.sort_values('Date')
    data['GrossProfit'] = data['TotalRevenue'] - data['CostOfRevenue']
    data_sub = data[['Date', 'TotalRevenue', 'GrossProfit', 'OperatingIncome', 'PretaxIncome', 'NetIncome']]

    # 設定折線圖顏色
    colors = {
        'TotalRevenue': 'black',  # 營收
        'GrossProfit': 'blue',    # 毛利
        'OperatingIncome': 'green',   # 營業收益
        'PretaxIncome': 'red',   # 稅前淨利
        'NetIncome': 'purple'    # 稅後淨利
    }

    # 繪製折線圖
    plt.figure(figsize=(14, 8))
    for column in data_sub.columns[1:]:
        plt.plot(data_sub['Date'], data_sub[column] / 1e9, label=column, marker='o', color=colors[column])  # 轉換為十億
    plt.title(f'{stock_code} Key Financials Over Time', fontsize=14)
    plt.xlabel('Date', fontsize=10)
    plt.ylabel('Amount (Billion)', fontsize=10)
    plt.legend(fontsize=8)
    plt.grid(True)
    current_dir = os.getcwd()
    # 構建儲存圖片的路徑
    save_path = os.path.join(current_dir, 'static', 'financials')
    plt.savefig(save_path)
    


#月均價跟損益表
def plot_accounting_and_stock(accounting_subject, stock_code):
    # 讀取股價資料
    stock_data_path = os.path.join(os.getcwd(), f'{stock_code}_financial_stock_evolution.csv')
    stock_data = pd.read_csv(stock_data_path)
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data.set_index('Date', inplace=True)  # 將日期設置為索引
    stock_data_monthly = stock_data.resample('M').last()  # 將每日資料重新取樣為每月資料

    # 讀取會計科目資料
    data_path = os.path.join(os.getcwd(), f'{stock_code}_financial_income_statement.csv')
    data = pd.read_csv(data_path)
    data['asOfDate'] = pd.to_datetime(data['asOfDate'])
    new_column_names = {
        'asOfDate': 'Date'
    }
    data.rename(columns=new_column_names, inplace=True)
    data = data.sort_values('Date')
    data['GrossProfit'] = data['TotalRevenue'] - data['CostOfRevenue']
    data_sub = data[['Date', 'TotalRevenue', 'GrossProfit', 'OperatingIncome', 'PretaxIncome', 'NetIncome']]

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_title(f'{stock_code} {accounting_subject} and Close Price', fontsize=14)

    # 調整柱子的寬度為 0.9
    ax1.bar(data_sub['Date'], data_sub[accounting_subject] / 1e9, alpha=0.6, width=30)

    ax1.set_xlabel('Date', fontsize=10)
    ax1.set_ylabel(f'{accounting_subject} (Billion)', fontsize=10)
    ax1.tick_params(axis='y')
    ax2 = ax1.twinx()

    start_date = data_sub['Date'].iloc[0]
    end_date = datetime.today()
    selected_data = stock_data_monthly[(stock_data_monthly.index >= start_date) & (stock_data_monthly.index <= end_date)]

    ax2.plot(selected_data.index, selected_data['Close'], color='red')
    ax2.set_ylabel('Monthly_Close Price', fontsize=10)
    ax2.tick_params(axis='y')
    ax1.legend([accounting_subject], loc='upper left')
    ax2.legend(['Close Price'], loc='upper right')

    current_dir = os.getcwd()
    # 構建儲存圖片的路徑
    save_path = os.path.join(current_dir, 'static', 'accounting_and_stock')
    plt.savefig(save_path)
    
#財務比率 
def plot_stock_valuation(stock_code):
    # 讀取股價資料
    stock_data_path = os.path.join(os.getcwd(), f'{stock_code}_financial_stock_evolution.csv')
    stock_data = pd.read_csv(stock_data_path)
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data.set_index('Date', inplace=True)  # 將日期設置為索引
    stock_data_monthly = stock_data.resample('M').last()

    # 讀取財務評估指標資料
    data_path = os.path.join(os.getcwd(), f'{stock_code}_financial_valuation_measures.csv')
    data = pd.read_csv(data_path)
    data = data.rename(columns={'asOfDate': 'Date'})
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.fillna(method='ffill')
    data = data.fillna(method='bfill')
    start_date = data['Date'].iloc[0]
    end_date = datetime.today()
    selected_data = stock_data_monthly[(stock_data_monthly.index >= start_date) & (stock_data_monthly.index <= end_date)]

    # Select the ratios to plot
    ratios_to_plot = ['PbRatio', 'PeRatio', 'PegRatio', 'PsRatio']

    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Plot each ratio on ax1
    for ratio in ratios_to_plot:
        ax1.plot(data['Date'], data[ratio], label=ratio, marker='o')

    ax1.set_xlabel('Date', fontsize=10)
    ax1.set_ylabel('Ratio', fontsize=10)
    ax1.set_title(f'{stock_code} Financial Valuation Ratios over Time', fontsize=14)

    # Create ax2 with twinx to share the same x-axis as ax1
    ax2 = ax1.twinx()

    # Plot the monthly close price on ax2
    ax2.plot(selected_data.index, selected_data['Close'], label='Monthly Close Price', color='black', marker='o')
    ax2.set_ylabel('Close Price', fontsize=10)

    # Combine the legends of ax1 and ax2
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.grid(True)
    current_dir = os.getcwd()
    # 構建儲存圖片的路徑
    save_path = os.path.join(current_dir, 'static', 'stock_valuation1')
    plt.savefig(save_path)
    
    # Select the other measures to plot
    measures_to_plot = ['EnterpriseValue', 'MarketCap']

    fig, axs = plt.subplots(len(measures_to_plot), 1, figsize=(14, 8))

    # Plot each measure
    for i, measure in enumerate(measures_to_plot):
        axs[i].plot(data['Date'], data[measure], label=measure,  marker='o')
        axs[i].set_xlabel('Date', fontsize=10)
        axs[i].set_ylabel('Value', fontsize=10)
        axs[i].set_title(f'MSFT {measure} over Time', fontsize=14)
        axs[i].legend()
        axs[i].grid(True)

    fig.tight_layout()
    current_dir = os.getcwd()
    # 構建儲存圖片的路徑
    save_path = os.path.join(current_dir, 'static', 'stock_valuation2')
    plt.savefig(save_path)
    
   
def plot_stock_event_casual_analysis(ticker_symbol, event, event_casual, extra_events=None):
    ticker = yf.Ticker(ticker_symbol)
    #畫股價跟事件圖
    try:
        plot_only_stock_event_analysis(ticker_symbol, event)
    except Exception as e:
        print(f"Error occurred during stock event plotting: {e}")
    #畫歷史事件圖
    #plot_stock_event_analysis(ticker_symbol, event, extra_events)
    #畫RSI
    try:
        plot_RSI_event_analysis(ticker_symbol, event)
    except Exception as e:
        print(f"Error occurred during RSI plotting: {e}")
    #畫布林通道
    try: 
        plot_bollinger_event_analysis(ticker_symbol, event)
    except Exception as e:
        print(f"Error occurred during bollinger plotting: {e}")
    #畫MACD
    try:  
        plot_MACD_event_analysis(ticker_symbol, event)
    except Exception as e:
        print(f"Error occurred during MACD plotting: {e}")
    #畫損益表
    try:
        plot_financials(ticker_symbol)
    except Exception as e:
        print(f"Error occurred during financials plotting: {e}")
    #畫月均價跟損益表
    try:
        plot_accounting_and_stock('OperatingIncome', ticker_symbol)
    except Exception as e:
        print(f"Error occurred during accounting and stock plotting: {e}")
    try:
        plot_accounting_and_stock('GrossProfit', ticker_symbol)
    except Exception as e:
        print(f"Error occurred during accounting and stock plotting: {e}")
    #畫財務比率
    try:
        plot_stock_valuation(ticker_symbol)
    except Exception as e:
        print(f"Error occurred during stock valuation plotting: {e}")
    #因果分析(進chain)
    #先處理csv
    # 股價歷史數據
    try:
        event_date = pd.to_datetime(event[1]).date()
        start_date = event_date - pd.DateOffset(months=3)
        end_date_for_casual = event_date + pd.DateOffset(months=6)
    except Exception as e:
        end_date_for_casual = pd.Timestamp.today().date()
        start_date = end_date_for_casual - timedelta(days=365)
    df_casual = ticker.history(start=start_date, end=end_date_for_casual)
    df_casual = df_casual.reset_index()
    df_casual['Date'] = pd.to_datetime(df_casual['Date']).dt.date
    #取出日期跟價格
    df_casual_close = df_casual[['Date', 'Close']]
    #加入月報酬率
    df_casual_close['monthly return'] = df_casual_close['Close'].pct_change(20)
    #去除na值
    df_casual_close = df_casual_close.dropna()
    csv_file_prefix = f"{ticker_symbol}_financial_"
    data_csv_filename = csv_file_prefix + "stock_evolution_for_casual.csv"
    df_casual_close.to_csv(data_csv_filename, index=False)
    
    #csvLoader
    from langchain.document_loaders import CSVLoader
    loader = CSVLoader(data_csv_filename)
    documents = loader.load()
    
    #加上因果分析結果
    casual_result = f'事件名稱:{event[0]}\n 事件發生時間:{event[1]}\n 對該事件因果分析:{event_casual}'
    #create document
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""])
    final_casual_result = text_splitter.create_documents([casual_result])
    final = final_casual_result + documents 
    #進chain
    llm = ChatOpenAI(temperature  =0.5, model_name = "gpt-3.5-turbo-16k")
    prompt_template = """
    您是一位專業的金融專家，我將為您提供 '公司股價數據（收盤價，每月回報）'、'事件發生情況' 和 '事件的因果關係'。請幫助我通過整合 '股價數據'、
    '事件發生情況' 和 '事件的因果關係' 來分析股價趨勢，並以書面描述的方式呈現。這份分析應該包括對事件發生時股價趨勢的詳細描述和檢驗，
    以及事件發生前三個月和事件發生後六個月內的股價趨勢。請確保您的論點具有邏輯性。
    ---------
    {text}
    ---------
    請協助我進行分析，通過將「股價數據」、「事件發生」以及「事件的因果關係」整合在一份書面描述中，來分析股價趨勢。請確保你的論點是合乎邏輯的，
    並且你不需要寫下數值數據。
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = LLMChain(llm=llm, prompt=prompt)
    #chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT)
    price_plus_casual_result = chain.run(final) 
    with open (f"price_plus_casual_result.txt", "w", encoding='utf-8') as f:
        f.write(price_plus_casual_result)
    
    #刪除檔案
    file_names = [data_csv_filename]
    for file_name in file_names:
        if os.path.isfile(file_name):
            os.remove(file_name)
            print(f"檔案 {file_name} 已被刪除")
        else:
            print(f"檔案 {file_name} 不存在或不是一個檔案")
    return price_plus_casual_result
    
#抓財報
def get_financial_statements(ticker):
    from yahooquery import Ticker
    company = Ticker(ticker)
    
    balance_sheet = company.balance_sheet()
    cash_flow = company.cash_flow()
    income_statement = company.income_statement()
    valuation_measures = company.valuation_measures
    
    # 設定檔案名稱
    csv_file_prefix = f"{ticker}_financial_"
    
    # 抓取股價資料
    data = yf.download(ticker, period='5y', interval='1d')
    
    # 轉換日期時間為不帶時區的格式
    data.index = data.index.tz_localize(None)
    
    # 儲存股價資料到 CSV 檔案
    data_csv_filename = csv_file_prefix + "stock_evolution.csv"
    data.to_csv(data_csv_filename)
    
    # 儲存財務報表資料到各個 CSV 檔案
    balance_sheet_csv_filename = csv_file_prefix + "balance_sheet.csv"
    cash_flow_csv_filename = csv_file_prefix + "cash_flow.csv"
    income_statement_csv_filename = csv_file_prefix + "income_statement.csv"
    valuation_measures_csv_filename = csv_file_prefix + "valuation_measures.csv"
    
    balance_sheet.to_csv(balance_sheet_csv_filename)
    cash_flow.to_csv(cash_flow_csv_filename)
    income_statement.to_csv(income_statement_csv_filename)
    valuation_measures.to_csv(valuation_measures_csv_filename)
    
    print('股價資料和財務報表已儲存為 CSV 檔案')
    return data_csv_filename, balance_sheet_csv_filename, cash_flow_csv_filename, income_statement_csv_filename, valuation_measures_csv_filename


#主函數
#定義tools的集合
def get_my_agent(event_casual, event, company_name):
    llm = ChatOpenAI(temperature  =0.5, model_name = "gpt-3.5-turbo-16k")
    embeddings = OpenAIEmbeddings()
    
    #財務分析工具
    class Financial_analysis_Input(BaseModel):
        """Inputs for get_stock_performance"""
        ticker: str = Field(description="Ticker symbol of the stock")
    class Financial_analysis_Tool(BaseTool):
        name = "get_stock_financial_analysis"
        description = """
            Use this tool when you want to understand the past performance of a stock.
            You should enter the stock ticker symbol recognized by the yahoo finance.
            """
        args_schema: Type[BaseModel] = Financial_analysis_Input
        def _run(self, ticker: str):
            llm = ChatOpenAI( temperature  =0.5, model_name = "gpt-3.5-turbo-16k")
            #抓財務數據
            data_csv_filename, balance_sheet_csv_filename, cash_flow_csv_filename, income_statement_csv_filename, valuation_measures_csv_filename = get_financial_statements(ticker)
            #最近新聞
            title_news, link_news, source, date = get_company_news(company_name, 3)
            news_data = {
                'title': title_news,
                'link': link_news,
            }
            news_data['title'].append('財務資料參考來源')
            news_data['link'].append('https://finance.yahoo.com/quote/{ticker}/financials?p={ticker}'.format(ticker=ticker))
            with open ('reference.txt', 'w') as f:
                f.write(json.dumps(news_data))
            
            documents = []
            for link in link_news:
                try:
                    loader = WebBaseLoader(link)
                    document = loader.load()
                    documents += document
                except Exception as e:
                    print(f"Error loading document: {e}")
                    continue
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50, separators=["\n\n", "\n", "\t", " ", ""]) 
            split_doce = text_splitter.split_documents(documents)
            split_doce = remove_unnessary_word(split_doce)
            if len(split_doce) > 10:
                split_doce = split_doce[:10]
            prompt_template = """請保持文字總數在400字以下。身為一名經濟學家和金融專家的你，我正在尋求協助來組織和編譯一份關於特定股票的所有近期事件和新聞的綜合清單。
            請憑藉你的專業知識，我將提供你所需的最新消息，並請求您的幫助來組織和總結這些信息，如果您發現一些消息似乎與特定股票無關，請忽略它。
            -----------
            #以下圍觀於特定股要得近期新聞:
            {text}
            -----------
            請以繁體中文簡明摘要:"""
            PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
            chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT)
            summary_of_recent_news = chain.run(split_doce)
            
            #資產負債表
            loader = CSVLoader(balance_sheet_csv_filename)
            documents = loader.load()
            prompt_template = """
            請將字數限制在400字以內。作為經驗豐富的會計師和金融專家，我希望請您評估過去幾年特定股票的資產負債表。我希望您能利用您豐富的會計知識，
            徹底分析資產負債表的內容。在您的回應中，請對基於資產負債表中的資訊，提供一個詳細的公司優勢和劣勢評估。此外，
            請辨識可能出現的未來潛在疑慮。(請勿重複我的提示)
            -----------
            #以下為資產負債表數據
            {text}
            -----------
            請以繁體中文簡明摘要:"""
            PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
            chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT)
            summary_balance_sheet = chain.run(documents)
            
            #損益表
            loader = CSVLoader(income_statement_csv_filename)
            documents = loader.load()
            prompt_template = """請務必將字數限制在400字以內。作為一名經驗豐富的會計師和金融專家，我希望請求您對某特定股票過去幾年的損益表進行評估。
            我謹此懇請您以您豐富的會計知識和專業技能，對損益表的內容進行深入分析。請全面評估該公司的優勢和劣勢，並基於損益表突顯可能的未來風險。
            此外，在引用具體數據時，為確保清晰和準確，請以占位符的形式（%〜%）呈現。
            ---------------
            #以下為損益表數據
            {text}
            ---------------

            請以繁體中文簡明摘要:"""
            PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
            chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT)
            summary_income_statement = chain.run(documents)
            
            ##財務比率
            loader = CSVLoader(valuation_measures_csv_filename)
            documents = loader.load()
            prompt_template = """請保持字數在400字以下。作為一名專業的會計師和金融專家，我將為您提供過去幾年股票的財務比率和一些指標。請根據這些指標
            的內容以及您數據庫中的廣泛會計知識評估該股票。請盡量詳細介紹該公司的優勢和劣勢，以及可能的未來風險。
            ---------------
            #以下為財務比率數據
            {text}
            ---------------
            請以繁體中文簡明摘要:"""
            PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
            chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT)
            summary_valuation_measures = chain.run(documents)
            
            
            #抓歷史事件和畫圖
            print(event)
            price_plus_casual_result = plot_stock_event_casual_analysis(ticker, event, event_casual)
            #總結
            final_data = f"股票代號: {company_name}\n\n近期新聞: {summary_of_recent_news} 財務比率: {summary_valuation_measures} \n\n 資產負債表: {summary_balance_sheet} \n\n 損益表: {summary_income_statement}"
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""])
            final_document = text_splitter.create_documents([final_data])
            
            #投資建議
            
            prompt_template = """請提供一篇約500字的股票分析。我將提供您有關特定公司的信息。請從該公司的股票代碼以及該公司所屬行業的簡要介紹和最近的新聞摘要以及盈利來源的優勢和劣勢開始。
            ---
            #有關特定公司的信息
            {text}
            ---
            請提供詳細的財務分析，約500字，使用繁體中文進行:"""
            PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
            chain = LLMChain(llm=llm, prompt=PROMPT)
            Investment_Recommendations = chain.run(final_document) 
            with open('investment.txt', 'w', encoding='utf-8') as file:
                file.write(f"彙總報告: {Investment_Recommendations}\n")  
                
            #儲存財報資料
            with open('data.txt', 'w', encoding='utf-8') as file:
                file.write(f"股票代號: {ticker}\n")
                file.write(f"近期新聞: {summary_of_recent_news}\n")
                file.write(f"財務比率: {summary_valuation_measures}\n")
                file.write(f"資產負債表: {summary_balance_sheet}\n")
                file.write(f"損益表: {summary_income_statement}\n")
                file.write(f"股價變化分析: {price_plus_casual_result}\n")
                
                                
            
            
            #刪除檔案
            file_names = [data_csv_filename, balance_sheet_csv_filename, cash_flow_csv_filename, income_statement_csv_filename, valuation_measures_csv_filename]
            for file_name in file_names:
                if os.path.isfile(file_name):
                    os.remove(file_name)
                    print(f"檔案 {file_name} 已被刪除")
                else:
                    print(f"檔案 {file_name} 不存在或不是一個檔案")
            
            
            return Investment_Recommendations   
        def _arun(self, ticker: str):
            raise NotImplementedError("get_stock_performance does not support async")
    
    customize_tools = [
        Financial_analysis_Tool()
    ]
    def _handle_error(error) -> str:
       return str(error)[:150]
    
    PREFIX = '''In order to obtain a precise, comprehensive, and high-quality response, it is requested that you answer the following prompt in Traditional Chinese. 
    Please refrain from making any changes to the listed results if they have already been presented in bullet point format. Instead, provide your response using the original answer as it is.'''
    SUFFIX = '''In order to obtain a precise, comprehensive, and high-quality response, it is requested that you answer the following prompt in Traditional Chinese. 
    Please re agent_prompt_format_instructionfrain from making any changes to the listed results if they have already been presented in bullet point format. Instead, provide your response using the original answer as it is.'''
    
    
    
    
    
    my_agent = initialize_agent(
        tools = customize_tools, 
        llm = llm, 
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, 
        verbose=True,
        agent_kwargs={
        'prefix':PREFIX,
        'suffix':SUFFIX
        }
        )
    return my_agent


def point_of_view_analysis(new_result, processed_result, event, user_input_hint):
    llm = ChatOpenAI(temperature  =0.5, model_name = "gpt-3.5-turbo-16k")
    input = f"事件名稱:{event[0]}\n\n事件發生日期:{event[1]}\n\n事件總整理:{new_result}\n\n事件因果分析:{processed_result}"
    print(input)
    prompt_template = """
    Here is a context:
    {input}
    -----------------------
    你現在是一名{user_input_hint}高級主管，請從你的角度出發，以清晰、結構化和詳盡的方式分析以下情況：這件事件可能對您所在的產業產生什麼影響？同時，請給出對這個事件未來發展的預測。
    請以繁體中文回答
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["input", "user_input_hint"])
    chain = LLMChain(llm=llm, prompt=prompt)
    point_of_view_analysis = chain.run(input = input, user_input_hint = user_input_hint)
    return point_of_view_analysis



def cut_result(result):
    background_marker = "原因:\n"
    alternative_background_marker = "原因：\n"
    if background_marker in result:
        background, remaining = result.split(background_marker, 1)
    else:
        background, remaining = result.split(alternative_background_marker, 1)   
    effect_marker = "未來影響:\n"
    effect_marker2 = "未來影響：\n"
    effect_marker3 = "未来影响：\n"
    effect_marker4 = "未来影响:\n"
    if effect_marker in remaining:
        cause, effect = remaining.split(effect_marker, 1)
    elif effect_marker2 in remaining:
        cause, effect = remaining.split(effect_marker2, 1)
    elif effect_marker3 in remaining:
        cause, effect = remaining.split(effect_marker3, 1)
    elif effect_marker4 in remaining:
        cause, effect = remaining.split(effect_marker4, 1)
    else:
        cause, effect = None, None
    pointer_maker = "我的觀點:\n"
    pointer_maker2 = "我的观点:\n"
    pointer_maker3 = "我的观點：\n"
    pointer_maker4 = "我的觀點：\n"
    pointer_maker5 = "我的觀點:"
    
    if pointer_maker in effect:
        effect, pointer = effect.split(pointer_maker, 1)
    elif pointer_maker2 in effect:
        effect, pointer = effect.split(pointer_maker2, 1)
    elif pointer_maker3 in effect:
        effect, pointer = effect.split(pointer_maker3, 1)
    elif pointer_maker4 in effect:
        effect, pointer = effect.split(pointer_maker4, 1)
    elif pointer_maker5 in effect:
        effect, pointer = effect.split(pointer_maker5, 1)
    else:
        pointer = None
    
    return background, cause, effect, pointer
