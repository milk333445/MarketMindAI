# casual_analysis_on_langchain

## 安裝步驟
### 下載
- 確保資料夾"因果分析agent"已成功下載到自己電腦中

### 確保資料夾中有以下資料
- app.py
- config.py
- demo.py
- static資料夾
- templates資料夾

## 事前準備
### 申請自己的serp API key
- 進入 https://serpapi.com/plan 註冊帳號
- 登入後在 Your account即可看到 Your Private API Key
### 向量資料庫申請
- 前往以下連結(右上申請)
https://www.pinecone.io/
- 註冊帳號並且註冊api
1. 註冊完之後可以到index(在pinecone官網等入後會看到)這個地方創建向量資料庫
2. 點擊右上角create index
3. 設定向量資料庫的name(後面輸入到config.py中)
4. 點開後設定維度為1536
5. 其他設定可以用預設就好
6. 創建完後去API keys得到keys跟Environment

### 填入openapi key 跟 serpapi key和pinecone的api keys跟Environment跟indexname
- 在config.py中以下填入你自己的openapi key和serpapi key
```python=
OPEN_API_KEY = ""
serpapi = ""
pinecone_api = ""
pinecone_env = ""
pinecone_index_name = ""
```


### 安裝必要套件
- 執行以下程式安裝依賴包
```python=
pip install -r requirements.txt
```
### 安裝畫圖套件
```python=
npm install -g @mermaid-js/mermaid-cli
```

### 打開終端機
- 將終端機路徑切到因果分析檔案中
```python=
cd + 因果分析agent資料夾路徑
```

### 執行
- 執行以下程式
```python=
python app.py
```

### 開啟網站
- 複製該連結貼至瀏覽器中開啟網站

### 即可開始對頁面進行操作



### 如果出現問題，可參考下面解決方法
#### 抽取函數出現問題
- 更改程式碼
- 找到demo.py 183行，應該會看到下面程式碼
```python=
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
        、、、、、
```
- 找到以下代碼片段(233~236行)
```python=
data = extraction_chain.run((doc))
    data = data.split("\n")
    data = data[1]
    return data
```
- 更改為
```python=
data = extraction_chain.run((doc))['data']
    return data
```

#### 畫圖函數出現問題
- 如果發現畫圖結果與預期不盡相同，圖片右上較可以進行重整，偶爾會有不穩定的時候
