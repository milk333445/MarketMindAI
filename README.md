# casual-ananlysis
## 安裝步驟
### 下載
- 確保資料夾"因果分析agent"已成功下載到自己電腦中

### 確保資料夾中有以下資料
- app.py
- config.py
- demo.py
- images資料夾
- static資料夾
- templates資料夾

### 填入openapi key 跟 serpapi key和pinecone的api跟env跟indexname
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
npm install -g @mermaid-js/mermaid-cli`
```

### 打開終端機
- 將終端機路徑切到因果分析檔案中
```python=
cd  因果分析agent資料夾路徑(看你存放在哪裡)
```

### 執行
- 執行以下程式
```python=
python app.py
```

### 開啟網站
- 上一步驟執行完後應該會在終端機看到以下畫面
![](https://hackmd.io/_uploads/HyMWuZ73h.png)
- 複製該連結貼至瀏覽器中開啟網站
![](https://hackmd.io/_uploads/S1nUd-7n2.png)

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
- 如果發現畫圖結果與預期不盡相同，還請見諒，偶爾會有不穩定的時候
