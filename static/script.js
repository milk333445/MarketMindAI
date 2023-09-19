//新增部分
var config = {
    'limit': 10,
    'languages': ['zh', 'en'], /*https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes*/
    'maxDescChars': 100,
  };
  
  KGSearchWidget(key="", document.getElementById("user_input_event"), config);
  KGSearchWidget(key="", document.getElementById("user_input_company"), config);

var modal = document.getElementById('modal');
var bgImg = document.getElementById('bgImg');
function showBgImg(e) {
    document.getElementById('modal').style.display = 'block';
    document.getElementById('bgImg').src = e.src;
}

modal.onclick = function() {
    modal.style.display = 'none';
}



var loading = false;

function setLoading(value) {
    loading = value;
    var loadingContainer = document.getElementById('loadingContainer');
    if (loading) {
        loadingContainer.classList.remove('hidden');
    } else {
        loadingContainer.classList.add('hidden');
    }
}
$(document).ready(function() {
    $('#input-form').submit(function(e) {
        e.preventDefault();
        setLoading(true);
        var userInput = $('#user_input_event').val();
        var company_name = $('#user_input_company').val();
        var user_input_hint = $('#user_input_hint').val();

        
        
        console.log('User Input:', userInput);
        console.log('Company Name:', company_name);
        console.log('User Input Hint:', user_input_hint);
        var requestData = {
            'user_input_event': userInput,
            'user_input_company': company_name,
            'user_input_hint': user_input_hint,
            'user_input_history_keyword': '這是測試',
        }; 


        //彈出視窗
        function showModal(eventName, eventTime) {
            var modal = document.getElementById('confirmationModal');
            var modalText = document.getElementById('modalText');
            var re_modal = document.getElementById('regenertationModal');

            modalText.innerHTML = '事件名稱：' + eventName + '<br>時間：' + eventTime + '<br>是否繼續或重新搜尋？';
            modal.style.display = 'flex';
            
            var confirmButton = document.getElementById('confirmButton');
            var cancelButton = document.getElementById('cancelButton');
            
            confirmButton.onclick = function() {
                modal.style.display = 'none';
                setLoading(true);
                processHistoryRequest();
            };
            
            cancelButton.onclick = function() {
                modal.style.display = 'none';
                re_modal.style.display = 'flex';

                var regenerateForm = document.getElementById('regenerate_form');
                var userInputEvent2 = document.getElementById('user_input_event_2');
                regenerateForm.addEventListener('submit', function(e) {
                    e.preventDefault();

                    var newUserInput = userInputEvent2.value;
                    requestData.user_input_event = newUserInput;
                    re_modal.style.display = 'none';
                    setLoading(true);
                    processFirstRequest();

                });
            };
            
        }
        //處理歷史觀點搜尋
        function showHistoryModal(history_keyword) {
            var his_modal = document.getElementById('his_confirmationModal');
            var his_confirmButton = document.getElementById('his_confirmButton');
            var his_adjust = document.getElementById('his_adjust');
            var his_adjustInput = document.querySelector('#his_adjust input');
            var historyKeywordContainer = document.getElementById('historyKeywordContainer'); 

            //設定彈出框
            var modalText = his_modal.querySelector('.modal-content p');
            modalText.textContent = '請確認是否要以此關鍵字去搜尋歷史事件';
            his_modal.style.display = 'flex';
            historyKeywordContainer.innerHTML = '歷史事件關鍵字：<span id="historyKeyword">' + history_keyword + '</span> ' +
                                      '<button id="his_adjustment" onclick="adjustment()" style="border: none; background: none; font-size: 10px; text-decoration: underline;">修改</button><br>';

            var his_adjustButton = document.getElementById('his_adjustment');

            his_confirmButton.onclick = function() {

                var newKeyword = his_adjustInput.value.trim();
                console.log('newKeyword:', newKeyword);
                if (newKeyword !== '')
                {
                    requestData.user_input_history_keyword = newKeyword;
                    console.log('requestData.user_input_history_keyword:', requestData.user_input_history_keyword);
                }
                else
                {
                    requestData.user_input_history_keyword = history_keyword;
                }
                his_modal.style.display = 'none';
                processSecondRequest();
            };
            
        }




        //處理歷史觀點搜尋
        function processHistoryRequest()
        {
            setLoading(true)
            $.ajax({

                url: '/get_history_keyword',
                type: 'POST',
                data: JSON.stringify(requestData),
                contentType: 'application/json',
                success: function(response) {
                    console.log('Response:', response);
                    console.log('History Keyword:', response.history_keyword);
                    var history_keyword = response.history_keyword;
                    //彈出視窗
                    setLoading(false);
                    showHistoryModal(history_keyword);
            

                


                    
                }

            });
        }



        //第一個請求
        function processFirstRequest()
        {
            $.ajax({
                url: '/get_event',
                type: 'POST',
                data: JSON.stringify(requestData),
                contentType: 'application/json',
                success: function(response) {
                    console.log('Response:', response);
                    console.log('Events:', response.result);
                    //插入表頭
                    var eventName = response.eventname;
                    var eventTime = response.eventtime;
                    console.log('eventName:', eventName);
                    console.log('eventTime:', eventTime);

                    //彈出視窗
                    setLoading(false);
                    showModal(eventName, eventTime); 
                    }});
        }

        function bindFormSubmitEvent() {
            $(document).on('submit', '#generating-form', function(e) {
                e.preventDefault();
                var userInput = $('#user_input_event').val();
                var company_name = $('#user_input_company').val();
                var user_input_hint = $('#user_input_hint').val();
        
                if (userInput.trim() === '') {
                    alert('請輸入事件名稱');
                    return;
                }
                setLoading(true);
                console.log('Generating');
                console.log('User Input:', userInput);
                console.log('Company Name:', company_name);
                console.log('User Input Hint:', user_input_hint);
                var requestData = {
                    'user_input_event': userInput,
                    'user_input_company': company_name,
                    'user_input_hint': user_input_hint
                }; 
        
                // 請求後端重新產生圖片
                $('#graph').html('Loading...');
                $.ajax({
                    url: '/regenerate',
                    type: 'POST',
                    data: JSON.stringify(requestData),
                    contentType: 'application/json',
                    success: function(response) {
                        console.log('Response:', response);
                        console.log('confirm:', response.confirm);
                        var randomParam = Date.now(); // 获取当前时间戳作为随机参数
                        var imgTag = '<img class="thum-img" src="../static/my_graph.svg?' + randomParam + '" onclick="showBgImg(this)" style="width: 100%; cursor: zoom-in;"/>';

                        //var imgTag = '<img class="thum-img" src="../static/my_graph.svg" onclick="showBgImg(this)" style="width: 100%; cursor: zoom-in;"/>';
                        setTimeout(function() {
                            // 替换为新图片
                            $('#graph').html(imgTag);
                        }, 1000);
                        //$('#graph').html(imgTag);
                        setLoading(false);
                    }
                });
            });
        }

        function processSecondRequest(){
            setLoading(true);
            $.ajax({
                url: '/process',
                type: 'POST',
                data: JSON.stringify(requestData),
                contentType: 'application/json',
                success: function(response) {
                    console.log('Response:', response);
                    console.log('Related History Events:', response.related_history_events);
                    //插入表頭
                    var processedResult = response.result;
                        console.log('processedResult:', processedResult);
                    
    
                    //處理結果
                    if (response.background === null || response.cause === null || response.effect === null || response.pointer === null)
                    {
                        var processedResult = response.result;
                        console.log('processedResult:', processedResult);
                        var news = response.news;
                        var newswithbreaks = news.replace(/\n/g,'<br>');
                        var processedResultWithLineBreaks = processedResult.replace(/\n\n/g, '<br><br>').replace('/\n/g', '<br><br>').replace(/- /g, '<br>');
                        var imgTag = '<img class="thum-img" src="../static/my_graph.svg" onclick="showBgImg(this)" style="width: 100%; cursor: zoom-in;"/>';
                        var html = `
                        <div id="event_detail" class="contain_css">
                        <h5>事件名稱: ${userInput} 公司: ${company_name} 觀點: ${user_input_hint}</h5>
                        </div>
                        <h4">回答</h4>
                        <div id="result">${processedResultWithLineBreaks}</div>
                        <div id="graph">
                            ${imgTag}
                        </div>
                        `;
                        $('#div_casual_answer').html(html);
                        $('#news').html(newswithbreaks);
                    }
                    else
                    {
                        var background = response.background;
                        var backgroundWithLineBreaks = background.replace(/- /g, ''); 
                        console.log('background:', background);
                        var cause = response.cause;
                        var secondDashIndex = cause.indexOf('- ', cause.indexOf('-') + 1);
                        var causeWithLineBreaks = cause.substring(0, secondDashIndex) + cause.substring(secondDashIndex).replace(/- /g, '<br><br>');
                        var causeWithLineBreaks = causeWithLineBreaks.replace(/- /g, '');
                        console.log('cause:', cause);
                        var effect = response.effect;
                        var secondDashIndex = effect.indexOf('- ', effect.indexOf('-') + 1);
                        var effectWithLineBreaks = effect.substring(0, secondDashIndex) + effect.substring(secondDashIndex).replace(/- /g, '<br><br>');
                        var effectWithLineBreaks = effectWithLineBreaks.replace(/- /g, '');
                        console.log('effect:', effect);
                        var pointer = response.pointer;
                        var news = response.news;
                        var newswithbreaks = news.replace(/\n/g,'<br>');




                        console.log('backgroundWithLineBreaks:', backgroundWithLineBreaks);
                        console.log('causeWithLineBreaks:', causeWithLineBreaks);
                        console.log('effectWithLineBreaks:', effectWithLineBreaks);
                        var imgTag = '<img class="thum-img" src="../static/my_graph.svg" onclick="showBgImg(this)" style="width: 100%; cursor: zoom-in;"/>';
                        var html = `
                        <div id="event_detail" class="contain_css">
                        <h5>事件名稱: ${userInput} 公司: ${company_name} 觀點: ${user_input_hint}</h5>
                        </div>
                        <h4 class="title-heading">背景介紹</h4>
                        <div id="casual_background" class="contain_css">
                            <p>${backgroundWithLineBreaks}</p>
                        </div>
                        <h4 class="title-heading">因果關聯圖</h4>
                            <form id="generating-form" style="float: right;">
                                <button style="border: none;" type="submit" ><i class="fa-solid fa-rotate-right" style="color: #000000;"></i></button>
                            </form>
                            <div id="graph">
                                ${imgTag}
                            </div>
                        <div>
                            <div style="width:49%; float:left;">
                            <h4 class="title-heading">事件原因</h4>
                                <div id="casual_reason" class="contain_css" >
                                    <p>${causeWithLineBreaks}</p>
                            </div>
                            </div>
                            <div style="width:49%; float:right;">
                            <h4 class="title-heading">未來影響</h4>
                                <div id="casual_impact" class="contain_css" >
                                    <p>${effectWithLineBreaks}</p>
                                </div>
                            </div>
                        </div>
                        `;
                        $('#div_casual_answer').html(html);
                        $('#point').html(pointer);
                        $('#news').html(newswithbreaks);

                        bindFormSubmitEvent()
                    }
    
    
    
    
                    /*
                    var processedResult = response.result;
                    var processedResultWithLineBreaks = processedResult.replace(/\n\n/g, '<br><br>')
                    $('#result').html(processedResultWithLineBreaks);
                    */
    
                    //處理觀點分析
                    var pointresult = response.point_of_view_analysis;
                    console.log('pointresult:', pointresult);
                    if (pointresult !== null)
                    {
                        var pointresultWithLineBreaks = pointresult.replace(/\n\n/g, '<br><br>').replace('/\n/g', '<br><br>');
                        console.log('pointresultWithLineBreaks:', pointresultWithLineBreaks);
                        $('#casual_point').html(pointresultWithLineBreaks);
                        
                    }
                    else
                    {
                        $('#casual_point').html('');
    
                    }
    
    
    
                    var investmentResult = response.investment_result;
                    if (investmentResult !== null)
                    {
                        var investmentResultWithLineBreaks = investmentResult.replace(/\n\n/g, '<br><br>').replace('/\n/g', '<br><br>');
                        $('#invest_result').html('分析結果：<br>' + investmentResultWithLineBreaks);
                    }
                    else
                    {
                        $('#invest_result').html('');
                    }
                    //處理股價分析
                    var priceanalysisResult = response.price_plus_casual_result;
                    console.log('priceanalysisResult:', priceanalysisResult);
                    if (priceanalysisResult !== null)
                    {
                        var query = `<h5>事件名稱: ${userInput} 公司: ${company_name} 觀點: ${user_input_hint}</h5>`
                        $('#event_detail_stock_analysis').html(query)
    
                        var priceanalysisResultWithLineBreaks = priceanalysisResult.replace(/\n\n/g, '<br><br>');
                        console.log('priceanalysisResultWithLineBreaks:', priceanalysisResultWithLineBreaks);
                        $('#event_effect').html(priceanalysisResultWithLineBreaks);
                    }
                    else
                    {
                        $('#event_effect').html('');
                    }
    
                    //處理歷史相似事件
                    var relatedhistoryevents = response.related_history_events  
                    var eventsObj = JSON.parse(relatedhistoryevents);
                    console.log('Related History Events:', eventsObj);
    
                    var eventsDiv = document.getElementById("timeline");
                    //獲取長度
                    var nameEventList = eventsObj['類似事件名稱'];
                    var lengthOfNameEventList = nameEventList.length;
    
                    for (var i=0; i<lengthOfNameEventList; i++)
                    {
                        var timelineGroup = document.createElement('div');
                        timelineGroup.className = 'timeline_group';
    
                        var timelineYear = document.createElement('span');
                        timelineYear.className = 'timeline_year time';
                        timelineYear.textContent = eventsObj['類似事件年'][i];
                        timelineYear.setAttribute('aria-hidden', 'true');
                        timelineGroup.appendChild(timelineYear);
    
                        var timelineCards = document.createElement('div');
                        timelineCards.className = 'timeline_cards';
                        timelineGroup.appendChild(timelineCards);
    
                        var timelineCard = document.createElement('div');
                        timelineCard.className = 'timeline_card card';
                        timelineCards.appendChild(timelineCard);
    
                        var cardHead = document.createElement('header');
                        cardHead.className = 'card_head';
                        timelineCard.appendChild(cardHead);
    
                        
                        var time = document.createElement('time');
                        time.className = 'time';
                        cardHead.appendChild(time);
                        
                        /*
                        var timeDay = document.createElement('span');
                        timeDay.className = 'time_day';
                        timeDay.textContent = eventsObj['類似事件日'][i] + '日'; 
                        time.appendChild(timeDay);
                        */
                        
                        var timeMonth = document.createElement('span');
                        timeMonth.className = 'time_month';
                        timeMonth.textContent = eventsObj['類似事件起始結束日期'][i];
                        time.appendChild(timeMonth);
                        
    
                        var cardTitle = document.createElement('h3');
                        cardTitle.className = 'card_title r-title';
                        cardTitle.textContent = eventsObj['類似事件名稱'][i];
                        cardHead.appendChild(cardTitle);
    
                        var cardContent = document.createElement('div');
                        cardContent.className = 'card_content';
                        timelineCard.appendChild(cardContent);
    
                        var descP = document.createElement('p');
                        descP.textContent = '類似事件描述: ' + eventsObj['類似事件描述'][i];
                        cardContent.appendChild(descP);
    
                        var causeP = document.createElement('p');
                        causeP.textContent = '類似事件因果分析: ' + eventsObj['類似事件因果分析'][i];
                        cardContent.appendChild(causeP);
    
                        var econP = document.createElement('p');
                        econP.textContent = '類似事件期間的經濟狀況: ' + eventsObj['類似事件期間的經濟狀況'][i];
                        cardContent.appendChild(econP);
    
                        eventsDiv.appendChild(timelineGroup);
                    }
                    
                    
                    if (company_name.trim() !== '')
                    {
                        
                        var imageSources = [
                            '../static/PRICE.png',
                            '../static/RSI.png',
                            '../static/bollinger.png',
                            '../static/MACD.png',
                            '../static/stock_valuation2.png',
                            '../static/accounting_and_stock.png',
                            '../static/financials.png',
                            '../static/stock_valuation1.png',
                        ];
                        
                        var titles = [
                            'graph_event_effect',
                            'graph_RSI',
                            'graph_bands',
                            'graph_MACD',
                            'graph_value',
                            'graph_gprofitclose',
                            'graph_keyfinancial',
                            'graph_ratio'
                        ];
                        
                        for (var i = 0; i < imageSources.length; i++) {
                            (function(index) {
                                var imageSource = imageSources[index];
                                var title = titles[index];
                        
                                var img = new Image();
                                img.src = imageSource;
                        
                                img.onload = function() {
                                    var html = `<img src="${imageSource}" alt="${title}" style="max-width: 100%; max-height: 100%;">`;
                                    console.log('Image loaded:', this.src);
                                    $('#' + title).html(html);
                                };
                        
                                img.onerror = function() {
                                    console.log('Image not found:', this.src);
                                };
                            })(i);
                        }
    
                    }
                    else
                    {
                        $('#graph_event_effect').html('');
                        $('#graph_RSI').html('');
                        $('#graph_bands').html('');
                        $('#graph_MACD').html('');
                        $('#graph_value').html('');
                        $('#graph_gprofitclose').html('');
                        $('#graph_keyfinancial').html('');
                        $('#graph_ratio').html('');
                    }
    
                    //處理reference
                    var reference = response.reference_data;
                    if (reference !== null)
                    {
                        var referenceObj = JSON.parse(reference);
                        var referenceDiv = document.getElementById('reference');
                        for (var i=0; i < referenceObj['title'].length; i++)
                        {
                            var title = referenceObj['title'][i];
                            var link = referenceObj['link'][i];
    
                            var paragraph = document.createElement('p');
                            var anchor = document.createElement('a');
    
                            anchor.href = link;
                            anchor.textContent = title;
                            paragraph.appendChild(anchor);
                            referenceDiv.appendChild(paragraph);
                        }
                    }
                    else
                    {
                        $('#reference').html('');
                    }
                    var newsreference = response.news_reference;
                    var newsreferenceObj = JSON.parse(newsreference);
                    var ref_button = document.getElementById('ref_button');
                    ref_button.onclick = function(){
                        var newsreferenceData = JSON.stringify(newsreferenceObj);
                        var url = "news_reference.html?data=" + encodeURIComponent(newsreferenceData);
                        window.open(url, '_blank') ;
                    }
                    setLoading(false);
      
         
                },
                error: function(error) {
                    console.error('Error:', error);
                    setLoading(false);
                }
            });
        }
        processFirstRequest();

        
    }
    
    
    
    
    );


    /*
    $('#generating-form').submit(function(e) {
        e.preventDefault();
        
        var userInput = $('#user_input_event').val();
        var company_name = $('#user_input_company').val();
        var user_input_hint = $('#user_input_hint').val();
        var requestData = {
            'user_input_event': userInput,
            'user_input_company': company_name,
            'user_input_hint': user_input_hint
        };
        

        if (userInput.trim() === '')
        {
            alert('請輸入事件名稱');
            return;
        }
        setLoading(true);
        console.log('Generating');
        console.log('User Input:', userInput);
        console.log('Company Name:', company_name);
        console.log('User Input Hint:', user_input_hint);
        

        $.ajax({
            url: '/regenerate',
            type: 'POST',
            data: JSON.stringify(requestData),
            contentType: 'application/json',
            success: function(response) {
                console.log('Response:', response);
                console.log('confirm:', response.confirm);
                var imgTag = '<img class="thum-img" src="../static/my_graph.svg" onclick="showBgImg(this)" style="width: 100%; cursor: zoom-in;"/>';
                $('#graph').html(imgTag);
                setLoading(false);

              
                }});




    }
  
    );
    */
});

var ref_button = document.getElementById('ref_button');
ref_button.onclick = function(){
    window.open("news_reference.html", '_blank') ;
}

