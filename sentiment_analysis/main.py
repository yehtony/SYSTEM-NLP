from senti_c import SentenceSentimentClassification

sentence_classifier = SentenceSentimentClassification(logging_level = "warning")
test_data = ["我覺得聲音在空氣中比水快",
             "你連這個都不懂？",
             "我們可以試試在水中做實驗",]
result = sentence_classifier.predict(test_data, run_split = True, aggregate_strategy = False)
print(result.iloc[:, 1:])