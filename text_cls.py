# STEP1
from transformers import pipeline

# STEP2
# classifier = pipeline("sentiment-analysis", model="stevhliu/my_awesome_model")
classifier = pipeline("sentiment-analysis", model="snunlp/KR-FinBert-SC")

# STEP3
# text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."
# text = "현대바이오, '폴리탁셀' 코로나19 치료 가능성에 19% 급등"
# 비정형
# text = "샤오미의 폴더블 폰의 점유율이 삼성전자 보다 높아졌다."
text = "매출 21.7조원 역대 최대...'주력 사업과 미래 사업 균형적인 성장"

# STEP4
result = classifier(text)

# STEP5
print(result)