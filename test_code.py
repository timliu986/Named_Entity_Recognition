import os
import sys
sys.path.append(os.getcwd()+'./version4')
os.chdir('./version4')

from model import ner_model


model = ner_model(number_class=13)
model.load_model('1loss')

model.idx2label = {
                 0:'O',
                 1:'B-TREATMENT',
                 2:'I-TREATMENT',
                 3:'B-BODY',
                 4:'I-BODY',
                 5:'B-SIGNS',
                 6:'I-SIGNS',
                 7:'B-CHECK',
                 8:'I-CHECK',
                 9:'B-DISEASE',
                 10:'I-DISEASE',
                 11:'O',
                 12:'O'
                }

while True:
    sentence = input('sentence : ')
    if sentence == 'quit':
        break
    d = model.test_for_input(sentence)
    print(d)

import numpy as np
sentence = "r-麩胺酸轉化酶(r-GT)，請問 r-麩胺酸轉化酶(r-GT) 檢查結果是6  , 低於正常值代表什麼意思? 需要注意什麼? "
d,df = model.test_for_input(sentence)



from transformers import  BertTokenizer

tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
tokenizer.tokenize('健檢[UNK][UNK][UNK]為6.85，FreeT4為1.05健檢TSH為6.85，FreeT4為1.05')

import re
sentence = "r-麩胺酸轉化酶(r-GT)，請問 r-麩胺酸轉化酶(r-GT) 檢查結果是6 ， 低於正常值代表什麼意思? 需要注意什麼? "
sentence = re.sub("[a-zA-Z]", "[UNK]", sentence)
sentence = re.sub(r'[0-9]',"[UNK]" ,sentence)
sentence = re.sub("\\s+", "", sentence)
sentence = sentence.replace(".", "[UNK]")
# 刪除多餘換行
sentence = re.sub("\r*\n*", "", sentence)
# 刪除括號內英文
