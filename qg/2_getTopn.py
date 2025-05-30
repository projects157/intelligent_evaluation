import nltk
from nltk.corpus import wordnet as wn
import requests
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from zhipuai import ZhipuAI

client = ZhipuAI(
    api_key="f4af3bdd28e6e99dd48d5a9eb102ee50.WND95hmeDeaxFGGE"
)  # 填写您自己的APIKey
from tqdm import tqdm
import json

data = open("/home/qhn/Codes/Projects/zz-exp/qg-zhengli/data/1_getRelation/all_data.jsonl", "r").readlines()

def getTopn(src, tgt, src_knowledge, tgt_knowledge):
    try:
        response = client.chat.completions.create(
            model="glm-4-air",
            messages=[
                {
                    "role": "user",
                    "content": f"将根据提供的<文本1>、<文本2>及其对应<语义知识>，选取与之最相关的4条语义知识,返回json格式如下:{{文本1:xxx, 文本1语义知识:yyy, 文本2:zzz, 文本2语义知识:mmm}},要求提取出的语义知识为提供的语义知识中选择的，不要自己生成\n文本1： <{src}> , 语义知识1： <{src_knowledge}> , 文本2： <{tgt}> , 语义知识2： <{tgt_knowledge}>",
                },
            ],
            response_format={"type": "json_object"},
        )
        output = response.choices[0].message.content
        json_output = json.loads(output)
        temp_dict = {
            "src": json_output["文本1"],
            "src_knowledge": json_output["文本1语义知识"],
            "tgt": json_output["文本2"],
            "tgt_knowledge": json_output["文本2语义知识"],
        }
        return temp_dict
    except Exception as e:
        print(e)
        return {}

def getTopnJson(data):
    result = []
    f = open("/home/qhn/Codes/Projects/zz-exp/qg-zhengli/data/2_getTopn/topn.jsonl", "a+")
    for line in tqdm(data):
        data = json.loads(line)
        src = data["src"]["text"]
        tgt = data["tgt"]["text"]
        src_knowledge = data["src"]["natural_form"]
        tgt_knowledge = data["tgt"]["natural_form"]
        topn = getTopn(src, tgt, src_knowledge, tgt_knowledge)
        # f.write(json.dumps(topn, ensure_ascii=False) + "\n")
        print(topn)
        result.append(topn)
    # with open("/home/qhn/Codes/Projects/zz-exp/qg-zhengli/data/2_getTopn/topn.json", "w") as f:
    #     json.dump(result, f, ensure_ascii=False)
    return result

if __name__ == "__main__":
    getTopnJson(data)
