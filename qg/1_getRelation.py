'''主要函数是getKnowledge
    1. 获取句子中所有单词
    2. 过滤出有实际意义的单词
    3. 获取ConceptNet结果
    4. 获取WordNet结果
    5. 将ConceptNet结果和WordNet结果转换为自然语言形式
'''

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

def getLLMFilter(text, word_list):
    '''
        过滤出有实际意义的单词
    '''
    try:
        response = client.chat.completions.create(
            model="glm-4-air",
            messages=[
                {
                    "role": "user",
                    "content": f"根据<文本>和<单词集合>，过滤出有实际意义的单词。返回json格式如下：{{'entity':[xx1, xx2...]}} <文本>：{text} <单词集合>：{word_list}",
                },
            ],
            response_format={"type": "json_object"},
        )
        output = response.choices[0].message.content
        json_output = json.loads(output)
        return json_output
    except Exception as e:
        print(e)


def getConceptNet(word):
    # ConceptNet API的基本URL
    base_url = "http://api.conceptnet.io"
    # 构建查询URL
    query_url = f"{base_url}/c/en/{word}"
    # 发送GET请求
    response = requests.get(query_url)
    if response.status_code == 200:
        data = response.json()
        triples = []
        relation_counts = {"Synonymy": 0, "RelatedTo": 0, "IsA": 0}
        # 遍历edges列表
        for edge in data.get("edges", []):
            relation_label = edge["rel"]["label"]
            if (
                relation_label in ["Synonymy", "RelatedTo", "IsA"]
                and relation_counts[relation_label] < 2
            ):
                start_concept = edge["start"]["label"]
                end_concept = edge["end"]["label"]
                triple = (start_concept, relation_label, end_concept)
                triples.append(triple)
                relation_counts[relation_label] += 1

        return triples
    else:
        print(f"请求失败，状态码: {response.status_code}")
        return []


def getWordNet(word):
    synsets = wn.synsets(word)[:3]
    filtered_synsets = list(set([syn.name().split(".")[0] for syn in synsets]))
    result = []
    for syn in filtered_synsets:
        result.append((syn, "Synonymy", word))
    return result


def getNaturalForm(triples):
    try:
        response = client.chat.completions.create(
            model="glm-4-air",
            messages=[
                {
                    "role": "user",
                    "content": f"用英文将下述关系三元组转换为自然语言形式，三元组按照 <头部概念><关系><尾部概念> 提供：。返回json格式如下：{{'natural_form':[xx1, xx2...]}}.<三元组>：{triples}",
                },
            ],
            response_format={"type": "json_object"},
        )
        output = response.choices[0].message.content
        json_output = json.loads(output)
        return json_output["natural_form"]
    except Exception as e:
        print(e)
        return []


def getKnowledge(text):
    word_list = text.strip().split()
    filteredWordList = getLLMFilter(text, word_list)["entity"][:10]
    print(f"filteredWordList: {filteredWordList}")
    conceptnet_list = []
    wordnet_list = []
    natural_form_list = []
    try:
        for word in filteredWordList:
            conceptnet = getConceptNet(word)
            wordnet = getWordNet(word)
            print(f"conceptnet: {conceptnet}")
            print(f"wordnet: {wordnet}")
            conceptnet_list.extend(conceptnet)
            wordnet_list.extend(wordnet)
            natural_form = getNaturalForm(conceptnet + wordnet)
            print(f"natural_form: {natural_form}")
            natural_form_list.extend(natural_form)
        temp_dict = {
            "text": text,
            "wordList": word_list,
            "filteredWordList": filteredWordList,
            "conceptnet": conceptnet_list,
            "wordnet": wordnet_list,
            "natural_form": natural_form_list,
        }
        return temp_dict
    except Exception as e:
        print(e)


if __name__ == "__main__":
    data1 = open("/home/qhn/Codes/Projects/zz-exp/qg-zhengli/data/rawData/src-train.txt", "r").readlines()
    data2 = open("/home/qhn/Codes/Projects/zz-exp/qg-zhengli/data/rawData/tgt-train.txt", "r").readlines()
    # f = open("/home/qhn/Codes/Projects/zz-exp/qg-zhengli/data/1_getRelation/result.jsonl", "a+")
    result = []
    for idx, (line, line2) in enumerate(tqdm(list(zip(data1, data2)))):
        src_dict = getKnowledge(line.strip())
        tgt_dict = getKnowledge(line2.strip())
        temp_dict = {"idx": idx, "src": src_dict, "tgt": tgt_dict}
        result.append(temp_dict)
        print(temp_dict)
        # f.write(json.dumps(temp_dict, ensure_ascii=False) + "\n")
    # with open("/home/qhn/Codes/Projects/zz-exp/qg-zhengli/data/1_getRelation/result.json", "w") as f:
    #     json.dump(result, f, indent=4, ensure_ascii=False)
