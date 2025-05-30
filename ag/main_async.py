import copy
import json
import sys
import asyncio
from string import Template

from tqdm import tqdm
sys.path.append('')

from chat.chat import chat_qwen_api as Teacher1, chat_baichuan_api as Judger, chat_llama_api as Teacher2
from loguru import logger as log
from prompts import base_elements, addition_instruction, format_instruction, standard_prompt, second_turn_prompt, judgment_prompt

import os

current_file = os.path.basename(__file__)
current_filename = os.path.splitext(current_file)[0]
os.makedirs('./log', exist_ok=True)
is_logger = False
log_file_path = f"./log/{current_filename}.log"
log.add(
    log_file_path,
    encoding="utf-8",
    format="{level} | {time:YYYY-MM-DD HH:mm:ss} | {file} | {line} | {message}",
    rotation="500 MB",
)
TURNS = 5


async def firstTurn(base_elements):
    standar_prompt_all = standard_prompt.format(**base_elements)
    if is_logger:
        log.info(f"standar_prompt: {standar_prompt_all}")
        
    response_1_task = Teacher1(standar_prompt_all)
    response_2_task = Teacher2(standar_prompt_all)
    
    response_1 = await response_1_task
    response_2 = await response_2_task
    
    return response_1["response"], response_2["response"]

async def laterTurn(output_1, output_2, base_elements):
    is_logger = False

    teacher_1_all_output = [output_1]
    teacher_2_all_output = [output_2]
    for i in range(TURNS - 1):
        second_turn_prompt_1 = second_turn_prompt.format(Former=output_1, Reference=output_2, **base_elements)
        second_turn_prompt_2 = second_turn_prompt.format(Former=output_2, Reference=output_1, **base_elements)
        if is_logger:
            log.info(f"second_turn_prompt_1: {second_turn_prompt_1}")
            log.info(f"second_turn_prompt_2: {second_turn_prompt_2}")
        
        output_1_task = Teacher1(second_turn_prompt_1)
        output_2_task = Teacher2(second_turn_prompt_1)
        
        output_1 = (await output_1_task)["response"]
        output_2 = (await output_2_task)["response"]
        
        teacher_1_all_output.append(output_1)
        teacher_2_all_output.append(output_2)
    
    teacher_1_scores = []
    teacher_2_scores = []
    for item in teacher_1_all_output:
        try:
            json_item = json.loads(item)
            teacher_1_scores.append(int(json_item['score']))
        except:
            teacher_1_scores.append(-1)
            continue

    for item in teacher_2_all_output:
        try:
            json_item = json.loads(item)
            teacher_2_scores.append(int(json_item['score']))
        except:
            teacher_2_scores.append(-1)
            continue

    teacher_1_score_counts = {}
    teacher_2_score_counts = {}
    
    for score in teacher_1_scores:
        if score in teacher_1_score_counts:
            teacher_1_score_counts[score] += 1
        else:
            teacher_1_score_counts[score] = 1
            
    for score in teacher_2_scores:
        if score in teacher_2_score_counts:
            teacher_2_score_counts[score] += 1
        else:
            teacher_2_score_counts[score] = 1
            
    teacher_1_final_score = max(teacher_1_score_counts.items(), key=lambda x: x[1])[0]
    teacher_2_final_score = max(teacher_2_score_counts.items(), key=lambda x: x[1])[0]
    teacher_1_final_index = teacher_1_scores.index(teacher_1_final_score)
    teacher_2_final_index = teacher_2_scores.index(teacher_2_final_score)
    teacher_1_final_output = teacher_1_all_output[teacher_1_final_index]
    teacher_2_final_output = teacher_2_all_output[teacher_2_final_index]
    
    if is_logger:
        log.info(f"Teacher 1 final score: {teacher_1_final_score}, index: {teacher_1_final_index}")
        log.info(f"Teacher 2 final score: {teacher_2_final_score}, index: {teacher_2_final_index}")
        
    return {
        "teacher_1_score_counts": teacher_1_score_counts,
        "teacher_2_score_counts": teacher_2_score_counts,
        "teacher_1_all_output": teacher_1_all_output,
        "teacher_2_all_output": teacher_2_all_output,
        "teacher_1_final_output": teacher_1_final_output,
        "teacher_2_final_output": teacher_2_final_output
    }

async def judgeAll(teacher_1_final_output, teacher_2_final_output, base_elements):
    judgment_prompt_1 = judgment_prompt.format(**base_elements, Judger1=teacher_1_final_output, Judger2=teacher_2_final_output)
    final_response = await Judger(judgment_prompt_1)
    return final_response["response"]

async def judgeAndArgue(data_path):
    data = json.load(open(data_path, 'r', encoding='utf-8'))
    question, rule, score_range = data['questions'], data['rules'], data['score_range']
    result = {
        "questions": question,
        "rules": rule,
        "score_range": score_range,
        "essays_and_scores": []
    }
    f = open(os.path.join("", data_path.split("/")[-1]+"_result.jsonl"), 'a+', encoding='utf-8')
    for item in tqdm(data['essays_and_scores'], desc=f"Processing {data_path.split('/')[-1]}..."):
        student_answer, real_score = item['text'], item['score']
        result_item = copy.deepcopy(item)
        base_elements = {
            "Q": question,
            "S": score_range,
            "R": rule,
            "A": student_answer
        }
        first_response_1, first_response_2 = await firstTurn(base_elements)
        
        laterTurnResult = await laterTurn(first_response_1, first_response_2, base_elements)
        teacher_1_all_output = laterTurnResult["teacher_1_all_output"]
        teacher_2_all_output = laterTurnResult["teacher_2_all_output"]
        final_response_1 = laterTurnResult["teacher_1_final_output"]
        final_response_2 = laterTurnResult["teacher_2_final_output"]
        
        judge_response = await judgeAll(final_response_1, final_response_2, base_elements)
        result_item["teacher_1_all_output"] = teacher_1_all_output
        result_item["teacher_2_all_output"] = teacher_2_all_output
        result_item["final_response_1"] = final_response_1
        result_item["final_response_2"] = final_response_2
        result_item["judge_response"] = judge_response
        
        result["essays_and_scores"].append(result_item)
        
        json_str = json.dumps(result_item, ensure_ascii=False)
        if is_logger:
            log.info(f"first_response_1: {first_response_1}")
            log.info(f"first_response_2: {first_response_2}")
            log.info(f"final_response_1: {final_response_1}")
            log.info(f"final_response_2: {final_response_2}")
            log.info(f"judge_response: {judge_response}")

async def main():
    data_path = ''
    await judgeAndArgue(data_path)

async def main2():
    base_path = ''
    
    tasks = []
    for i in range(1, 9):
        path = os.path.join(base_path, f"essay_sets_{i}.json")
        tasks.append(judgeAndArgue(path))
    
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main2())