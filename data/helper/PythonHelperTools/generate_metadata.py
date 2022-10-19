import json
import os
import random

from vqaTools.vqa import VQA

dataDir		='../../VQA'
versionType ='v2_' # this should be '' when using VQA v2.0 dataset
taskType    ='OpenEnded' # 'OpenEnded' only for v2.0. 'OpenEnded' or 'MultipleChoice' for v1.0
dataType    ='mscoco'  # 'mscoco' only for v1.0. 'mscoco' for real and 'abstract_v002' for abstract for v1.0.
dataSubType ='train2014'
annFile     ='%s/Annotations/%s%s_%s_annotations.json'%(dataDir, versionType, dataType, dataSubType)
quesFile    ='%s/Questions/%s%s_%s_%s_questions.json'%(dataDir, versionType, taskType, dataType, dataSubType)
imgDir 		= '%s/Images/%s/%s/' %(dataDir, dataType, dataSubType)

# initialize VQA api for QA annotations
vqa_train=VQA(annFile, quesFile)

unique_questions_train = {}

for question_id in vqa_train.qa.keys():
    if vqa_train.qa[question_id]["answer_type"] in ["number","other"]:
        ques = vqa_train.qqa[question_id]['question']
        if ques not in unique_questions_train.keys():
            unique_questions_train[ques] = {"count":1,"ids":[question_id]}
        else:
            unique_questions_train[ques]["count"] +=1
            unique_questions_train[ques]["ids"].append(question_id)

dataSubType ='val2014'
annFile     ='%s/Annotations/%s%s_%s_annotations.json'%(dataDir, versionType, dataType, dataSubType)
quesFile    ='%s/Questions/%s%s_%s_%s_questions.json'%(dataDir, versionType, taskType, dataType, dataSubType)
imgDir 		= '%s/Images/%s/%s/' %(dataDir, dataType, dataSubType)

# initialize VQA api for QA annotations
vqa_val=VQA(annFile, quesFile)

unique_questions_val = {}

for question_id in vqa_val.qa.keys():
    if vqa_val.qa[question_id]["answer_type"] in ["number","other"]:
        ques = vqa_val.qqa[question_id]['question']
        if ques not in unique_questions_val.keys():
            unique_questions_val[ques] = {"count":1,"ids":[question_id]}
        else:
            unique_questions_val[ques]["count"] +=1
            unique_questions_val[ques]["ids"].append(question_id)

unique_questions_trainval = unique_questions_train
for i, item in unique_questions_val.items():
    if i in unique_questions_train:
        unique_questions_trainval[i]['count'] += item['count']
        unique_questions_trainval[i]['ids'] += item['ids']
    else:
        unique_questions_trainval[i] = {'count':item['count'], 'ids': item['ids']}

vqa_trainval_qa = {**vqa_train.qa, **vqa_val.qa}
vqa_trainval_qqa = {**vqa_train.qqa, **vqa_val.qqa}

unique_questions_trainval_selected ={}
for key, item in unique_questions_trainval.items():
    image_ans = {}
    image_list = []
    for i in item["ids"]:
        a = vqa_trainval_qa[i]['multiple_choice_answer'] # multiple_choice_answer: most frequent ground-truth answer.
        # answers[a] = answers.get(a,0) + 1
        image_id = vqa_trainval_qa[i]['image_id']
        if i in vqa_train.qa:
            image_name = 'COCO_train2014_'+ str(image_id).zfill(12) 
        elif i in vqa_val.qa:
            image_name = 'COCO_val2014_'+ str(image_id).zfill(12)
        
        if image_name not in image_list: # Avoding same image appear for the same task
            image_list.append(image_name)
            if a not in image_ans:
                image_ans[a] = [(image_name,i)]
            else:
                image_ans[a].append((image_name,i))

    # delete the answer that only appears once
    image_ans = {key:val for key, val in image_ans.items() if len(val) > 1}
    answers = {key:len(val) for key, val in image_ans.items()}
    unique_questions_trainval_selected[key] = {"count":sum(answers.values()), "answers_count": len(answers), "answers": answers ,"image_ans":image_ans}

ques_ans_count = {}
ques_image_ans = {}

count = 0
for key, item in unique_questions_trainval_selected.items():
    cond1 = item["count"] > 20          # question appear at least 20 times
    cond2 = item["answers_count"] > 1   # question contains multiple answers
    cond3 = "or" not in key.split()     # question not in "choose from" form
    if cond1 and cond2 and cond3:
        ques_ans_count[key] = item["answers"]
        ques_image_ans[key] = item["image_ans"]
        count += item["count"]

color = 0
count = 0
for ques in ques_ans_count.keys():
    if 'color' in ques:     # question about color
        color += 1
    if 'How many' in ques:  # question about counting
        count += 1
    
meta_traintest = {}

for ques, ans in ques_image_ans.items():
    train = []
    test = []
    for a, data in ans.items():
        split = round(len(data) * 0.7)
        shuffled_data = data.copy()
        random.Random(2021).shuffle(shuffled_data)
        train += [(i[0],a) for i in shuffled_data[:split]]
        test += [(i[0],a) for i in shuffled_data[split:]]

    meta_traintest[ques] = {"train" : train, "test": test, "answers": list(ans.keys())}   

tasks = list(meta_traintest.keys())

meta_train_tasks = []
meta_test_tasks = []
split = round(len(tasks) * 0.7)
shuffled_data = tasks.copy()
random.Random(2021).shuffle(shuffled_data)
meta_train_tasks += [i for i in shuffled_data[:split]]
meta_test_tasks += [i for i in shuffled_data[split:]]

meta_test = {}

for task in meta_test_tasks:
    meta_test[task] = meta_traintest[task]

with open(os.path.join(dataDir,"Meta/meta_test.json"),"w") as file:
    json.dump(meta_test,file)

meta_train = {}

for task in meta_train_tasks:
    meta_train[task] = meta_traintest[task]

with open(os.path.join(dataDir,"Meta/meta_train.json"),"w") as file:
    json.dump(meta_train,file)
