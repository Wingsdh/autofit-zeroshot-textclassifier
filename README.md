# autofit-zeroshot-textclassifier
Classify Text without any data

## Installation
``` bash
pip install -r requirements.txt
```

## Command
* Download tnews
``` bash
cd data/agnews && bash get_data.sh && cd ../../
```
|description|label|title|
|--|--|--|
|AMD #39;s new dual-core Opteron chip is designed mainly for corporate computing applications, including databases, Web services, and financial transactions.|3 (Sci/Tech)|AMD Debuts Dual-Core Opteron Processor|
|Reuters - Major League Baseball\Monday announced a decision on the appeal filed by Chicago Cubs\pitcher Kerry Wood regarding a suspension stemming from an\incident earlier this season.|1 (Sports)|Wood's Suspension Upheld (Reuters)|
|President Bush #39;s quot;revenue-neutral quot; tax reform needs losers to balance its winners, and people claiming the federal deduction for state and local taxes may be in administration planners #39; sights, news reports say.|2 (Business)|Bush reform may have blue states seeing red|

* classify
```bash
python classify.py -model_path news -text_path data/agnews/test.txt -annotation_path data/agnews/test_labels.txt -label_path data/agnews/label_names.txt
```

```
2022-05-14 10:24:39.375 | INFO     | models.base_model:read_labels:23 - Read 4 from data/agnews/label_names.txt
2022-05-14 10:24:39.375 | INFO     | models.base_model:read_labels:24 - Labels: ['politics', 'sports', 'business', 'technology']
2022-05-14 10:24:46.147 | INFO     | __main__:main:25 - Acc: 5081 / 7600 = 0.6685526315789474
```

## Resource<br>
https://www.kaggle.com/code/procode/sif-embeddings-got-69-accuracy
