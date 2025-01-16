# A geological knowledge-constrained entity and relation extraction method


## Dataset
We used a Chinese geological entity-relation dataset created by ourselves, which includes train.json, dev.json, and test.json files. The data format is similar to SCIERC dataset. Each file consists of a set of records in the following format:
'''
{"clusters": [], "sentences": [["前", "人", "在", "川", "西", "甲", "基", "卡", "地", "区", "进", "行", "了", "锂", "辉", "石", "等", "伟", "晶", "岩", "稀", "有", "金", "属", "矿", "床", "相", "关", "矿", "石", "矿", "物", "的", "波", "谱", "测", "试", "研", "究", "和", "伟", "晶", "岩", "遥", "感", "地", "质", "填", "图", "(", "代", "晶", "晶", "等", ",", "2017", ",", "2018", ")", "。"]], "ner": [[[13, 15, "MINE"], [3, 9, "SPAP"], [17, 19, "MAG"], [20, 23, "CATE"]]], "relations": [[[13, 15, 3, 9, "空间关系"], [17, 19, 3, 9, "空间关系"], [20, 23, 3, 9, "空间关系"], [17, 19, 13, 15, "岩浆作用"], [17, 19, 20, 23, "属性关系"]]], "doc_key": "train_0"}
'''

## Usage
1. Run NER1/run_entity.py or NER2/run_acener.py to identify entities with entity types from sentences.
2. Run RE/run_relation.py for relation extraction.
3. Run RE/run_eval.py for model evalution.


## Reference
1. Zhong, Z., Chen, D., 2021. A frustratingly easy approach for joint entity and relation extraction. In: Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies. Association for Computational Linguistics, Online, pp. 50–61. https://doi.org/10.18653/v1/2021.naacl-main.5
2. Ye, D., Lin, Y., Li, P., Sun, M., 2022. Packed Levitated Marker for Entity and Relation Extraction. In: Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, Dublin, Ireland, pp. 4904-4917. https://doi.org/10.18653/v1/2022.acl-long.337

