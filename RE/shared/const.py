task_ner_labels = {
    'ace04': ['FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER'],
    'ace05': ['FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER'],
    'scierc': ['Method', 'OtherScientificTerm', 'Task', 'Generic', 'Material', 'Metric'],
    'geodata': ['ALTE', 'CATE', 'DEPO', 'FAUL', 'GCA',
		  'GENE', 'GPA', 'GRA', 'MAG', 'META','MINE',
		  'MINU', 'OREB', 'RSA', 'SEDI', 'SIZE',
		  'SPAP', 'STRA', 'TECD', 'TECU', 'TIME', 'ZONA'],
}

task_rel_labels = {
    'ace04': ['PER-SOC', 'OTHER-AFF', 'ART', 'GPE-AFF', 'EMP-ORG', 'PHYS'],
    'ace05': ['ART', 'ORG-AFF', 'GEN-AFF', 'PHYS', 'PER-SOC', 'PART-WHOLE'],
    'scierc': ['PART-OF', 'USED-FOR', 'FEATURE-OF', 'CONJUNCTION', 'EVALUATE-FOR', 'HYPONYM-OF', 'COMPARE'],
    'geodata': ["侵入关系", "包含关系", "变质作用", "围岩蚀变作用", "属性关系", "岩浆作用", "成矿作用", "成矿载体（物质来源）",
                "指示作用", "控制作用", "时间关系", "构造作用", "沉积作用", "空间关系", "赋存关系", "限定作用"],
}


def get_labelmap(label_list):
    label2id = {}
    id2label = {}
    for i, label in enumerate(label_list):
        label2id[label] = i + 1
        id2label[i + 1] = label
    return label2id, id2label
