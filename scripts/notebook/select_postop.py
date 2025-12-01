import json

input_file = '/yinghepool/mabingqi/dataset/vlm/head_report/tiantan/series_level/20251101-yimai_tiantan-balance-train-V1.jsonl'  # 输入文件路径
output_file = '/yinghepool/mabingqi/dataset/vlm/head_report/tiantan/series_level/tt79k_ym122k_QwenTemplate_PostOperation-train.jsonl'

with open(input_file, 'r', encoding='utf-8') as f_in, \
     open(output_file, 'w', encoding='utf-8') as f_out:
    
    for line in f_in:
        data = json.loads(line.strip())
        
        # 检查 messages 中 assistant 的 content 是否包含"术后"
        for msg in data.get('messages', []):
            if msg.get('role') == 'assistant':
                content = msg.get('content', '')
                if '术后' in content:
                    f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                    break  # 找到一条匹配即可，跳出内层循环

print(f"筛选完成，结果已保存到 {output_file}")