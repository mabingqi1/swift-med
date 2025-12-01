import json
from collections import Counter
from pathlib import Path


def analyze_impressions(jsonl_file: str):
    """
    分析JSONL文件中impression字段的疾病类型
    
    Args:
        jsonl_file: 输入的JSONL文件路径
    """
    input_path = Path(jsonl_file)
    
    if not input_path.exists():
        print(f"文件不存在: {jsonl_file}")
        return
    
    # 用于统计的数据结构
    all_diseases = []  # 所有疾病列表
    disease_counter = Counter()  # 疾病计数器
    impression_samples = {}  # 每种疾病的示例
    total_count = 0
    normal_count = 0  # "无异常"的数量
    
    print("开始分析文件...")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                total_count += 1
                
                # 提取impression内容
                messages = data.get('messages', [])
                impression = None
                
                for msg in messages:
                    if msg.get('role') == 'assistant':
                        content = msg.get('content', '')
                        # 提取Impression部分
                        if 'Impression:' in content:
                            impression = content.split('Impression:')[1].strip()
                        break
                
                if not impression:
                    continue
                
                # 统计"无异常"
                if impression == "无异常":
                    normal_count += 1
                    continue
                
                # 按逗号分割疾病
                diseases = [d.strip() for d in impression.split(',') if d.strip()]
                
                for disease in diseases:
                    all_diseases.append(disease)
                    disease_counter[disease] += 1
                    
                    # 保存每种疾病的示例（最多保存5个）
                    if disease not in impression_samples:
                        impression_samples[disease] = []
                    if len(impression_samples[disease]) < 5:
                        impression_samples[disease].append({
                            'line': line_num,
                            'impression': impression,
                            'video': data.get('videos', [''])[0]
                        })
                
            except json.JSONDecodeError as e:
                print(f"JSON解析错误 (行 {line_num}): {e}")
            except Exception as e:
                print(f"处理错误 (行 {line_num}): {e}")
    
    # 输出统计结果
    print("\n" + "=" * 80)
    print(f"分析完成！")
    print("=" * 80)
    print(f"\n总样本数: {total_count}")
    print(f"无异常样本数: {normal_count} ({normal_count/total_count*100:.2f}%)")
    print(f"有疾病样本数: {total_count - normal_count} ({(total_count-normal_count)/total_count*100:.2f}%)")
    print(f"疾病种类数: {len(disease_counter)}")
    print(f"疾病总次数: {sum(disease_counter.values())}")
    
    # 按频率排序输出
    print("\n" + "-" * 80)
    print("疾病频率统计（按出现次数降序）:")
    print("-" * 80)
    print(f"{'排名':<6} {'疾病名称':<30} {'出现次数':<10} {'占比':<10}")
    print("-" * 80)
    
    total_diseases = sum(disease_counter.values())
    for idx, (disease, count) in enumerate(disease_counter.most_common(), 1):
        percentage = count / total_diseases * 100
        print(f"{idx:<6} {disease:<30} {count:<10} {percentage:.2f}%")
    
    # 输出Top 10疾病的示例
    print("\n" + "=" * 80)
    print("Top 10 疾病示例:")
    print("=" * 80)
    
    for idx, (disease, count) in enumerate(disease_counter.most_common(10), 1):
        print(f"\n{idx}. {disease} (出现 {count} 次)")
        print("-" * 80)
        for sample_idx, sample in enumerate(impression_samples[disease][:3], 1):
            print(f"  示例 {sample_idx} (行 {sample['line']}):")
            print(f"    Impression: {sample['impression']}")
            print(f"    Video: {sample['video']}")
    
    # 保存完整统计到文件
    output_file = input_path.parent / f"{input_path.stem}_disease_analysis.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"疾病统计分析报告\n")
        f.write(f"{'=' * 80}\n")
        f.write(f"文件: {jsonl_file}\n")
        f.write(f"总样本数: {total_count}\n")
        f.write(f"无异常样本数: {normal_count} ({normal_count/total_count*100:.2f}%)\n")
        f.write(f"疾病种类数: {len(disease_counter)}\n\n")
        
        f.write(f"所有疾病列表（按频率降序）:\n")
        f.write(f"{'-' * 80}\n")
        for idx, (disease, count) in enumerate(disease_counter.most_common(), 1):
            percentage = count / total_diseases * 100
            f.write(f"{idx}. {disease}: {count} 次 ({percentage:.2f}%)\n")
    
    print(f"\n完整统计已保存到: {output_file}")


if __name__ == "__main__":
    jsonl_file = "/yinghepool/mabingqi/dataset/vlm/head_report/tiantan/series_level/tt79k_ym122k_BaiChuanAISU-QwenTemplate-train.jsonl"
    analyze_impressions(jsonl_file)