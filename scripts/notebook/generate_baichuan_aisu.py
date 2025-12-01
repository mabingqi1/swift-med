import json
from pathlib import Path
from typing import Dict, Any
import math


def convert_to_qwen_format(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    将原始数据格式转换为Qwen格式
    
    Args:
        input_data: 包含 wyg_series, gt, series_zst_path 的字典
        
    Returns:
        转换后的Qwen格式数据
    """
    # 构建assistant的回复内容
    findings = input_data.get('wys_series', '')
    impressions = input_data.get('gt', '')
    
    # 处理NaN值：如果gt为NaN、None、空字符串，则替换为"无异常"
    if impressions is None or impressions == '' or (isinstance(impressions, float) and math.isnan(impressions)):
        impressions = "无异常"
    
    # 同样处理findings的NaN值
    if findings is None or findings == '' or (isinstance(findings, float) and math.isnan(findings)):
        findings = "未见明显异常"
    
    # 去掉findings中所有的\r字符
    findings = findings.replace('\r', '')

    qwen_data = {
        "messages": [
            {
                "role": "system",
                "content": "You are a professional medical assistant specialized in analyzing medical imaging and generating accurate clinical reports."
            },
            {
                "role": "user",
                "content": "<video>Please extract Findings and Impression from medical video."
            },
            {
                "role": "assistant",
                "content": f"Findings: {findings}\nImpression: {impressions}"
            }
        ],
        "videos": [input_data['series_zst_path']]
    }
    return qwen_data


def convert_jsonl_file(input_file: str, output_file: str) -> None:
    """
    批量转换JSONL文件
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    total_count = 0
    success_count = 0
    error_count = 0
    nan_count = 0  # 统计NaN替换次数
    filtered_window_count = 0  # 非标准窗过滤
    filtered_depth_count = 0  # Depth不在范围过滤
    duplicate_count = 0  # 重复标准窗过滤
    selected_series_ids = set()  # 记录已选标准窗的series
    
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        for line_num, line in enumerate(f_in, 1):
            total_count += 1
            try:
                # 解析输入数据
                data = json.loads(line.strip())
                
                # 验证必要字段
                if not all(key in data for key in ['wys_series', 'gt', 'series_zst_path']):
                    print(f"Warning: Line {line_num} missing required fields, skipping...")
                    error_count += 1
                    continue
                
                # 过滤1：只选取window_type为"标准窗"的条目
                window_type = data.get('window_type', '')
                if window_type != '标准窗':
                    filtered_window_count += 1
                    continue
                
                # 过滤2：Depth需在[10, 64]区间
                depth_value = data.get('Depth')
                try:
                    depth_float = float(depth_value)
                    if not (10 <= depth_float <= 64):
                        filtered_depth_count += 1
                        continue
                except (TypeError, ValueError):
                    filtered_depth_count += 1
                    continue

                # 过滤3：同一 series 只保留首个标准窗
                series_id = data.get('series_instance_uid') or data.get('series_zst_path')
                if series_id in selected_series_ids:
                    duplicate_count += 1
                    continue
                selected_series_ids.add(series_id)

                # 检查是否有NaN值
                gt_value = data.get('gt', '')
                if gt_value is None or gt_value == '' or (isinstance(gt_value, float) and math.isnan(gt_value)):
                    nan_count += 1
                
                # 转换格式
                qwen_data = convert_to_qwen_format(data)
                
                # 写入输出文件
                f_out.write(json.dumps(qwen_data, ensure_ascii=False) + '\n')
                success_count += 1
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                error_count += 1
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                error_count += 1
    
    # 打印统计信息
    print(f"\n转换完成!")
    print(f"=" * 50)
    print(f"总行数: {total_count}")
    print(f"成功转换: {success_count}")
    print(f"NaN替换: {nan_count}")
    print(f"-" * 50)
    print(f"过滤统计:")
    print(f"  非标准窗过滤: {filtered_window_count}")
    print(f"  Depth范围过滤: {filtered_depth_count}")
    print(f"  重复标准窗过滤: {duplicate_count}")
    print(f"  错误行数: {error_count}")
    print(f"=" * 50)
    print(f"输出文件: {output_path}")


if __name__ == "__main__":
    # 输入输出文件路径
    input_file = "/yinghepool/huangmengqian/code/daily/2025-11/1126/generate_data/26w(7.9w)_tiantan_yimai_brain_structured_disease_dedup_CLEANED_v2_train_test.jsonl"
    output_file = "/yinghepool/mabingqi/dataset/vlm/head_report/tiantan/series_level/tt79k_ym122k_BaiChuanAISU-QwenTemplate-train.jsonl"
    
    # 执行转换
    convert_jsonl_file(input_file, output_file)
    
    # 读取并显示第一条转换后的数据作为示例
    print("\n第一条转换后的数据示例:")
    with open(output_file, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        if first_line:
            print(json.dumps(json.loads(first_line), ensure_ascii=False, indent=2))
        else:
            print("输出文件为空")