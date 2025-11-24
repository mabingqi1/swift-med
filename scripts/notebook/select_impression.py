import json
import sys
from pathlib import Path

def extract_impressions_qwen(line: str) -> str:
    record = json.loads(line)
    for msg in record.get("messages", []):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if "Impressions:" in content:
                msg["content"] = content.split("Impressions:", 1)[1].strip()
            else:
                msg["content"] = ""
    return json.dumps(record, ensure_ascii=False)

def extract_impressions_raw(line: str) -> str:
    record = json.loads(line)
    report = record.get("report", "")
    if "Impressions:" in report:
        record["report"] = report.split("Impressions:", 1)[1].strip()
    else:
        record["report"] = ""
    return json.dumps(record, ensure_ascii=False)

def main():
    if len(sys.argv) != 3:
        print("用法: python tools/extract_impressions.py 输入.jsonl 输出.jsonl")
        return
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            if line.strip():
                fout.write(extract_impressions_qwen(line) + "\n")

if __name__ == "__main__":
    main()