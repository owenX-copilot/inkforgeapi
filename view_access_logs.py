#!/usr/bin/env python3
"""
查看访问日志工具
CSV格式的访问日志查看和分析
"""

import csv
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import pandas as pd

def read_access_logs(log_dir: str = "/mnt/data/logs", date_str: str = None):
    """读取访问日志"""
    log_dir = Path(log_dir)

    if date_str:
        # 读取指定日期的日志
        log_file = log_dir / f"access_{date_str}.csv"
        if not log_file.exists():
            print(f"日志文件不存在: {log_file}")
            return []

        files = [log_file]
    else:
        # 读取所有日志文件
        files = sorted(log_dir.glob("access_*.csv"))
        if not files:
            print(f"没有找到访问日志文件，目录: {log_dir}")
            return []

    all_records = []
    for file in files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row['_source_file'] = file.name
                    all_records.append(row)
            print(f"读取: {file.name} ({len(all_records)} 条记录)")
        except Exception as e:
            print(f"读取文件失败 {file}: {e}")

    return all_records

def print_summary(records):
    """打印摘要统计"""
    if not records:
        print("没有访问记录")
        return

    print("\n" + "="*60)
    print("访问日志摘要")
    print("="*60)

    # 基本统计
    total_requests = len(records)
    unique_ips = len(set(r['client_ip'] for r in records))

    # 状态码分布
    status_codes = {}
    for r in records:
        code = r['status_code']
        status_codes[code] = status_codes.get(code, 0) + 1

    # 路径统计
    paths = {}
    for r in records:
        path = r['path']
        paths[path] = paths.get(path, 0) + 1

    print(f"总请求数: {total_requests}")
    print(f"独立IP数: {unique_ips}")

    print("\n状态码分布:")
    for code, count in sorted(status_codes.items(), key=lambda x: int(x[0])):
        percentage = (count / total_requests) * 100
        print(f"  {code}: {count} ({percentage:.1f}%)")

    print("\n最常访问的路径 (前10):")
    for path, count in sorted(paths.items(), key=lambda x: x[1], reverse=True)[:10]:
        percentage = (count / total_requests) * 100
        print(f"  {path}: {count} ({percentage:.1f}%)")

    # 时间分布
    if records:
        first_time = min(r['timestamp'] for r in records)
        last_time = max(r['timestamp'] for r in records)
        print(f"\n时间范围: {first_time} 到 {last_time}")

def print_recent_requests(records, limit=20):
    """打印最近的请求"""
    if not records:
        return

    print(f"\n最近的 {limit} 条请求:")
    print("-"*100)

    # 按时间排序
    sorted_records = sorted(records, key=lambda x: x['timestamp'], reverse=True)[:limit]

    for i, record in enumerate(sorted_records, 1):
        print(f"{i:3d}. {record['timestamp'][11:19]} | "
              f"{record['client_ip']:15s} | "
              f"{record['method']:6s} | "
              f"{record['path']:30s} | "
              f"{record['status_code']:3s} | "
              f"{record['response_time_ms']:6s}ms")

def print_ip_stats(records):
    """打印IP统计"""
    if not records:
        return

    ip_counts = {}
    for r in records:
        ip = r['client_ip']
        ip_counts[ip] = ip_counts.get(ip, 0) + 1

    print("\nIP地址统计 (前20):")
    print("-"*60)

    for ip, count in sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)[:20]:
        percentage = (count / len(records)) * 100
        print(f"{ip:20s}: {count:5d} 次 ({percentage:.1f}%)")

def export_to_excel(records, output_file="access_logs.xlsx"):
    """导出到Excel"""
    if not records:
        print("没有数据可导出")
        return

    try:
        # 转换为DataFrame
        df = pd.DataFrame(records)

        # 删除内部字段
        if '_source_file' in df.columns:
            df = df.drop('_source_file', axis=1)

        # 保存到Excel
        df.to_excel(output_file, index=False)
        print(f"\n数据已导出到: {output_file}")
        print(f"总记录数: {len(df)}")

    except Exception as e:
        print(f"导出失败: {e}")

def main():
    parser = argparse.ArgumentParser(description="查看访问日志")
    parser.add_argument("--date", help="指定日期 (格式: YYYY-MM-DD)")
    parser.add_argument("--recent", type=int, default=20, help="显示最近的N条请求")
    parser.add_argument("--summary", action="store_true", help="显示摘要统计")
    parser.add_argument("--ips", action="store_true", help="显示IP统计")
    parser.add_argument("--export", help="导出到Excel文件")
    parser.add_argument("--log-dir", default="/mnt/data/logs", help="日志目录")

    args = parser.parse_args()

    # 读取日志
    records = read_access_logs(args.log_dir, args.date)

    if not records:
        return

    # 显示摘要
    if args.summary:
        print_summary(records)

    # 显示IP统计
    if args.ips:
        print_ip_stats(records)

    # 显示最近请求
    if not args.summary and not args.ips and not args.export:
        print_recent_requests(records, args.recent)

    # 导出到Excel
    if args.export:
        export_to_excel(records, args.export)

    print(f"\n总记录数: {len(records)}")

if __name__ == "__main__":
    main()