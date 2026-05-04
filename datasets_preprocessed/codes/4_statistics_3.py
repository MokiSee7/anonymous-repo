#!/usr/bin/env python3
"""
Step 4: Statistics for Step 3 Aligned Documents and Summaries

This script generates statistics for aligned documents and summaries including:
- Average number of shots per sentence (document vs summary)
- Average number of words per sentence (document vs summary)
- Sentence compression ratio (summary vs document)
- Word compression ratio (summary vs document)

Input:  datasets_preprocessed/data/processed/{dataset_name}_data.json
Output: datasets_preprocessed/data/statistics/{dataset_name}_statistic_3.json
"""

import os
import json
import argparse
from collections import defaultdict


def count_words(text):
    """Count words in a sentence"""
    if not text:
        return 0
    # Simple word count by splitting on whitespace
    return len(text.split())


def calculate_document_statistics(data_file, output_file):
    """
    Calculate statistics for aligned documents and summaries

    Args:
        data_file: Path to the input JSON file (e.g., summe_data.json)
        output_file: Path to save the statistics JSON file

    Returns:
        dict: Statistics dictionary
    """
    # Load data
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        return None

    with open(data_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Statistics containers
    total_videos = 0
    videos_with_documents = 0
    videos_with_summaries = 0

    # Document statistics
    doc_total_sentences = 0
    doc_total_shots_aligned = 0
    doc_total_words = 0
    doc_shots_per_sentence_list = []
    doc_words_per_sentence_list = []

    # Document vs Shot-level compression
    doc_total_shot_words = 0  # Total words from aligned shots
    doc_sentence_vs_shot_compression_list = []  # Per-sentence compression ratios

    # Summary statistics
    sum_total_sentences = 0
    sum_total_shots_aligned = 0
    sum_total_words = 0
    sum_shots_per_sentence_list = []
    sum_words_per_sentence_list = []

    # Summary vs Shot-level compression
    sum_total_shot_words = 0  # Total words from aligned shots
    sum_sentence_vs_shot_compression_list = []  # Per-sentence compression ratios

    # Summary vs Document compression
    sum_vs_doc_compression_list = []  # Per-video summary/document word ratio

    video_stats = []

    # Process each video
    for video_id, video_data in dataset.items():
        total_videos += 1

        has_document = 'aligned_full_document' in video_data
        has_summary = 'aligned_text_summary' in video_data
        clips = video_data.get('clips', [])

        # Get total number of shots for this video
        total_video_shots = len(clips)

        if not has_document and not has_summary:
            continue

        video_stat = {
            "video_id": video_id,
            "has_document": has_document,
            "has_summary": has_summary,
            "total_shots": total_video_shots
        }

        # Process full document
        if has_document:
            videos_with_documents += 1
            doc = video_data['aligned_full_document']

            num_sentences = doc.get('num_sentences', 0)
            sentences = doc.get('sentences', [])
            alignments = doc.get('alignments', [])

            # Calculate shots per sentence and word compression
            total_shots = 0
            total_doc_words = 0
            total_aligned_shot_words = 0

            for i, alignment in enumerate(alignments):
                shot_ids = alignment.get('shot_ids', [])
                num_shots = len(shot_ids)
                total_shots += num_shots
                doc_shots_per_sentence_list.append(num_shots)

                # Get sentence word count
                if i < len(sentences):
                    sentence_words = count_words(sentences[i])
                    total_doc_words += sentence_words
                    doc_words_per_sentence_list.append(sentence_words)

                    # Get aligned shots' total word count
                    aligned_shot_words = 0
                    for shot_id in shot_ids:
                        # Find the corresponding clip
                        matching_clips = [c for c in clips if c.get('clip_id') == shot_id]
                        if matching_clips:
                            shot_summary = matching_clips[0].get('summary', '')
                            aligned_shot_words += count_words(shot_summary)

                    total_aligned_shot_words += aligned_shot_words

                    # Calculate per-sentence compression ratio
                    if aligned_shot_words > 0:
                        sent_compression = (sentence_words / aligned_shot_words) * 100
                        doc_sentence_vs_shot_compression_list.append(sent_compression)

            doc_total_sentences += num_sentences
            doc_total_shots_aligned += total_shots
            doc_total_words += total_doc_words
            doc_total_shot_words += total_aligned_shot_words

            video_stat['document'] = {
                'num_sentences': num_sentences,
                'num_shots_aligned': total_shots,
                'num_words': total_doc_words,
                'aligned_shots_total_words': total_aligned_shot_words,
                'avg_shots_per_sentence': total_shots / num_sentences if num_sentences > 0 else 0,
                'avg_words_per_sentence': total_doc_words / num_sentences if num_sentences > 0 else 0,
                'sentence_compression_vs_shots': (num_sentences / total_video_shots * 100) if total_video_shots > 0 else 0,
                'word_compression_vs_shots': (total_doc_words / total_aligned_shot_words * 100) if total_aligned_shot_words > 0 else 0
            }

        # Process summary
        if has_summary:
            videos_with_summaries += 1
            summary = video_data['aligned_text_summary']

            num_sentences = summary.get('num_sentences', 0)
            sentences = summary.get('sentences', [])
            alignments = summary.get('alignments', [])

            # Calculate shots per sentence and word compression
            total_shots = 0
            total_sum_words = 0
            total_aligned_shot_words = 0

            for i, alignment in enumerate(alignments):
                shot_ids = alignment.get('shot_ids', [])
                num_shots = len(shot_ids)
                total_shots += num_shots
                sum_shots_per_sentence_list.append(num_shots)

                # Get sentence word count
                if i < len(sentences):
                    sentence_words = count_words(sentences[i])
                    total_sum_words += sentence_words
                    sum_words_per_sentence_list.append(sentence_words)

                    # Get aligned shots' total word count
                    aligned_shot_words = 0
                    for shot_id in shot_ids:
                        # Find the corresponding clip
                        matching_clips = [c for c in clips if c.get('clip_id') == shot_id]
                        if matching_clips:
                            shot_summary = matching_clips[0].get('summary', '')
                            aligned_shot_words += count_words(shot_summary)

                    total_aligned_shot_words += aligned_shot_words

                    # Calculate per-sentence compression ratio
                    if aligned_shot_words > 0:
                        sent_compression = (sentence_words / aligned_shot_words) * 100
                        sum_sentence_vs_shot_compression_list.append(sent_compression)

            sum_total_sentences += num_sentences
            sum_total_shots_aligned += total_shots
            sum_total_words += total_sum_words
            sum_total_shot_words += total_aligned_shot_words

            # Get GT shot count (shots with gt=1)
            gt_shots = [c for c in clips if c.get('gt', 0) == 1]
            num_gt_shots = len(gt_shots)

            video_stat['summary'] = {
                'num_sentences': num_sentences,
                'num_shots_aligned': total_shots,
                'num_gt_shots': num_gt_shots,
                'num_words': total_sum_words,
                'aligned_shots_total_words': total_aligned_shot_words,
                'avg_shots_per_sentence': total_shots / num_sentences if num_sentences > 0 else 0,
                'avg_words_per_sentence': total_sum_words / num_sentences if num_sentences > 0 else 0,
                'sentence_compression_vs_shots': (num_sentences / num_gt_shots * 100) if num_gt_shots > 0 else 0,
                'word_compression_vs_shots': (total_sum_words / total_aligned_shot_words * 100) if total_aligned_shot_words > 0 else 0
            }

        # Calculate Summary vs Document compression
        if has_document and has_summary:
            doc_words = video_stat['document']['num_words']
            sum_words = video_stat['summary']['num_words']

            video_stat['summary_vs_document_compression'] = {
                'word_compression_ratio': (sum_words / doc_words * 100) if doc_words > 0 else 0
            }

            if doc_words > 0:
                sum_vs_doc_compression_list.append((sum_words / doc_words) * 100)

        video_stats.append(video_stat)

    # Calculate overall averages
    # Document averages
    avg_doc_shots_per_sentence = sum(doc_shots_per_sentence_list) / len(doc_shots_per_sentence_list) if doc_shots_per_sentence_list else 0
    avg_doc_words_per_sentence = sum(doc_words_per_sentence_list) / len(doc_words_per_sentence_list) if doc_words_per_sentence_list else 0

    # Summary averages
    avg_sum_shots_per_sentence = sum(sum_shots_per_sentence_list) / len(sum_shots_per_sentence_list) if sum_shots_per_sentence_list else 0
    avg_sum_words_per_sentence = sum(sum_words_per_sentence_list) / len(sum_words_per_sentence_list) if sum_words_per_sentence_list else 0

    # Document vs Shot-level compression
    avg_doc_sentence_vs_shot_compression = sum(doc_sentence_vs_shot_compression_list) / len(doc_sentence_vs_shot_compression_list) if doc_sentence_vs_shot_compression_list else 0
    overall_doc_word_vs_shot_compression = (doc_total_words / doc_total_shot_words * 100) if doc_total_shot_words > 0 else 0

    # Summary vs Shot-level compression
    avg_sum_sentence_vs_shot_compression = sum(sum_sentence_vs_shot_compression_list) / len(sum_sentence_vs_shot_compression_list) if sum_sentence_vs_shot_compression_list else 0
    overall_sum_word_vs_shot_compression = (sum_total_words / sum_total_shot_words * 100) if sum_total_shot_words > 0 else 0

    # Summary vs Document compression
    avg_sum_vs_doc_compression = sum(sum_vs_doc_compression_list) / len(sum_vs_doc_compression_list) if sum_vs_doc_compression_list else 0

    # Compile statistics
    statistics = {
        "dataset_summary": {
            "total_videos": total_videos,
            "videos_with_documents": videos_with_documents,
            "videos_with_summaries": videos_with_summaries,

            # Document statistics
            "document": {
                "total_sentences": doc_total_sentences,
                "total_shots_aligned": doc_total_shots_aligned,
                "total_words": doc_total_words,
                "aligned_shots_total_words": doc_total_shot_words,
                "avg_shots_per_sentence": round(avg_doc_shots_per_sentence, 2),
                "avg_words_per_sentence": round(avg_doc_words_per_sentence, 2),
                "avg_sentence_compression_vs_shots": round(avg_doc_sentence_vs_shot_compression, 2),
                "overall_word_compression_vs_shots": round(overall_doc_word_vs_shot_compression, 2)
            },

            # Summary statistics
            "summary": {
                "total_sentences": sum_total_sentences,
                "total_shots_aligned": sum_total_shots_aligned,
                "total_words": sum_total_words,
                "aligned_shots_total_words": sum_total_shot_words,
                "avg_shots_per_sentence": round(avg_sum_shots_per_sentence, 2),
                "avg_words_per_sentence": round(avg_sum_words_per_sentence, 2),
                "avg_sentence_compression_vs_shots": round(avg_sum_sentence_vs_shot_compression, 2),
                "overall_word_compression_vs_shots": round(overall_sum_word_vs_shot_compression, 2)
            },

            # Summary vs Document compression
            "summary_vs_document": {
                "avg_word_compression_ratio": round(avg_sum_vs_doc_compression, 2)
            }
        },
        "per_video_stats": video_stats
    }

    # Save statistics
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, indent=2, ensure_ascii=False)

    return statistics


def print_statistics(stats, dataset_name):
    """Print statistics in a readable format"""
    summary = stats['dataset_summary']

    print(f"\n{'='*80}")
    print(f"统计结果: {dataset_name} (Aligned Documents & Summaries)")
    print(f"{'='*80}")

    print(f"📊 总体统计:")
    print(f"   - 视频总数:                                 {summary['total_videos']}")
    print(f"   - 包含完整文档的视频数:                     {summary['videos_with_documents']}")
    print(f"   - 包含摘要的视频数:                         {summary['videos_with_summaries']}")

    print(f"\n📄 完整文档 (Full Document):")
    doc = summary['document']
    print(f"   - 句子总数:                                 {doc['total_sentences']}")
    print(f"   - 对齐的 shot 总数:                         {doc['total_shots_aligned']}")
    print(f"   - Document 单词总数:                        {doc['total_words']}")
    print(f"   - 对齐 shots 的单词总数:                    {doc['aligned_shots_total_words']}")
    print(f"   - 平均每句对应 shot 数:                     {doc['avg_shots_per_sentence']}")
    print(f"   - 平均每句单词数:                           {doc['avg_words_per_sentence']}")
    print(f"   - 句子压缩比 (vs all shots):                {doc['avg_sentence_compression_vs_shots']:.2f}%")
    print(f"   - 单词压缩比 (vs aligned shots):            {doc['overall_word_compression_vs_shots']:.2f}%")

    print(f"\n📝 摘要 (Summary):")
    summ = summary['summary']
    print(f"   - 句子总数:                                 {summ['total_sentences']}")
    print(f"   - 对齐的 shot 总数:                         {summ['total_shots_aligned']}")
    print(f"   - Summary 单词总数:                         {summ['total_words']}")
    print(f"   - 对齐 shots 的单词总数:                    {summ['aligned_shots_total_words']}")
    print(f"   - 平均每句对应 shot 数:                     {summ['avg_shots_per_sentence']}")
    print(f"   - 平均每句单词数:                           {summ['avg_words_per_sentence']}")
    print(f"   - 句子压缩比 (vs GT shots):                 {summ['avg_sentence_compression_vs_shots']:.2f}%")
    print(f"   - 单词压缩比 (vs aligned GT shots):         {summ['overall_word_compression_vs_shots']:.2f}%")

    print(f"\n📉 Summary vs Document 压缩比:")
    sum_vs_doc = summary['summary_vs_document']
    print(f"   - 单词压缩比 (Summary/Document):            {sum_vs_doc['avg_word_compression_ratio']:.2f}%")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Generate statistics for Step 3 aligned documents and summaries'
    )
    parser.add_argument('--dataset', type=str, default=None,
                       help='Dataset name to process (e.g., summe, tvsum, ovp, youtube)')
    parser.add_argument('--all', action='store_true',
                       help='Process all datasets')
    parser.add_argument('--data-dir', type=str, default='datasets_preprocessed/data/processed',
                       help='Directory containing processed JSON files (default: datasets_preprocessed/data/processed)')
    parser.add_argument('--output-dir', type=str, default='datasets_preprocessed/data/statistics',
                       help='Directory for output statistics files (default: datasets_preprocessed/data/statistics)')

    args = parser.parse_args()

    print(f"{'='*80}")
    print(f"Step 4: Statistics for Step 3 Aligned Documents")
    print(f"{'='*80}\n")

    print(f"📂 数据目录: {args.data_dir}")
    print(f"📂 输出目录: {args.output_dir}")

    # Determine which datasets to process
    if args.all:
        # Find all *_data.json files in the data directory
        import glob
        data_files = glob.glob(f"{args.data_dir}/*_data.json")
        datasets = [os.path.basename(f).replace('_data.json', '') for f in data_files]
        print(f"🎯 处理所有数据集: {', '.join(datasets)}")
    elif args.dataset:
        datasets = [args.dataset]
        print(f"🎯 目标数据集: {args.dataset}")
    else:
        print("❌ 错误: 请指定 --dataset 或 --all")
        return

    print(f"{'='*80}\n")

    # Process each dataset
    success_count = 0
    for dataset in datasets:
        print(f"🎬 处理数据集: {dataset}")
        print(f"{'-'*80}")

        data_file = os.path.join(args.data_dir, f"{dataset}_data.json")
        output_file = os.path.join(args.output_dir, f"{dataset}_statistic_3.json")

        print(f"📂 输入文件: {data_file}")
        print(f"📂 输出文件: {output_file}")

        if not os.path.exists(data_file):
            print(f"❌ 数据文件不存在，跳过")
            continue

        stats = calculate_document_statistics(data_file, output_file)

        if stats:
            print(f"✅ 统计信息已生成: {output_file}")
            print_statistics(stats, dataset)
            success_count += 1
        else:
            print(f"❌ 处理失败: {dataset}")

        print(f"{'-'*80}\n")

    print(f"{'='*80}")
    print(f"✅ 完成! 成功处理 {success_count}/{len(datasets)} 个数据集")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
