#!/usr/bin/env python3
"""Extract a glossary of EN-JA term pairs from parallel training data.

Identifies high-frequency English n-grams (1-3 words) that consistently
map to the same Japanese substring across multiple training pairs.

Usage:
    python -m scripts.build_glossary \
        --input data/splits/train_v1.jsonl \
        --output kb/glossary.csv \
        --min-freq 3 --max-terms 300
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _tokenize_en(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", text.lower())


def _extract_ngrams(tokens: list[str], max_n: int = 3) -> list[str]:
    ngrams = []
    for n in range(1, max_n + 1):
        for i in range(len(tokens) - n + 1):
            ngrams.append(" ".join(tokens[i : i + n]))
    return ngrams


_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "can", "could", "must", "and", "or", "but",
    "if", "then", "else", "when", "at", "by", "for", "with", "about",
    "against", "between", "through", "during", "before", "after", "above",
    "below", "to", "from", "up", "down", "in", "out", "on", "off", "over",
    "under", "again", "further", "than", "once", "here", "there", "all",
    "each", "every", "both", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "too", "very", "just",
    "because", "as", "until", "while", "of", "into", "that", "this", "it",
    "its", "i", "me", "my", "we", "our", "you", "your", "he", "him", "his",
    "she", "her", "they", "them", "their", "what", "which", "who", "whom",
    "how", "where", "why", "also", "still", "even", "much", "many",
}

CURATED_TERMS = [
    ("password", "パスワード", "Authentication term"),
    ("email", "メール", "Communication"),
    ("login", "ログイン", "Authentication"),
    ("logout", "ログアウト", "Authentication"),
    ("settings", "設定", "UI navigation"),
    ("user", "ユーザー", "General"),
    ("account", "アカウント", "Authentication"),
    ("download", "ダウンロード", "Action"),
    ("upload", "アップロード", "Action"),
    ("search", "検索", "UI action"),
    ("delete", "削除", "Destructive action"),
    ("cancel", "キャンセル", "Action"),
    ("confirm", "確認", "Action"),
    ("error", "エラー", "System message"),
    ("warning", "警告", "System message"),
    ("notification", "通知", "System message"),
    ("update", "更新", "Action"),
    ("save", "保存", "Action"),
    ("edit", "編集", "Action"),
    ("profile", "プロフィール", "User management"),
    ("dashboard", "ダッシュボード", "UI navigation"),
    ("menu", "メニュー", "UI navigation"),
    ("button", "ボタン", "UI element"),
    ("page", "ページ", "UI element"),
    ("link", "リンク", "UI element"),
    ("file", "ファイル", "Data"),
    ("folder", "フォルダ", "Data"),
    ("image", "画像", "Media"),
    ("video", "動画", "Media"),
    ("server", "サーバー", "Infrastructure"),
    ("database", "データベース", "Infrastructure"),
    ("network", "ネットワーク", "Infrastructure"),
    ("security", "セキュリティ", "Security"),
    ("privacy", "プライバシー", "Security"),
    ("permission", "権限", "Access control"),
    ("admin", "管理者", "Roles"),
    ("customer", "顧客", "Business"),
    ("product", "製品", "Business"),
    ("service", "サービス", "Business"),
    ("company", "会社", "Business"),
    ("report", "レポート", "Business"),
    ("project", "プロジェクト", "Business"),
    ("schedule", "スケジュール", "Planning"),
    ("meeting", "会議", "Planning"),
    ("budget", "予算", "Finance"),
    ("payment", "支払い", "Finance"),
    ("invoice", "請求書", "Finance"),
    ("price", "価格", "Finance"),
    ("cost", "コスト", "Finance"),
    ("tax", "税金", "Finance"),
    ("contract", "契約", "Legal"),
    ("policy", "ポリシー", "Legal"),
    ("agreement", "契約", "Legal"),
    ("license", "ライセンス", "Legal"),
    ("copyright", "著作権", "Legal"),
    ("system", "システム", "Technical"),
    ("application", "アプリケーション", "Technical"),
    ("software", "ソフトウェア", "Technical"),
    ("hardware", "ハードウェア", "Technical"),
    ("internet", "インターネット", "Technical"),
    ("website", "ウェブサイト", "Technical"),
    ("browser", "ブラウザ", "Technical"),
    ("mobile", "モバイル", "Technical"),
    ("desktop", "デスクトップ", "Technical"),
    ("cloud", "クラウド", "Technical"),
    ("data", "データ", "Technical"),
    ("backup", "バックアップ", "Technical"),
    ("install", "インストール", "Technical"),
    ("version", "バージョン", "Technical"),
    ("feature", "機能", "Product"),
    ("option", "オプション", "UI"),
    ("status", "ステータス", "System"),
    ("message", "メッセージ", "Communication"),
    ("comment", "コメント", "Communication"),
    ("review", "レビュー", "Feedback"),
    ("feedback", "フィードバック", "Feedback"),
    ("support", "サポート", "Customer service"),
    ("help", "ヘルプ", "Customer service"),
    ("guide", "ガイド", "Documentation"),
    ("tutorial", "チュートリアル", "Documentation"),
    ("manual", "マニュアル", "Documentation"),
    ("document", "ドキュメント", "Documentation"),
    ("template", "テンプレート", "Content"),
    ("category", "カテゴリ", "Organization"),
    ("tag", "タグ", "Organization"),
    ("label", "ラベル", "Organization"),
    ("filter", "フィルター", "UI action"),
    ("sort", "ソート", "UI action"),
    ("list", "リスト", "UI element"),
    ("table", "テーブル", "UI element"),
    ("chart", "チャート", "Visualization"),
    ("graph", "グラフ", "Visualization"),
    ("calendar", "カレンダー", "UI element"),
    ("form", "フォーム", "UI element"),
    ("input", "入力", "UI element"),
    ("output", "出力", "System"),
    ("process", "プロセス", "System"),
    ("task", "タスク", "Workflow"),
    ("workflow", "ワークフロー", "Workflow"),
    ("automation", "自動化", "Workflow"),
    ("integration", "統合", "Technical"),
    ("api", "API", "Technical"),
    ("token", "トークン", "Authentication"),
    ("authentication", "認証", "Security"),
    ("authorization", "認可", "Security"),
    ("encryption", "暗号化", "Security"),
    ("certificate", "証明書", "Security"),
    ("domain", "ドメイン", "Technical"),
    ("hosting", "ホスティング", "Infrastructure"),
    ("deployment", "デプロイ", "Infrastructure"),
    ("environment", "環境", "Infrastructure"),
    ("configuration", "設定", "System"),
    ("parameter", "パラメータ", "Technical"),
    ("variable", "変数", "Technical"),
    ("function", "関数", "Technical"),
    ("module", "モジュール", "Technical"),
    ("component", "コンポーネント", "Technical"),
    ("interface", "インターフェース", "Technical"),
    ("plugin", "プラグイン", "Technical"),
    ("extension", "拡張機能", "Technical"),
    ("theme", "テーマ", "UI"),
    ("layout", "レイアウト", "UI"),
    ("design", "デザイン", "UI"),
    ("style", "スタイル", "UI"),
    ("font", "フォント", "UI"),
    ("color", "色", "UI"),
    ("icon", "アイコン", "UI"),
    ("logo", "ロゴ", "Brand"),
    ("brand", "ブランド", "Brand"),
    ("marketing", "マーケティング", "Business"),
    ("campaign", "キャンペーン", "Marketing"),
    ("analytics", "アナリティクス", "Data"),
    ("metric", "メトリクス", "Data"),
    ("performance", "パフォーマンス", "System"),
    ("optimization", "最適化", "Technical"),
    ("cache", "キャッシュ", "Technical"),
    ("log", "ログ", "System"),
    ("debug", "デバッグ", "Development"),
    ("test", "テスト", "Development"),
    ("production", "本番", "Infrastructure"),
    ("staging", "ステージング", "Infrastructure"),
    ("development", "開発", "Infrastructure"),
    ("release", "リリース", "Development"),
    ("management", "管理", "Business"),
    ("organization", "組織", "Business"),
    ("team", "チーム", "Business"),
    ("member", "メンバー", "Business"),
    ("role", "ロール", "Access control"),
    ("group", "グループ", "Organization"),
    ("community", "コミュニティ", "Social"),
    ("forum", "フォーラム", "Social"),
    ("blog", "ブログ", "Content"),
    ("article", "記事", "Content"),
    ("content", "コンテンツ", "Content"),
    ("media", "メディア", "Content"),
    ("platform", "プラットフォーム", "Technical"),
    ("solution", "ソリューション", "Business"),
    ("strategy", "戦略", "Business"),
    ("analysis", "分析", "Data"),
    ("research", "研究", "Academic"),
    ("education", "教育", "Academic"),
    ("training", "トレーニング", "Learning"),
    ("certification", "資格", "Professional"),
    ("compliance", "コンプライアンス", "Legal"),
    ("regulation", "規制", "Legal"),
    ("standard", "標準", "Quality"),
    ("quality", "品質", "Quality"),
    ("maintenance", "メンテナンス", "Operations"),
    ("monitoring", "モニタリング", "Operations"),
    ("alert", "アラート", "System"),
    ("incident", "インシデント", "Operations"),
    ("recovery", "復旧", "Operations"),
    ("migration", "移行", "Technical"),
    ("upgrade", "アップグレード", "Technical"),
    ("downgrade", "ダウングレード", "Technical"),
    ("subscription", "サブスクリプション", "Business"),
    ("trial", "トライアル", "Business"),
    ("premium", "プレミアム", "Business"),
    ("free", "無料", "Pricing"),
    ("discount", "割引", "Pricing"),
    ("coupon", "クーポン", "Pricing"),
    ("order", "注文", "E-commerce"),
    ("cart", "カート", "E-commerce"),
    ("checkout", "チェックアウト", "E-commerce"),
    ("shipping", "配送", "E-commerce"),
    ("delivery", "配達", "E-commerce"),
    ("return", "返品", "E-commerce"),
    ("refund", "返金", "E-commerce"),
    ("exchange", "交換", "E-commerce"),
    ("inventory", "在庫", "E-commerce"),
    ("stock", "在庫", "E-commerce"),
    ("warehouse", "倉庫", "Logistics"),
    ("supply chain", "サプライチェーン", "Logistics"),
    ("logistics", "物流", "Logistics"),
    ("government", "政府", "General"),
    ("economy", "経済", "General"),
    ("society", "社会", "General"),
    ("environment", "環境", "General"),
    ("technology", "技術", "General"),
    ("innovation", "イノベーション", "General"),
    ("information", "情報", "General"),
    ("communication", "コミュニケーション", "General"),
    ("responsibility", "責任", "General"),
    ("opportunity", "機会", "General"),
    ("challenge", "課題", "General"),
    ("solution", "解決策", "General"),
    ("improvement", "改善", "General"),
    ("implementation", "実装", "Technical"),
    ("evaluation", "評価", "General"),
    ("recommendation", "推薦", "General"),
    ("conclusion", "結論", "General"),
    ("introduction", "紹介", "General"),
    ("summary", "要約", "General"),
    ("overview", "概要", "General"),
    ("detail", "詳細", "General"),
    ("example", "例", "General"),
    ("reference", "参考", "General"),
    ("resource", "リソース", "General"),
    ("tool", "ツール", "General"),
    ("method", "方法", "General"),
    ("approach", "アプローチ", "General"),
    ("framework", "フレームワーク", "Technical"),
    ("model", "モデル", "Technical"),
    ("algorithm", "アルゴリズム", "Technical"),
    ("machine learning", "機械学習", "AI"),
    ("artificial intelligence", "人工知能", "AI"),
    ("deep learning", "ディープラーニング", "AI"),
    ("neural network", "ニューラルネットワーク", "AI"),
    ("natural language", "自然言語", "AI"),
    ("translation", "翻訳", "Language"),
    ("language", "言語", "Language"),
    ("dictionary", "辞書", "Language"),
    ("grammar", "文法", "Language"),
    ("vocabulary", "語彙", "Language"),
    ("pronunciation", "発音", "Language"),
    ("conversation", "会話", "Language"),
]


def extract_from_data(
    rows: list[dict], min_freq: int = 3, max_terms: int = 300
) -> list[tuple[str, str, str]]:
    """Extract frequent EN terms and their most common JA co-occurrences."""
    en_freq: Counter = Counter()
    en_ja_map: dict[str, Counter] = defaultdict(Counter)

    for row in rows:
        en = row.get("source_en", "")
        ja = row.get("target_ja", "")
        tokens = _tokenize_en(en)
        ngrams = _extract_ngrams(tokens, max_n=2)

        for ng in ngrams:
            if ng in _STOPWORDS or len(ng) < 3:
                continue
            if all(w in _STOPWORDS for w in ng.split()):
                continue
            en_freq[ng] += 1
            en_ja_map[ng][ja] += 1

    terms = []
    for en_term, freq in en_freq.most_common(max_terms * 3):
        if freq < min_freq:
            break
        ja_counter = en_ja_map[en_term]
        top_ja = ja_counter.most_common(1)[0][0] if ja_counter else ""
        if top_ja and len(en_term) >= 3:
            terms.append((en_term, top_ja[:30], f"freq={freq}"))
        if len(terms) >= max_terms // 2:
            break

    return terms


def main() -> None:
    parser = argparse.ArgumentParser(description="Build EN-JA glossary")
    parser.add_argument("--input", default="data/splits/train_v1.jsonl")
    parser.add_argument("--output", default="kb/glossary.csv")
    parser.add_argument("--min-freq", type=int, default=3)
    parser.add_argument("--max-terms", type=int, default=300)
    args = parser.parse_args()

    rows = []
    input_path = Path(args.input)
    if input_path.exists():
        with input_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        print(f"Loaded {len(rows)} rows from {input_path}")

    data_terms = extract_from_data(rows, args.min_freq, args.max_terms) if rows else []
    print(f"Extracted {len(data_terms)} terms from training data")

    seen = set()
    all_terms = []

    for en, ja, note in CURATED_TERMS:
        key = en.lower()
        if key not in seen:
            seen.add(key)
            all_terms.append((en, ja, note, ""))

    for en, ja, note in data_terms:
        key = en.lower()
        if key not in seen:
            seen.add(key)
            all_terms.append((en, ja, note, ""))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source_term_en", "approved_ja", "usage_note", "forbidden_variants"])
        for row in all_terms:
            writer.writerow(row)

    print(f"Wrote {len(all_terms)} glossary terms to {out_path}")


if __name__ == "__main__":
    main()
