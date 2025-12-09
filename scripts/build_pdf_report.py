"""Build a PDF report from the concise results MD and selected figures.
Creates: reports/experiment_results_discussion.pdf

This script uses matplotlib to compose pages and embed PNG/GIF frames.
"""
import os
from textwrap import wrap
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MD_PATH = os.path.join(ROOT, 'reports', 'experiment_results_discussion.md')
OUT_PDF = os.path.join(ROOT, 'reports', 'experiment_results_discussion.pdf')
FIG_DIR = os.path.join(ROOT, 'figures')

# Images to include with captions (order matters)
IMAGES = [
    ('top3_boxplot_20251110_171609.png', '図1．Top-3 の Q-table ごとの得点分布（箱ひげ図）．分布のばらつきや外れ値を示す．'),
    ('top3_per_qtable_means_20251110_171609.png', '図2．各 Q-table の平均得点を示す棒グラフ．Top 候補間の平均的な差を示す．'),
    ('best_qtable_render_20251110_171609.gif', '図3．代表 Q-table の再生 GIF（短時間のアニメーション）．挙動の安定度を直感的に確認できる．'),
]

PAGE_WIDTH = 8.27  # A4 inches
PAGE_HEIGHT = 11.69


def read_md_text(path):
    if not os.path.exists(path):
        return ''
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def draw_text_page(pdf, title, body_lines):
    fig = plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.94, bottom=0.06)
    ax = fig.add_subplot(111)
    ax.axis('off')

    # Title
    ax.text(0.5, 0.95, title, ha='center', va='top', fontsize=18, weight='bold')

    y = 0.88
    line_height = 0.024
    for line in body_lines:
        wrapped = wrap(line, 90)
        for w in wrapped:
            ax.text(0.02, y, w, ha='left', va='top', fontsize=10)
            y -= line_height
            if y < 0.08:
                pdf.savefig(fig)
                plt.close(fig)
                fig = plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT))
                fig.subplots_adjust(left=0.08, right=0.98, top=0.94, bottom=0.06)
                ax = fig.add_subplot(111)
                ax.axis('off')
                y = 0.95
    pdf.savefig(fig)
    plt.close(fig)


def add_image_page(pdf, image_path, caption):
    if not os.path.exists(image_path):
        # create a placeholder page
        fig = plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT))
        ax = fig.add_subplot(111)
        ax.axis('off')
        ax.text(0.5, 0.5, f"Missing: {os.path.basename(image_path)}", ha='center', va='center', fontsize=14, color='red')
        fig.text(0.02, 0.02, caption, ha='left', va='bottom', fontsize=9)
        pdf.savefig(fig)
        plt.close(fig)
        return

    # Handle GIF by taking first frame
    ext = os.path.splitext(image_path)[1].lower()
    if ext in ('.gif',):
        try:
            with Image.open(image_path) as im:
                im = im.convert('RGBA')
                # use first frame
                arr = im
                fig = plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT))
                ax = fig.add_subplot(111)
                ax.axis('off')
                ax.imshow(arr)
                fig.text(0.02, 0.02, caption, ha='left', va='bottom', fontsize=9)
                pdf.savefig(fig)
                plt.close(fig)
                return
        except Exception:
            pass

    # For PNG/JPG
    try:
        img = mpimg.imread(image_path)
        fig = plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT))
        ax = fig.add_subplot(111)
        ax.imshow(img)
        ax.axis('off')
        fig.text(0.02, 0.02, caption, ha='left', va='bottom', fontsize=9)
        pdf.savefig(fig)
        plt.close(fig)
    except Exception as e:
        fig = plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT))
        ax = fig.add_subplot(111)
        ax.axis('off')
        ax.text(0.5, 0.5, f"Error loading {os.path.basename(image_path)}:\n{e}", ha='center', va='center', fontsize=10, color='red')
        fig.text(0.02, 0.02, caption, ha='left', va='bottom', fontsize=9)
        pdf.savefig(fig)
        plt.close(fig)


def main():
    md_text = read_md_text(MD_PATH)
    if not md_text:
        print('Markdown not found, aborting')
        return

    # Extract title and body: use first header as title
    lines = md_text.splitlines()
    title = '実験結果と考察'
    body = []
    for ln in lines:
        if ln.strip().startswith('#'):
            continue
        body.append(ln)

    # Create PDF
    os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
    with PdfPages(OUT_PDF) as pdf:
        # Title page
        draw_text_page(pdf, title, ['以下は実験の結果と考察である．概要は省略し，結果と考察に焦点を当てている．', ''])
        # Add the main text pages
        draw_text_page(pdf, '結果と考察', body)

        # Add images with captions
        for fname, caption in IMAGES:
            img_path = os.path.join(FIG_DIR, fname)
            add_image_page(pdf, img_path, caption)

    print('PDF written to', OUT_PDF)


if __name__ == '__main__':
    main()
