"""Build results/nanoGPT_scaling_results.pdf from results/train_logs.json and results/scaling.png."""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

HERE = os.path.dirname(os.path.abspath(__file__))
logs = json.load(open(f'{HERE}/train_logs.json'))  # name -> [(step, val_loss, seconds)]
GRID = {  # name: (layers, d_model, heads, N in millions, peak lr, minutes on T4)
    's0': (2, 64, 2, 0.10, '3e-3', 20.5), 's1': (3, 128, 4, 0.60, '2e-3', 22.2),
    's2': (4, 192, 4, 1.78, '1.5e-3', 25.8), 's3': (6, 256, 8, 4.74, '1e-3', 30.4),
    's4': (6, 384, 8, 10.65, '8e-4', 38.6)}

# per-model validation curves vs training tokens
fig, ax = plt.subplots(figsize=(7, 4))
for name, rows in logs.items():
    ax.plot([s * 64 * 256 / 1e6 for s, _, _ in rows], [v for _, v, _ in rows], label=f'{name} (N={GRID[name][3]}M)')
ax.set(xscale='log', ylim=(1.5, 4), xlabel='Training tokens (millions)', ylabel='Validation loss',
       title='Validation loss during training')
ax.grid(alpha=.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f'{HERE}/training_curves.png', dpi=200)

# built-in Helvetica lacks Greek letters and math symbols; use a TTF that has them
FONT, FONT_B = 'Helvetica', 'Helvetica-Bold'
for reg, bold in [('C:/Windows/Fonts/arial.ttf', 'C:/Windows/Fonts/arialbd.ttf'),
                  ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf')]:
    if os.path.exists(reg) and os.path.exists(bold):
        pdfmetrics.registerFont(TTFont('Body', reg))
        pdfmetrics.registerFont(TTFont('Body-Bold', bold))
        pdfmetrics.registerFontFamily('Body', normal='Body', bold='Body-Bold', italic='Body', boldItalic='Body-Bold')
        FONT, FONT_B = 'Body', 'Body-Bold'
        break

ss = getSampleStyleSheet()
NAVY = colors.HexColor('#1f3a5f')
H1 = ParagraphStyle('H1', parent=ss['Heading1'], fontName=FONT_B, fontSize=20, textColor=NAVY, spaceAfter=4)
H2 = ParagraphStyle('H2', parent=ss['Heading2'], fontName=FONT_B, fontSize=13, textColor=NAVY, spaceBefore=14, spaceAfter=6)
B = ParagraphStyle('B', parent=ss['BodyText'], fontName=FONT, fontSize=10, leading=14)
SM = ParagraphStyle('SM', parent=B, fontSize=8.5, textColor=colors.grey)


def table(data, widths):
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY), ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), FONT_B), ('FONTNAME', (0, 1), (-1, -1), FONT), ('FONTSIZE', (0, 0), (-1, -1), 9), ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#eef2f7')]),
        ('GRID', (0, 0), (-1, -1), .4, colors.HexColor('#c5cdd8')),
        ('TOPPADDING', (0, 0), (-1, -1), 4), ('BOTTOMPADDING', (0, 0), (-1, -1), 4)]))
    return t


def img(path, w):
    iw, ih = PILImage.open(path).size
    return Image(path, width=w, height=w * ih / iw)


s = [Paragraph('nanoGPT Scaling Laws on Colab', H1),
     Paragraph('Five GPT models (0.10M to 10.65M non-embedding parameters) trained from scratch on TinyStories, '
               'on a Tesla T4 (15 GB) in Google Colab.', B)]

s += [Paragraph('1. Key results', H2),
      table([['Metric', 'Value'],
             ['Fitted size exponent α in L(N) = E + A·N^-α', '0.235'],
             ['Fitted irreducible loss E', '1.26 (nats/token)'],
             ['Compute-frontier slope, L ~ C^b', 'b = -0.089'],
             ['Best / worst final validation loss', '1.696 (s4) / 2.549 (s0)'],
             ['Total T4 training time', '≈ 2.3 hours (137.5 min)']], [10 * cm, 6 * cm]),
      Spacer(1, 6),
      Paragraph('Bigger models reached lower loss on the same 100M training tokens, and the improvement follows a smooth power law. '
                'The fitted α sits between the Kaplan et al. (≈0.076) and Chinchilla (≈0.34) reference values.', B)]

s += [Paragraph('2. Experimental setup', H2),
      Paragraph('<b>Model:</b> GPT-2-style decoder (pre-LayerNorm, GELU MLP, tied embeddings, flash attention), context 256, '
                'GPT-2 vocabulary padded to 50,304.<br/>'
                '<b>Data:</b> TinyStories tokenized with tiktoken (GPT-2 BPE). Each model sees 100M training tokens '
                '(6,103 steps × 64 × 256).<br/>'
                '<b>Optimizer:</b> AdamW (β = 0.9, 0.95; weight decay 0.1), 5% linear warm-up then cosine decay to 10% of peak LR, '
                'gradient clip 1.0.<br/>'
                '<b>Precision:</b> fp16 autocast with GradScaler (T4 has no bf16).<br/>'
                '<b>Evaluation:</b> 30 log-spaced validation points per run, 20 batches each. Compute is estimated as C = 6·N·D.<br/>'
                '<b>Seed:</b> one run per model (seed 0).', B)]

rows = [['Name', 'Layers', 'd_model', 'Heads', 'N (M)', 'Peak LR', 'Final val loss', 'T4 time (min)']]
for n, (L, d, h, N, lr, m) in GRID.items():
    rows.append([n, L, d, h, f'{N:.2f}', lr, f'{logs[n][-1][1]:.3f}', f'{m:.1f}'])
s += [Paragraph('3. Experiment grid and final losses', H2),
      table(rows, [1.5 * cm, 1.6 * cm, 1.9 * cm, 1.6 * cm, 1.7 * cm, 2 * cm, 3 * cm, 3 * cm])]

s += [PageBreak(), Paragraph('4. Scaling-law fit', H2), img(f'{HERE}/scaling.png', 17 * cm),
      Paragraph('Left: validation loss vs compute, one curve per model; the dashed line is a power-law fit to the lower envelope '
                '(compute frontier). Right: final loss vs model size with the fit L = 1.26 + 19.1·N<super>-0.235</super>.', SM)]

s += [Paragraph('5. Training curves', H2), img(f'{HERE}/training_curves.png', 14 * cm),
      Paragraph('Every model drops quickly in the first ~10M tokens, then improves slowly. '
                'Larger models learn faster per token and finish lower.', SM)]

s += [PageBreak(), Paragraph('6. How to read these results', H2),
      Paragraph('<b>Loss</b> is the average surprise (cross-entropy, in nats) the model has when predicting the next token. '
                'Lower is better. A random guesser over 50k tokens scores about 10.8; the models here reach 1.7 to 2.5.<br/><br/>'
                '<b>Scaling law.</b> Loss falls along a smooth curve as the model grows. This makes it possible to predict '
                'how a bigger model will behave from small ones.<br/><br/>'
                '<b>α = 0.235</b> measures how fast loss falls with size. Each 10× increase in parameters cuts the reducible '
                'part of the loss (L - E) by a factor of 10<super>0.235</super> ≈ 1.7.<br/><br/>'
                '<b>E = 1.26</b> is the loss an infinitely large model would still have, because language contains '
                'randomness no model can predict.<br/><br/>'
                '<b>Compute frontier.</b> Each curve in the left plot is one model. The best loss for a given compute budget '
                'is the lower edge of all curves: small models win at small budgets, large models at large budgets.', B)]

s += [Paragraph('7. Limitations', H2),
      Paragraph('• Only five models and one seed. A 3-parameter fit to 5 points is fragile; E and α are strongly coupled.<br/>'
                '• All models train on the same 100M tokens, so the largest are likely under-trained and the fit is not '
                'compute-optimal (Chinchilla-style).<br/>'
                '• Learning rates were set by hand per size, not tuned.<br/>'
                '• The plain log-log slope between the two largest models (s3 to s4) is only ≈ 0.065, below the fitted α, because '
                'E is not removed. Across all five models it is ≈ 0.087.<br/>'
                '• TinyStories is simple synthetic text; exponents may differ on web-scale data.<br/>'
                '• Reference exponents (Kaplan et al. 2020, arXiv 2001.08361; Hoffmann et al. 2022, arXiv 2203.15556) are approximate.', B)]

SimpleDocTemplate(f'{HERE}/nanoGPT_scaling_results.pdf', pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                  topMargin=2 * cm, bottomMargin=2 * cm, title='nanoGPT Scaling Laws on Colab').build(s)
print('ok')
