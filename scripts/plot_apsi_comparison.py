#!/usr/bin/env python3
"""Render paired differences and APSI coverage from completed comparisons.csv.

Uses only the standard library. The SVG is a standalone, editable vector figure.
Metric panels have separate axes. No inference or result calculation is performed.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from populism_axis.run_manifest import write_run_manifest

PANELS = [
    ('Spearman correlation', 'spearman', [
        ('english_holdout', 'holistic', 'GPD English speeches'),
        ('agora', 'holistic', 'AgoraSpeech paragraphs'),
        ('usgov_speeches', 'governors_2010_2018', 'US governors 2010–18'),
    ]),
    ('AUROC', 'auroc', [
        ('eu', 'both_vs_neither', 'EU-OOD: both vs. neither'),
        ('antielite_tweets', 'anti_elitism', 'Party tweets: anti-elitism'),
        ('popbert_en', 'anti_elitism', 'PopBERT: anti-elitism'),
        ('popbert_en', 'people_centrism', 'PopBERT: people-centrism'),
        ('portugal', 'anti_elitism', 'Portugal: anti-elitism'),
        ('portugal', 'people_centrism', 'Portugal: people-centrism'),
    ]),
    ('Macro-F1', 'macro_f1', [('eu', 'original_protocol_loso', 'EU-OOD: original protocol')]),
]
SYSTEMS = [('ours_political_debate', '#2878a3', 'Our approach with Political DEBATE large'),
           ('apsi_large_native', '#bd5a30', 'APSI large, native truncation')]


def number(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (ValueError, TypeError):
        return None


def interval(value):
    try:
        return [number(x) for x in json.loads(value.replace('nan', 'null'))]
    except (ValueError, AttributeError):
        return [None, None]


def render(rows, output):
    lookup = {(r['dataset'], r['task'], r['metric'], r['left_system']): r
              for r in rows if r['right_system'] == 'ours'}
    expected = [(d, task, metric, system) for _, metric, tasks in PANELS
                for d, task, _ in tasks for system, _, _ in SYSTEMS]
    missing = [key for key in expected if key not in lookup]
    if missing:
        raise ValueError(f'Missing primary comparisons: {missing}')
    width, height = 1240, 1190
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<title>APSI and Political DEBATE comparison</title>',
             '<desc>Paired metric differences against the existing axis plus BART measure, with 95 percent bootstrap intervals and APSI coverage. Each pair uses the same eligible documents.</desc>',
             '<rect width="100%" height="100%" fill="white"/>',
             '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#23313a} .small{font-size:12px} .label{font-size:14px} .title{font-size:23px;font-weight:700} .panel{font-size:16px;font-weight:700}</style>']

    def text(x, y, value, cls='label', **attrs):
        at = ' '.join(f'{k.replace("_", "-")}="{v}"' for k, v in attrs.items())
        parts.append(f'<text x="{x}" y="{y}" class="{cls}" {at}>{html.escape(str(value))}</text>')

    def line(x1, y1, x2, y2, color='#d8dfe3', stroke=1):
        parts.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="{stroke}"/>')

    text(35, 40, 'What does political specialization add?', 'title')
    text(35, 66, 'Difference from our existing axis + BART measure; points and 95% paired-bootstrap intervals.')
    for i, (_, color, label) in enumerate(SYSTEMS):
        y = 96 + i * 23
        parts.append(f'<circle cx="43" cy="{y-4}" r="5" fill="{color}"/>')
        text(57, y, label, 'small')
    text(900, 97, 'APSI coverage', 'panel')
    text(900, 119, 'Scored / eligible documents', 'small')
    plot_left, plot_right = 350, 845
    y = 165
    for panel_title, metric_name, tasks in PANELS:
        relevant = [lookup[(d, task, metric_name, s)] for d, task, _ in tasks for s, _, _ in SYSTEMS]
        bounds = [abs(x) for r in relevant for x in [number(r['delta']), *interval(r['delta_ci95'])] if x is not None]
        extent = max(.05, math.ceil(max(bounds, default=.05) * 20) / 20)
        axis = lambda v: plot_left + (v + extent) / (2 * extent) * (plot_right - plot_left)
        text(35, y, panel_title, 'panel')
        top = y + 20
        bottom = top + 61 * len(tasks)
        for tick in [-extent, -extent / 2, 0, extent / 2, extent]:
            x = axis(tick)
            line(x, top, x, bottom, '#aebbc3' if tick == 0 else '#edf0f2', 1.3 if tick == 0 else 1)
            text(x, bottom + 18, f'{tick:+.3f}' if tick else '0', 'small', text_anchor='middle')
        for i, (dataset, task, label) in enumerate(tasks):
            row_y = top + 27 + 61 * i
            text(35, row_y + 2, label)
            for j, (system, color, _) in enumerate(SYSTEMS):
                r = lookup[(dataset, task, metric_name, system)]
                yy = row_y - 10 + j * 19
                delta = number(r['delta']); low, high = interval(r['delta_ci95'])
                if low is not None and high is not None:
                    line(axis(low), yy, axis(high), yy, color, 2.2)
                    line(axis(low), yy-4, axis(low), yy+4, color, 1.2)
                    line(axis(high), yy-4, axis(high), yy+4, color, 1.2)
                if delta is not None:
                    parts.append(f'<circle cx="{axis(delta):.2f}" cy="{yy}" r="4.4" fill="{color}"/>')
                else:
                    text(plot_left + 8, yy + 4, 'Not estimable', 'small')
                valid = int(float(r['valid_draws'])); requested = int(float(r['requested_draws']))
                if valid < requested * .9:
                    text(852, yy + 4, '*', 'small')
            apsi = lookup[(dataset, task, metric_name, 'apsi_large_native')]
            n = int(float(apsi['left_n_scored'])); total = int(float(apsi['n_eligible']))
            frac = n / total if total else 0
            parts.append(f'<rect x="900" y="{row_y-15}" width="170" height="9" rx="3" fill="#edf0f2"/>')
            parts.append(f'<rect x="900" y="{row_y-15}" width="{170*frac:.2f}" height="9" rx="3" fill="#bd5a30"/>')
            text(900, row_y + 13, f'{n:,} / {total:,} ({100*frac:.1f}%)', 'small')
            text(1090, row_y + 3, f'n = {int(float(apsi["n_matched"])):,}', 'small')
        text((plot_left + plot_right) / 2, bottom + 40,
             f'Δ {panel_title}: negative favors our existing measure', 'small', text_anchor='middle')
        y = bottom + 82
    footnotes = [
        'Each APSI difference uses the same scored subset for both systems; different rows can have different coverage.',
        'APSI overall populism scores are compared with component labels without selecting hypothesis subsets.',
        'EU macro-F1 uses matched-sample leave-one-speaker-out thresholds. Its intervals condition on fixed predictions.',
        'Bootstrap units follow the original evaluation. * Fewer than 90% of requested bootstrap draws were estimable.',
        'Native APSI truncates each premise–hypothesis pair to the model limit. Base and chunk sensitivities are in the result tables.',
    ]
    for i, note in enumerate(footnotes):
        text(35, y + 6 + i * 20, note, 'small')
    actual_height = y + 120
    parts[0] = parts[0].replace(f'height="{height}"', f'height="{actual_height}"').replace(f'0 0 {width} {height}', f'0 0 {width} {actual_height}')
    parts.append('</svg>')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(parts) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('comparisons', type=Path)
    parser.add_argument('--output', type=Path, help='Defaults beside comparisons.csv')
    args = parser.parse_args()
    with args.comparisons.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    output = args.output or args.comparisons.with_name('paired_comparisons.svg')
    render(rows, output)
    write_run_manifest(output.parent, script='scripts/plot_apsi_comparison.py',
                       inputs={'comparisons': args.comparisons, 'plot_script': Path(__file__)},
                       extra={'output': str(output), 'format': 'standalone SVG',
                              'panels': [title for title, _, _ in PANELS],
                              'reference': 'ours', 'systems': [s for s, _, _ in SYSTEMS]},
                       filename='plot_run_manifest.json')
    print(output)


if __name__ == '__main__':
    main()
