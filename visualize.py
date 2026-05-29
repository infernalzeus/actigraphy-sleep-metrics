import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from datetime import datetime

def activityHeatmap(mat_df, title_str="Activity Heatmap", dates=None):
    # Produces a heatmap of mean-hourly SVM activity with noon on the left
    # and midnight centred, matching the convention used in the R visualisation.
    #
    # mat_df : numpy array [days x 24] of mean hourly SVM_sum (Activity column)
    # dates  : optional sequence of date labels for the Y-axis (length = days)
    #
    # Layout:  columns are rolled right by 12 so that hour 12 (noon) becomes
    # column 0 and hour 0 (midnight) falls in the middle of the x-axis:
    #   display col 0  = data hour 12  (noon)
    #   display col 12 = data hour  0  (midnight)
    #   display col 23 = data hour 11  (11:00)
    # This matches the R implementation:
    #   mat.temp <- cbind(mat.df, mat.df)[ , seq(13, 36) ]
    #
    # Colour scale:
    #   robust=True clips vmin/vmax at the 2nd and 98th percentiles of the
    #   non-NaN data, so a small number of very-high-activity spikes do not
    #   compress the rest of the colormap into a single dark band.
    #
    # NaN cells (hours not recorded on the first or last day):
    #   masked and drawn in light grey so they are visually distinct from
    #   genuine zero-activity / sleep periods.

    days = mat_df.shape[0]

    # roll columns right by 12: hour 12 moves to column 0
    mat_temp = np.roll(mat_df, shift=12, axis=1)

    # x-axis labels: col 0 = hour 12, col 1 = hour 13, ..., col 12 = hour 0, ...
    x_labels = [str((h + 12) % 24) for h in range(24)]

    # NaN mask — partial first/last recording days where no data was collected
    nan_mask = np.isnan(mat_temp)

    # figure height scales with number of days so labels don't overlap
    fig_h = max(4, days * 0.38)
    fig, ax = plt.subplots(figsize=(12, fig_h))

    # light grey background so masked (NaN) cells stand out from dark sleep cells
    ax.set_facecolor('#cccccc')

    sns.heatmap(
        mat_temp,
        mask=nan_mask,
        cmap='viridis',
        robust=True,          # vmin/vmax set at 2nd/98th percentile — prevents
                              # a few high-activity spikes from flattening contrast
        cbar_kws={'label': 'Activity (mean SVM / hour)'},
        ax=ax,
        linewidths=0,
        linecolor='none',
    )

    ax.set_xticks(np.arange(24) + 0.5)
    ax.set_xticklabels(x_labels, rotation=0, fontsize=8)
    ax.set_xlabel("Hour of Day")

    # Y-axis: use actual dates when supplied, otherwise fall back to day numbers
    y_labels = [str(d) for d in dates] if dates is not None else [str(i + 1) for i in range(days)]
    ax.set_yticks(np.arange(days) + 0.5)
    ax.set_yticklabels(y_labels, rotation=0, fontsize=8)
    ax.set_ylabel("Date")

    ax.set_title(title_str)
    fig.tight_layout()

    return fig


def generate_pdf_report(pdf_path, mat, dates, fname, results, pgrams,
                        daily_results=None):
    """
    Write a multi-page PDF report:
      p1      — activity heatmap
      p2      — non-parametric scalar metrics table  (when results is non-empty)
      p3+     — per-day M10 / L5 table, paginated    (when daily_results is non-empty)
      p(n)    — Enright periodogram plot             (when 'enright' in pgrams)
      p(n+1)  — Chi-square periodogram plot          (when 'chisquare' in pgrams)
    """
    with PdfPages(pdf_path) as pdf:

        # ── page 1: activity heatmap ─────────────────────────────────────────
        fig = activityHeatmap(mat, title_str=f"Activity Heatmap\n{fname}", dates=dates)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # ── page 2: non-parametric metrics table ─────────────────────────────
        if results:
            _metric_labels = {
                'IS':               'Interdaily Stability (IS)',
                'IV':               'Intradaily Variability (IV)',
                'M10_hour':         'M10 onset hour',
                'M10_mean_activity':'M10 mean activity',
                'L5_hour':          'L5 onset hour',
                'L5_mean_activity': 'L5 mean activity',
            }
            rows = []
            for k, v in results.items():
                label = _metric_labels.get(k, k)
                val   = f"{v:.4f}" if isinstance(v, float) else str(int(v))
                rows.append([label, val])

            fig, ax = plt.subplots(figsize=(8, 0.5 + 0.45 * len(rows)))
            ax.axis('off')
            tbl = ax.table(
                cellText=rows,
                colLabels=['Metric', 'Value'],
                loc='center',
                cellLoc='center',
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(11)
            tbl.scale(1, 1.8)
            ax.set_title('Non-Parametric Metrics', fontsize=13, pad=16)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        # ── page 3+: per-day M10 / L5 table (paginated) ─────────────────────
        if daily_results:
            _daily_col_labels = ['Date',
                                 'M10 Onset', 'M10 Mean Activity',
                                 'L5 Onset',  'L5 Mean Activity']

            def _fmt_hour(h):
                """Format a clock-hour as HH:00, or — for missing values."""
                if h is None or (isinstance(h, float) and np.isnan(h)):
                    return '—'          # em dash
                return f"{int(h):02d}:00"

            def _fmt_act(a):
                """Format an activity value to 1 d.p., or — for missing."""
                if a is None or (isinstance(a, float) and np.isnan(a)):
                    return '—'
                return f"{a:.1f}"

            # build display rows
            display_rows = []
            for r in daily_results:
                display_rows.append([
                    str(r['date']),
                    _fmt_hour(r.get('M10_hour')),
                    _fmt_act(r.get('M10_mean_activity')),
                    _fmt_hour(r.get('L5_hour')),
                    _fmt_act(r.get('L5_mean_activity')),
                ])

            # paginate: up to ROWS_PER_PAGE rows per PDF page
            ROWS_PER_PAGE = 30
            total_pages   = int(np.ceil(len(display_rows) / ROWS_PER_PAGE))

            for page_idx in range(total_pages):
                chunk = display_rows[page_idx * ROWS_PER_PAGE :
                                     (page_idx + 1) * ROWS_PER_PAGE]
                n_rows = len(chunk)

                fig_h = max(3, 0.42 * n_rows + 1.4)
                fig, ax = plt.subplots(figsize=(9, fig_h))
                ax.axis('off')

                tbl = ax.table(
                    cellText=chunk,
                    colLabels=_daily_col_labels,
                    loc='center',
                    cellLoc='center',
                )
                tbl.auto_set_font_size(False)
                tbl.set_fontsize(9)
                tbl.scale(1, 1.6)

                # shade header row
                for col_idx in range(len(_daily_col_labels)):
                    tbl[(0, col_idx)].set_facecolor('#ddeeff')

                page_note = (f'  (page {page_idx + 1} of {total_pages})'
                             if total_pages > 1 else '')
                ax.set_title(
                    f'Per-Day M10 / L5  —  noon-to-noon window{page_note}',
                    fontsize=12, pad=14,
                )
                fig.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

        # ── page (n): Enright periodogram ────────────────────────────────────
        if 'enright' in pgrams:
            pg = pgrams['enright']
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(pg['Period'], pg['Value'], 'b-o', markersize=4, linewidth=1.2)
            ax.axvline(24, color='r', linestyle='--', alpha=0.6, label='24 h')
            ax.set_xlabel('Period (hours)')
            ax.set_ylabel('Amplitude  $A_p$')
            ax.set_title('Enright Periodogram')
            ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
            ax.legend()
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        # ── page (n+1): chi-square periodogram ───────────────────────────────
        if 'chisquare' in pgrams:
            pg = pgrams['chisquare']
            fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

            axes[0].plot(pg['Period'], pg['Value'], 'g-o', markersize=4, linewidth=1.2)
            axes[0].axvline(24, color='r', linestyle='--', alpha=0.6, label='24 h')
            axes[0].set_ylabel('$Q_p$ statistic')
            axes[0].set_title('Chi-Square Periodogram')
            axes[0].legend()

            axes[1].plot(pg['Period'], pg['log_Pvalue'], 'm-o', markersize=4, linewidth=1.2)
            axes[1].axvline(24, color='r', linestyle='--', alpha=0.6)
            axes[1].set_xlabel('Period (hours)')
            axes[1].set_ylabel('log  $p$-value')
            axes[1].set_title('Chi-Square Periodogram — log($p$-value)')

            axes[1].xaxis.set_major_locator(mticker.MultipleLocator(2))
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        # stamp metadata into PDF info dict
        pdf.infodict().update({
            'Title':   f'Step 2 Report — {fname}',
            'Author':  'actigraphy-sleep-metrics',
            'CreationDate': datetime.now(),
        })
