"""
CTIA TRP Calculator v5
- Light theme, clean UI
- Log removed from main window
- 3D normalized EIRP pattern window added
"""

import sys, math, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QTextEdit, QFrame,
    QSizePolicy, QMessageBox, QDialog, QComboBox, QSlider
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QColor, QPalette

# -- Colors -------------------------------------------------------------------
BG       = "#f5f6fa"
PANEL    = "#ffffff"
CARD     = "#ffffff"
BDR      = "#dde1ea"
BDR2     = "#c5cad6"
BLUE     = "#2563eb"
BLUE_LT  = "#eff4ff"
GREEN    = "#16a34a"
GREEN_LT = "#f0fdf4"
WARN     = "#d97706"
TXT      = "#1e293b"
TXT2     = "#64748b"
TXT3     = "#94a3b8"

QSS = f"""
QMainWindow, QWidget {{
    background: {BG};
    color: {TXT};
    font-family: 'Segoe UI','PingFang SC','Microsoft YaHei',sans-serif;
    font-size: 13px;
}}
QFrame#card {{
    background: {CARD};
    border: 1px solid {BDR};
    border-radius: 10px;
}}
QFrame#result_card {{
    background: {BLUE_LT};
    border: 1.5px solid {BLUE};
    border-radius: 12px;
}}
QDialog {{
    background: {BG};
}}
QPushButton {{
    background: {PANEL};
    color: {TXT};
    border: 1px solid {BDR2};
    border-radius: 7px;
    padding: 8px 18px;
    font-size: 13px;
}}
QPushButton:hover {{
    background: {BLUE_LT};
    border-color: {BLUE};
    color: {BLUE};
}}
QPushButton:pressed {{
    background: {BLUE};
    color: #ffffff;
}}
QPushButton#pri {{
    background: {BLUE};
    color: #ffffff;
    border: none;
    font-weight: bold;
}}
QPushButton#pri:hover {{ background: #1d4ed8; }}
QPushButton#pri:pressed {{ background: #1e40af; }}
QPushButton#pri:disabled {{ background: {BDR}; color: {TXT3}; }}
QPushButton#export_btn {{
    background: {GREEN_LT};
    color: {GREEN};
    border: 1px solid #86efac;
}}
QPushButton#export_btn:hover {{
    background: {GREEN};
    color: #ffffff;
    border-color: {GREEN};
}}
QPushButton#pattern_btn {{
    background: #faf5ff;
    color: #7c3aed;
    border: 1px solid #c4b5fd;
}}
QPushButton#pattern_btn:hover {{
    background: #7c3aed;
    color: #ffffff;
    border-color: #7c3aed;
}}
QPushButton#log_btn {{
    background: #fff7ed;
    color: #c2410c;
    border: 1px solid #fed7aa;
}}
QPushButton#log_btn:hover {{
    background: #ea580c;
    color: #ffffff;
    border-color: #ea580c;
}}
QTextEdit {{
    background: #f8fafc;
    color: #334155;
    border: 1px solid {BDR};
    border-radius: 8px;
    font-family: 'Consolas','Courier New',monospace;
    font-size: 12px;
    padding: 10px;
}}
QLabel#trp_val {{
    color: {BLUE};
    font-size: 36px;
    font-weight: bold;
}}
QLabel#trp_unit {{
    color: {TXT2};
    font-size: 15px;
}}
QLabel#section {{
    color: {TXT2};
    font-size: 11px;
    font-weight: bold;
    letter-spacing: 1px;
}}
QLabel#kk {{ color: {TXT2}; font-size: 12px; }}
QLabel#kv {{ color: {TXT}; font-size: 12px; font-weight: bold; }}
QLabel#stat_val {{ font-size: 16px; font-weight: bold; }}
QLabel#stat_lbl {{ color: {TXT3}; font-size: 10px; }}
QComboBox {{
    background: {PANEL};
    color: {TXT};
    border: 1px solid {BDR2};
    border-radius: 6px;
    padding: 4px 10px;
    min-width: 120px;
}}
QComboBox::drop-down {{ border: none; width: 20px; }}
QScrollBar:vertical {{
    background: {BG};
    width: 7px;
    border-radius: 3px;
}}
QScrollBar::handle:vertical {{
    background: {BDR2};
    border-radius: 3px;
    min-height: 24px;
}}
"""


# -- CSV Parser ---------------------------------------------------------------
def parse_csv(path: str) -> pd.DataFrame:
    with open(path, 'r', encoding='utf-8-sig') as f:
        content = f.read()
    first_line = content.strip().split('\n')[0]
    sep = '\t' if '\t' in first_line else ','
    df_raw = pd.read_csv(path, sep=sep, header=0, index_col=None,
                         dtype=str, encoding='utf-8-sig')
    df_raw.columns = [str(c).strip() for c in df_raw.columns]
    col_lower = {c.strip().lower(): c for c in df_raw.columns}
    has_phi  = any(k in col_lower for k in ('phi','az','azimuth'))
    has_th   = any(k in col_lower for k in ('theta','el','elevation'))
    has_eirp = any(k in col_lower for k in ('eirp','eirp_dbm','power','value'))
    if has_phi and has_th and has_eirp:
        phi_c = next(col_lower[k] for k in ('phi','az','azimuth') if k in col_lower)
        th_c  = next(col_lower[k] for k in ('theta','el','elevation') if k in col_lower)
        ei_c  = next(col_lower[k] for k in ('eirp','eirp_dbm','power','value') if k in col_lower)
        out = df_raw[[phi_c, th_c, ei_c]].copy()
        out.columns = ['phi','theta','eirp']
        return out.astype(float)
    phi_cols = {}
    for col in df_raw.columns[1:]:
        try:
            phi_cols[col] = float(col.strip())
        except ValueError:
            pass
    if not phi_cols:
        raise ValueError(
            f"Cannot parse CSV. Columns: {list(df_raw.columns)}\n"
            "Supported: transposed format (phi angles as headers) "
            "or long table (phi/theta/eirp columns).")
    records = []
    for _, row in df_raw.iterrows():
        try:
            theta_val = float(str(row.iloc[0]).strip())
        except ValueError:
            continue
        for col_name, phi_val in phi_cols.items():
            try:
                records.append({'phi': phi_val, 'theta': theta_val,
                                 'eirp': float(str(row[col_name]).strip())})
            except ValueError:
                pass
    if not records:
        raise ValueError("No valid data points found.")
    return pd.DataFrame(records)


# -- TRP Calculation ----------------------------------------------------------
def calc_trp(df: pd.DataFrame) -> dict:
    df_use = df[df['phi'] < 180.0].copy()
    phis   = sorted(df_use['phi'].unique())
    thetas = sorted(df_use['theta'].unique())
    if len(phis) < 2 or len(thetas) < 2:
        raise ValueError("Not enough data after removing phi=180.")
    dphi_deg   = float(np.mean(np.diff(phis)))
    dtheta_deg = float(np.mean(np.diff(thetas)))
    K = math.radians(dphi_deg) * math.radians(dtheta_deg) / (4.0 * math.pi)
    pivot = df_use.pivot_table(values='eirp', index='theta',
                               columns='phi', aggfunc='mean')
    weighted_sum = 0.0
    rows_log = []
    for theta_val in thetas:
        weight  = math.sin(math.radians(abs(theta_val)))
        row_sum = sum(10.0**(float(pivot.loc[theta_val, p])/10.0) * weight
                      for p in phis)
        weighted_sum += row_sum
        rows_log.append({'theta': theta_val, 'sin_abs': weight,
                         'row_sum_mw': row_sum, 'contrib_mw': row_sum * K})
    trp_mw  = weighted_sum * K
    trp_dbm = 10.0 * math.log10(trp_mw) if trp_mw > 0 else float('-inf')

    # Method B: all 13 phi cols including phi=180
    phis_full  = sorted(df['phi'].unique())
    pivot_full = df.pivot_table(values='eirp', index='theta',
                                columns='phi', aggfunc='mean')
    S_full = sum(
        10.0 ** (float(pivot_full.loc[t, p]) / 10.0) * math.sin(math.radians(abs(t)))
        for t in thetas for p in phis_full
    )
    trp_full_mw  = S_full * K
    trp_full_dbm = 10.0 * math.log10(trp_full_mw) if trp_full_mw > 0 else float('-inf')

    # Method C & D: average phi=0 and phi=180, replace phi=0, use 12 cols
    has_180 = 180.0 in df['phi'].values
    if has_180:
        p0_s   = {row['theta']: row['eirp']
                  for _, row in df[df['phi'] == 0.0].iterrows()}
        p180_s = {row['theta']: row['eirp']
                  for _, row in df[df['phi'] == 180.0].iterrows()}

        df_c = df_use.copy()
        df_d = df_use.copy()
        for t in thetas:
            if t in p0_s and t in p180_s:
                lin0, lin180 = 10.0**(p0_s[t]/10.0), 10.0**(p180_s[t]/10.0)
                avg_lin_dbm = 10.0 * math.log10((lin0 + lin180) / 2.0)
                avg_dbm_dbm = (p0_s[t] + p180_s[t]) / 2.0
                mask = (df_c['theta'] == t) & (df_c['phi'] == 0.0)
                df_c.loc[mask, 'eirp'] = avg_lin_dbm
                df_d.loc[mask, 'eirp'] = avg_dbm_dbm

        def _trp(df_x):
            pv = df_x.pivot_table(values='eirp', index='theta',
                                  columns='phi', aggfunc='mean')
            S  = sum(10.0**(float(pv.loc[t,p])/10.0) * math.sin(math.radians(abs(t)))
                     for t in thetas for p in phis)
            return 10.0 * math.log10(S * K) if S > 0 else float('-inf')

        trp_avg_lin_dbm = _trp(df_c)
        trp_avg_dbm_dbm = _trp(df_d)
    else:
        trp_avg_lin_dbm = trp_dbm
        trp_avg_dbm_dbm = trp_dbm

    return {
        'trp_dbm': trp_dbm, 'trp_mw': trp_mw,
        'trp_full_dbm': trp_full_dbm, 'trp_full_mw': trp_full_mw,
        'trp_avg_lin_dbm': trp_avg_lin_dbm,
        'trp_avg_dbm_dbm': trp_avg_dbm_dbm,
        'weighted_sum': weighted_sum, 'K': K,
        'dphi_deg': dphi_deg, 'dtheta_deg': dtheta_deg,
        'n_phi': len(phis), 'n_theta': len(thetas),
        'n_raw': len(df), 'n_used': len(df_use),
        'phi_min': min(phis), 'phi_max': max(phis),
        'theta_min': min(thetas), 'theta_max': max(thetas),
        'eirp_max': df_use['eirp'].max(),
        'eirp_min': df_use['eirp'].min(),
        'eirp_mean': df_use['eirp'].mean(),
        'rows_log': rows_log, 'pivot': pivot,
        'df_use': df_use,
    }


# -- Log Text -----------------------------------------------------------------
def build_log(r: dict) -> str:
    lines = []
    a = lines.append
    a("=" * 58)
    a("  CTIA TRP Calculation Report  -  MVG StarLab Compatible")
    a("=" * 58)
    a("")
    a("Integration Rules")
    a(f"  phi  : phi=180 removed  |  range {r['phi_min']:.0f}~{r['phi_max']:.0f} deg  |  {r['n_phi']} cuts")
    a(f"  theta: all retained  |  range {r['theta_min']:.2f}~{r['theta_max']:.2f} deg  |  {r['n_theta']} steps")
    a(f"  weight: sin(|theta|)")
    a(f"  data points used: {r['n_used']}  (raw: {r['n_raw']})")
    a("")
    a("Formula")
    a("  TRP = dphi[rad] x dtheta[rad] / (4*pi)")
    a("        x SUM[phi][theta] EIRP_lin x sin(|theta|)")
    a("")
    a("Data Summary")
    a(f"  EIRP max  : {r['eirp_max']:.3f} dBm")
    a(f"  EIRP min  : {r['eirp_min']:.3f} dBm")
    a(f"  EIRP mean : {r['eirp_mean']:.3f} dBm")
    a("")
    a("K Factor")
    a(f"  dphi   = {r['dphi_deg']:.4f} deg = {math.radians(r['dphi_deg']):.8f} rad")
    a(f"  dtheta = {r['dtheta_deg']:.4f} deg = {math.radians(r['dtheta_deg']):.8f} rad")
    a(f"  K      = {r['K']:.10f}")
    a("")
    a("Theta Layer Detail")
    a(f"  {'theta':>10}  {'sin|theta|':>12}  {'sum phi(mW)':>16}  {'contrib(mW)':>14}")
    a("  " + "-" * 58)
    for row in r['rows_log']:
        a(f"  {row['theta']:>10.2f}  {row['sin_abs']:>12.4f}"
          f"  {row['row_sum_mw']:>16.4f}  {row['contrib_mw']:>14.6f}")
    a("  " + "-" * 58)
    a("")
    a("Steps")
    a(f"  1. EIRP_lin = 10^(EIRP_dBm / 10)  [mW]")
    a(f"  2. S = {r['weighted_sum']:.6f} mW*sr")
    a(f"  3. TRP = S x K = {r['trp_mw']:.8f} mW")
    a(f"  4. TRP = {r['trp_dbm']:.6f} dBm")
    a("")
    a("=" * 58)
    a(f"  TRP = {r['trp_dbm']:.4f} dBm  |  {r['trp_mw']:.6f} mW")
    a("=" * 58)
    return "\n".join(lines)


# -- 3D Pattern Window --------------------------------------------------------
class Pattern3DWindow(QDialog):
    def __init__(self, result: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("3D Normalized EIRP Radiation Pattern")
        self.resize(860, 700)
        self.setStyleSheet(f"QDialog {{ background: {BG}; }}")
        self.result = result
        self._build(result)

    def _build(self, r: dict):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Controls bar
        ctrl = QHBoxLayout()
        ctrl.setSpacing(10)

        lbl_cmap = QLabel("Colormap:")
        lbl_cmap.setStyleSheet(f"color:{TXT2}; font-size:12px;")
        self.cmb_cmap = QComboBox()
        self.cmb_cmap.addItems(["plasma", "jet", "RdYlBu_r", "viridis", "hot", "coolwarm"])
        self.cmb_cmap.currentTextChanged.connect(self._replot)

        lbl_style = QLabel("Style:")
        lbl_style.setStyleSheet(f"color:{TXT2}; font-size:12px;")
        self.cmb_style = QComboBox()
        self.cmb_style.addItems(["Surface", "Wireframe", "Surface + Wireframe"])
        self.cmb_style.currentTextChanged.connect(self._replot)

        lbl_scale = QLabel("Radius scale:")
        lbl_scale.setStyleSheet(f"color:{TXT2}; font-size:12px;")
        self.cmb_scale = QComboBox()
        self.cmb_scale.addItems(["Linear (mW)", "dB normalized"])
        self.cmb_scale.currentTextChanged.connect(self._replot)

        btn_refresh = QPushButton("Refresh")
        btn_refresh.clicked.connect(self._replot)

        for w in [lbl_cmap, self.cmb_cmap, lbl_style, self.cmb_style,
                  lbl_scale, self.cmb_scale, btn_refresh]:
            ctrl.addWidget(w)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        # Canvas
        canvas_frame = QFrame(); canvas_frame.setObjectName("card")
        cf = QVBoxLayout(canvas_frame); cf.setContentsMargins(4,4,4,4)
        self.fig3d = Figure(figsize=(10, 8), facecolor='#ffffff')
        self.canvas3d = FigureCanvas(self.fig3d)
        self.canvas3d.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        cf.addWidget(self.canvas3d)
        layout.addWidget(canvas_frame)

        # Info bar
        info = QLabel(
            f"TRP = {r['trp_dbm']:.4f} dBm   |   "
            f"EIRP max = {r['eirp_max']:.2f} dBm   |   "
            f"EIRP min = {r['eirp_min']:.2f} dBm   |   "
            f"phi: {r['phi_min']:.0f}~{r['phi_max']:.0f} deg ({r['n_phi']} cuts)   |   "
            f"theta: {r['theta_min']:.1f}~{r['theta_max']:.1f} deg ({r['n_theta']} steps)"
        )
        info.setStyleSheet(f"color:{TXT2}; font-size:11px; padding:4px;")
        info.setAlignment(Qt.AlignCenter)
        layout.addWidget(info)

        self._replot()

    def _build_grid(self):
        r = self.result
        pivot = r['pivot']
        thetas_raw = sorted(pivot.index.tolist())
        phis_raw   = sorted(pivot.columns.tolist())

        # Build interpolation point cloud
        pts, vals = [], []
        for t in thetas_raw:
            for p in phis_raw:
                pts.append((p, t))
                vals.append(float(pivot.loc[t, p]))
        pts  = np.array(pts)
        vals = np.array(vals)

        # Dense grid for smooth surface
        phi_g   = np.linspace(min(phis_raw),   max(phis_raw),   72)
        theta_g = np.linspace(min(thetas_raw), max(thetas_raw), 120)
        PHI, THETA = np.meshgrid(phi_g, theta_g)

        EIRP = griddata(pts, vals, (PHI, THETA), method='linear')
        EIRP = np.nan_to_num(EIRP, nan=float(np.nanmin(vals)))

        return PHI, THETA, EIRP

    def _replot(self):
        r      = self.result
        cmap   = self.cmb_cmap.currentText()
        style  = self.cmb_style.currentText()
        scale  = self.cmb_scale.currentText()

        PHI, THETA, EIRP = self._build_grid()

        # Radius
        if "dB" in scale:
            eirp_min = EIRP.min()
            eirp_max = EIRP.max()
            # normalize 0~1 in dB domain, then convert to 0~1 linear radius
            R = (EIRP - eirp_min) / (eirp_max - eirp_min + 1e-9)
            R = np.clip(R, 0, 1)
        else:
            # Linear mW, normalized to 0~1
            EIRP_lin = 10.0 ** (EIRP / 10.0)
            R = EIRP_lin / EIRP_lin.max()
            R = np.clip(R, 0, 1)

        # Convert to Cartesian
        # phi = azimuth cut (0~165), theta = elevation angle
        # x = r * cos(elev) * cos(phi)
        # y = r * cos(elev) * sin(phi)
        # z = r * sin(elev)
        THETA_rad = np.radians(THETA)
        PHI_rad   = np.radians(PHI)
        X = R * np.cos(THETA_rad) * np.cos(PHI_rad)
        Y = R * np.cos(THETA_rad) * np.sin(PHI_rad)
        Z = R * np.sin(THETA_rad)

        # Color mapping
        norm_c = (EIRP - EIRP.min()) / (EIRP.max() - EIRP.min() + 1e-9)

        self.fig3d.clear()
        ax = self.fig3d.add_subplot(111, projection='3d',
                                    facecolor='#f8fafc')

        if style == "Surface":
            surf = ax.plot_surface(
                X, Y, Z,
                facecolors=matplotlib.colormaps[cmap](norm_c),
                alpha=0.92, linewidth=0, antialiased=True,
                shade=True)
        elif style == "Wireframe":
            ax.plot_wireframe(X, Y, Z, color='#2563eb',
                              linewidth=0.5, alpha=0.7,
                              rstride=4, cstride=4)
        else:
            surf = ax.plot_surface(
                X, Y, Z,
                facecolors=matplotlib.colormaps[cmap](norm_c),
                alpha=0.85, linewidth=0, antialiased=True, shade=True)
            ax.plot_wireframe(X, Y, Z, color='#1e293b',
                              linewidth=0.25, alpha=0.2,
                              rstride=6, cstride=6)

        # Colorbar via ScalarMappable
        sm = matplotlib.cm.ScalarMappable(
            cmap=matplotlib.colormaps[cmap],
            norm=matplotlib.colors.Normalize(
                vmin=float(EIRP.min()), vmax=float(EIRP.max())))
        sm.set_array([])
        cb = self.fig3d.colorbar(sm, ax=ax, shrink=0.55, aspect=14,
                                  pad=0.08, fraction=0.03)
        cb.set_label('EIRP (dBm)', fontsize=9, color='#475569')
        cb.ax.tick_params(labelsize=8, colors='#64748b')

        # Reference axes (unit sphere outlines)
        u = np.linspace(0, 2*np.pi, 80)
        ax.plot(0.05*np.cos(u), 0.05*np.sin(u),
                np.zeros_like(u), color='#94a3b8',
                lw=0.6, alpha=0.4, zorder=1)

        # Axes labels and style
        ax.set_xlabel('X', fontsize=9, color='#475569', labelpad=6)
        ax.set_ylabel('Y', fontsize=9, color='#475569', labelpad=6)
        ax.set_zlabel('Z', fontsize=9, color='#475569', labelpad=6)
        ax.tick_params(labelsize=7, colors='#64748b')
        ax.set_title(
            f"Normalized EIRP 3D Pattern\n"
            f"Radius = {'dB-normalized' if 'dB' in scale else 'linear (mW), normalized'}   "
            f"|   TRP = {r['trp_dbm']:.4f} dBm",
            fontsize=11, color='#1e293b', fontweight='bold', pad=10)

        # Grid style
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('#dde1ea')
        ax.yaxis.pane.set_edgecolor('#dde1ea')
        ax.zaxis.pane.set_edgecolor('#dde1ea')
        ax.grid(True, color='#e2e8f0', linewidth=0.5, alpha=0.7)

        self.fig3d.tight_layout()
        self.canvas3d.draw()


# -- Log Window ---------------------------------------------------------------
class LogWindow(QDialog):
    def __init__(self, log_text: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calculation Log")
        self.resize(680, 580)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        txt = QTextEdit(); txt.setReadOnly(True)
        txt.setPlainText(log_text)
        layout.addWidget(txt)
        btn = QPushButton("Close")
        btn.clicked.connect(self.close)
        layout.addWidget(btn)


# -- 2D Charts Canvas ---------------------------------------------------------
class Canvas2D(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(10, 4.2), facecolor='#ffffff')
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._empty()

    def _empty(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111, facecolor='#f8fafc')
        ax.text(0.5, 0.5, "Load a CSV and click  Calculate TRP",
                ha='center', va='center', color='#94a3b8',
                fontsize=12, transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_edgecolor('#dde1ea')
        self.fig.tight_layout()
        self.draw()

    def plot(self, r: dict):
        self.fig.clear()
        self.fig.patch.set_facecolor('#ffffff')
        gs = gridspec.GridSpec(1, 2, figure=self.fig,
                               wspace=0.38, left=0.07, right=0.97,
                               top=0.86, bottom=0.14)
        pivot  = r['pivot']
        txt_kw = dict(color='#475569', fontsize=9)

        # Heatmap
        ax1 = self.fig.add_subplot(gs[0, 0], facecolor='#f8fafc')
        im  = ax1.imshow(
            pivot.values, aspect='auto', origin='lower', cmap='RdYlBu_r',
            extent=[pivot.columns.min(), pivot.columns.max(),
                    pivot.index.min(),   pivot.index.max()])
        cb = self.fig.colorbar(im, ax=ax1, pad=0.03, fraction=0.046)
        cb.set_label('EIRP (dBm)', color='#475569', fontsize=8)
        cb.ax.yaxis.set_tick_params(labelsize=7, labelcolor='#475569')
        ax1.set_xlabel('phi (deg)',   **txt_kw)
        ax1.set_ylabel('theta (deg)', **txt_kw)
        ax1.set_title('EIRP Distribution  (phi=180 removed)',
                      color='#1e293b', fontsize=10, pad=7, fontweight='bold')
        ax1.tick_params(colors='#64748b', labelsize=8)
        ax1.axhline(0,   color='#f59e0b', lw=1.0, ls='--', alpha=0.7)
        ax1.axhline(90,  color='#10b981', lw=0.7, ls=':',  alpha=0.6)
        ax1.axhline(-90, color='#10b981', lw=0.7, ls=':',  alpha=0.6)
        for sp in ax1.spines.values(): sp.set_edgecolor('#dde1ea')

        # Contribution bar
        ax2 = self.fig.add_subplot(gs[0, 1], facecolor='#f8fafc')
        rows   = r['rows_log']
        xs     = [row['theta']      for row in rows]
        ys     = [row['contrib_mw'] * 1000 for row in rows]
        max_y  = max(ys) if ys else 1
        bcolors = [matplotlib.colormaps['Blues'](0.3 + 0.65*(y/max_y)) for y in ys]
        ax2.bar(xs, ys, width=r['dtheta_deg']*0.72,
                color=bcolors, edgecolor='#cbd5e1', linewidth=0.4)
        ax2.set_xlabel('theta (deg)',       **txt_kw)
        ax2.set_ylabel('Contribution (uW)', **txt_kw)
        ax2.set_title('TRP Contribution per theta Layer',
                      color='#1e293b', fontsize=10, pad=7, fontweight='bold')
        ax2.tick_params(colors='#64748b', labelsize=8)
        ax2.axvline(0,   color='#f59e0b', lw=1.0, ls='--', alpha=0.7)
        ax2.axvline(90,  color='#10b981', lw=0.7, ls=':',  alpha=0.5)
        ax2.axvline(-90, color='#10b981', lw=0.7, ls=':',  alpha=0.5)
        ax2.set_facecolor('#f8fafc')
        for sp in ax2.spines.values(): sp.set_edgecolor('#dde1ea')

        self.fig.suptitle(
            f"TRP = {r['trp_dbm']:.4f} dBm   |   {r['trp_mw']:.4f} mW   |   "
            f"phi {r['n_phi']} cuts x theta {r['n_theta']} pts = {r['n_used']} points",
            color='#1e293b', fontsize=10, fontweight='bold')
        self.draw()


# -- Worker Thread ------------------------------------------------------------
class Worker(QThread):
    done  = pyqtSignal(dict)
    error = pyqtSignal(str)
    def __init__(self, df): super().__init__(); self.df = df
    def run(self):
        try:    self.done.emit(calc_trp(self.df))
        except Exception as e: self.error.emit(str(e))


# -- Helpers ------------------------------------------------------------------
def hline():
    f = QFrame(); f.setFrameShape(QFrame.HLine)
    f.setStyleSheet(f"color:{BDR};"); return f


# -- Main Window --------------------------------------------------------------
class MainWin(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TRP Calculator  -  MVG StarLab / AMS8800")
        self.resize(1200, 740)
        self.df      = None
        self._result = None
        self._log_win    = None
        self._pattern_win = None
        self._build_ui()

    def _build_ui(self):
        root = QWidget(); self.setCentralWidget(root)
        main = QHBoxLayout(root)
        main.setSpacing(12); main.setContentsMargins(14, 14, 14, 14)

        # ==================== Left panel =====================================
        left = QWidget(); left.setFixedWidth(300)
        lv = QVBoxLayout(left)
        lv.setSpacing(10); lv.setContentsMargins(0, 0, 0, 0)

        hdr = QLabel("TRP Calculator")
        hdr.setStyleSheet(f"color:{TXT}; font-size:18px; font-weight:bold; padding:2px 0")
        sub = QLabel("MVG StarLab  /  AMS8800  /  CTIA")
        sub.setStyleSheet(f"color:{TXT2}; font-size:11px;")
        lv.addWidget(hdr); lv.addWidget(sub)
        lv.addWidget(hline())

        # File
        fc = QFrame(); fc.setObjectName("card")
        fv = QVBoxLayout(fc); fv.setSpacing(8); fv.setContentsMargins(14,12,14,12)
        s1 = QLabel("CSV FILE"); s1.setObjectName("section"); fv.addWidget(s1)
        self.lbl_file = QLabel("No file loaded")
        self.lbl_file.setWordWrap(True)
        self.lbl_file.setStyleSheet(
            f"color:{TXT3}; font-size:11px; background:#f8fafc;"
            f"border:1px dashed {BDR2}; border-radius:6px; padding:8px;")
        fv.addWidget(self.lbl_file)
        btn_load = QPushButton("  Load CSV File")
        btn_load.setObjectName("pri"); btn_load.setFixedHeight(36)
        btn_load.clicked.connect(self._load)
        fv.addWidget(btn_load)
        lv.addWidget(fc)

        # Formula
        mc = QFrame(); mc.setObjectName("card")
        mv = QVBoxLayout(mc); mv.setSpacing(6); mv.setContentsMargins(14,12,14,12)
        s2 = QLabel("INTEGRATION RULES"); s2.setObjectName("section")
        mv.addWidget(s2)
        fml = QTextEdit(); fml.setReadOnly(True); fml.setFixedHeight(162)
        fml.setStyleSheet(
            f"background:#f8fafc; color:#334155; border:1px solid {BDR};"
            f"border-radius:6px; font-family:'Consolas','Courier New',monospace;"
            f"font-size:11px; padding:8px;")
        fml.setPlainText(
            "Formula:\n"
            "  TRP = K x SUM[phi][theta]\n"
            "        EIRP_lin x sin(|theta|)\n"
            "  K = dphi[rad] x dtheta[rad] / (4*pi)\n\n"
            "phi:\n"
            "  Remove phi=180 (same great circle)\n"
            "  Keep 0, 15, ..., 165  (12 cuts)\n\n"
            "theta:\n"
            "  Keep all -168.75 to +168.75\n"
            "  31 steps, no truncation\n\n"
            "Weight sin(|theta|):\n"
            "  theta=0    -> 0.000  (pole)\n"
            "  theta=+/-90 -> 1.000  (equator, max)"
        )
        mv.addWidget(fml)
        lv.addWidget(mc)

        # Matrix info
        ic = QFrame(); ic.setObjectName("card")
        iv = QVBoxLayout(ic); iv.setSpacing(5); iv.setContentsMargins(14,12,14,12)
        s3 = QLabel("TEST MATRIX"); s3.setObjectName("section"); iv.addWidget(s3)
        self._kv = {}
        for key, lbl in [
            ('phi_range',   'phi range'),
            ('theta_range', 'theta range'),
            ('n_phi',       'phi cuts'),
            ('n_theta',     'theta steps'),
            ('n_used',      'data points'),
            ('dphi',        'delta phi'),
            ('dtheta',      'delta theta'),
            ('K',           'K factor'),
        ]:
            row = QHBoxLayout(); row.setSpacing(4)
            k = QLabel(lbl + ":"); k.setObjectName("kk"); k.setFixedWidth(82)
            v = QLabel("--");      v.setObjectName("kv")
            row.addWidget(k); row.addWidget(v)
            iv.addLayout(row)
            self._kv[key] = v
        lv.addWidget(ic)

        # Action buttons
        self.btn_calc = QPushButton("  Calculate TRP")
        self.btn_calc.setObjectName("pri")
        self.btn_calc.setFixedHeight(42)
        self.btn_calc.setEnabled(False)
        self.btn_calc.clicked.connect(self._calc)
        lv.addWidget(self.btn_calc)

        btn_row = QHBoxLayout(); btn_row.setSpacing(8)
        self.btn_3d = QPushButton("  3D Pattern")
        self.btn_3d.setObjectName("pattern_btn")
        self.btn_3d.setFixedHeight(36)
        self.btn_3d.setEnabled(False)
        self.btn_3d.clicked.connect(self._show_3d)

        self.btn_log = QPushButton("  View Log")
        self.btn_log.setObjectName("log_btn")
        self.btn_log.setFixedHeight(36)
        self.btn_log.setEnabled(False)
        self.btn_log.clicked.connect(self._show_log)

        btn_row.addWidget(self.btn_3d)
        btn_row.addWidget(self.btn_log)
        lv.addLayout(btn_row)

        btn_exp = QPushButton("  Export Report")
        btn_exp.setObjectName("export_btn")
        btn_exp.setFixedHeight(36)
        btn_exp.clicked.connect(self._export)
        lv.addWidget(btn_exp)

        lv.addStretch()
        main.addWidget(left)

        # ==================== Right panel ====================================
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setSpacing(10); rv.setContentsMargins(0, 0, 0, 0)

        # Result card
        res = QFrame(); res.setObjectName("result_card"); res.setFixedHeight(168)
        res_v = QVBoxLayout(res)
        res_v.setContentsMargins(24, 8, 24, 8); res_v.setSpacing(4)

        # Top row: TRP (phi=180 removed) + stats
        rh = QHBoxLayout(); rh.setSpacing(0)
        tl = QVBoxLayout(); tl.setSpacing(1)
        lbl_head = QLabel("A: TRP  phi=0~165  (phi=180 removed)")
        lbl_head.setStyleSheet(
            f"color:{BLUE}; font-size:10px; font-weight:bold; letter-spacing:1px;")
        self.lbl_trp = QLabel("--"); self.lbl_trp.setObjectName("trp_val")
        tl.addWidget(lbl_head); tl.addWidget(self.lbl_trp)
        rh.addLayout(tl)
        ul = QVBoxLayout(); ul.setSpacing(0); ul.addStretch()
        lbl_u = QLabel("dBm"); lbl_u.setObjectName("trp_unit")
        ul.addWidget(lbl_u); ul.addStretch()
        rh.addLayout(ul); rh.addStretch()

        for attr, label, color in [
            ('lbl_max',  'MAX EIRP', WARN),
            ('lbl_min',  'MIN EIRP', TXT2),
            ('lbl_mean', 'AVG EIRP', GREEN),
        ]:
            sep = QFrame(); sep.setFrameShape(QFrame.VLine)
            sep.setStyleSheet(f"color:{BLUE};"); rh.addWidget(sep); rh.addSpacing(8)
            m = QVBoxLayout(); m.setSpacing(2)
            ml = QLabel(label); ml.setObjectName("stat_lbl")
            mv2 = QLabel("--"); mv2.setObjectName("stat_val")
            mv2.setStyleSheet(f"color:{color};")
            m.addStretch(); m.addWidget(ml); m.addWidget(mv2); m.addStretch()
            setattr(self, attr, mv2)
            rh.addLayout(m); rh.addSpacing(8)
        res_v.addLayout(rh)

        # Row 2: all comparison methods
        def _cmp_row(lbl_text, attr, color):
            row = QHBoxLayout(); row.setSpacing(6)
            lh = QLabel(lbl_text)
            lh.setStyleSheet(f"color:{TXT2}; font-size:11px;")
            lh.setFixedWidth(310)
            lv2 = QLabel("--")
            lv2.setStyleSheet(f"color:{color}; font-size:15px; font-weight:bold;")
            lu = QLabel("dBm")
            lu.setStyleSheet(f"color:{TXT2}; font-size:11px;")
            ld = QLabel("")
            ld.setStyleSheet(f"color:{TXT3}; font-size:11px;")
            row.addWidget(lh); row.addWidget(lv2); row.addWidget(lu)
            row.addSpacing(10); row.addWidget(ld); row.addStretch()
            setattr(self, attr, lv2)
            setattr(self, attr + "_diff", ld)
            return row

        res_v.addLayout(_cmp_row(
            "B: phi=0~180 all 13 cuts included:",
            "lbl_full", "#7c3aed"))
        res_v.addLayout(_cmp_row(
            "C: phi=0 = linear avg(phi=0, phi=180), 12 cuts:",
            "lbl_avg_lin", "#0891b2"))
        res_v.addLayout(_cmp_row(
            "D: phi=0 = dBm avg(phi=0, phi=180), 12 cuts:",
            "lbl_avg_dbm", "#059669"))

        rv.addWidget(res)

        # Charts only (no log)
        cf = QFrame(); cf.setObjectName("card")
        cfl = QVBoxLayout(cf); cfl.setContentsMargins(4,4,4,4)
        self.canvas = Canvas2D()
        cfl.addWidget(self.canvas)
        rv.addWidget(cf)

        main.addWidget(right, stretch=1)

    # -- Load -----------------------------------------------------------------
    def _load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select EIRP CSV", "",
            "CSV Files (*.csv *.tsv *.txt);;All Files (*)")
        if not path: return
        try:
            self.df = parse_csv(path)
            fname   = os.path.basename(path)
            n180    = int((self.df['phi'] == 180.0).sum())
            self.lbl_file.setText(
                f"{fname}\n"
                f"{len(self.df)} rows  |  phi=180: {n180} rows will be removed")
            self.lbl_file.setStyleSheet(
                f"color:{GREEN}; font-size:11px; background:{GREEN_LT};"
                f"border:1px solid #86efac; border-radius:6px; padding:8px;")
            self._update_kv()
            self.btn_calc.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Load Failed", str(e))

    def _update_kv(self):
        if self.df is None: return
        df_c   = self.df[self.df['phi'] < 180.0]
        phis   = sorted(df_c['phi'].unique())
        thetas = sorted(df_c['theta'].unique())
        dphi   = float(np.mean(np.diff(phis)))   if len(phis)>1   else 0.0
        dtheta = float(np.mean(np.diff(thetas))) if len(thetas)>1 else 0.0
        K      = math.radians(dphi) * math.radians(dtheta) / (4*math.pi)
        self._kv['phi_range'].setText(f"{min(phis):.0f} ~ {max(phis):.0f} deg")
        self._kv['theta_range'].setText(f"{min(thetas):.2f} ~ {max(thetas):.2f}")
        self._kv['n_phi'].setText(str(len(phis)))
        self._kv['n_theta'].setText(str(len(thetas)))
        self._kv['n_used'].setText(str(len(df_c)))
        self._kv['dphi'].setText(f"{dphi:.2f} deg")
        self._kv['dtheta'].setText(f"{dtheta:.4f} deg")
        self._kv['K'].setText(f"{K:.6f}")

    # -- Calculate ------------------------------------------------------------
    def _calc(self):
        if self.df is None: return
        self.btn_calc.setEnabled(False)
        self.btn_calc.setText("Calculating...")
        self._worker = Worker(self.df)
        self._worker.done.connect(self._on_done)
        self._worker.error.connect(self._on_err)
        self._worker.start()

    def _on_done(self, r):
        self._result = r
        a = r['trp_dbm']
        self.lbl_trp.setText(f"{a:.4f}")
        self.lbl_max.setText(f"{r['eirp_max']:.2f} dBm")
        self.lbl_min.setText(f"{r['eirp_min']:.2f} dBm")
        self.lbl_mean.setText(f"{r['eirp_mean']:.2f} dBm")

        def _set(attr, val):
            getattr(self, attr).setText(f"{val:.4f}")
            d = val - a
            getattr(self, attr + "_diff").setText(f"diff vs A = {d:+.4f} dB")

        _set("lbl_full",    r['trp_full_dbm'])
        _set("lbl_avg_lin", r['trp_avg_lin_dbm'])
        _set("lbl_avg_dbm", r['trp_avg_dbm_dbm'])

        self.canvas.plot(r)
        self.btn_calc.setText("  Calculate TRP")
        self.btn_calc.setEnabled(True)
        self.btn_3d.setEnabled(True)
        self.btn_log.setEnabled(True)

    def _on_err(self, msg):
        QMessageBox.critical(self, "Calculation Error", msg)
        self.btn_calc.setText("  Calculate TRP")
        self.btn_calc.setEnabled(True)

    # -- Show 3D pattern window -----------------------------------------------
    def _show_3d(self):
        if self._result is None: return
        if self._pattern_win is not None:
            self._pattern_win.close()
        self._pattern_win = Pattern3DWindow(self._result, parent=self)
        self._pattern_win.show()

    # -- Show log window ------------------------------------------------------
    def _show_log(self):
        if self._result is None: return
        if self._log_win is not None:
            self._log_win.close()
        self._log_win = LogWindow(build_log(self._result), parent=self)
        self._log_win.show()

    # -- Export ---------------------------------------------------------------
    def _export(self):
        if self._result is None:
            QMessageBox.warning(self, "Notice", "Please run calculation first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Report", "trp_report.txt", "Text Files (*.txt)")
        if not path: return
        with open(path, 'w', encoding='utf-8') as f:
            f.write(build_log(self._result))
        QMessageBox.information(self, "Saved", f"Report saved:\n{path}")


# -- Entry --------------------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    pal = QPalette()
    for role, color in [
        (QPalette.Window,          BG),
        (QPalette.WindowText,      TXT),
        (QPalette.Base,            PANEL),
        (QPalette.AlternateBase,   BG),
        (QPalette.Text,            TXT),
        (QPalette.ButtonText,      TXT),
        (QPalette.Button,          PANEL),
        (QPalette.Highlight,       BLUE),
        (QPalette.HighlightedText, "#ffffff"),
    ]:
        pal.setColor(role, QColor(color))
    app.setPalette(pal)
    app.setStyleSheet(QSS)
    win = MainWin()
    win.show()
    sys.exit(app.exec_())