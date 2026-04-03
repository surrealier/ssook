"""클래스 매핑 다이얼로그 — 선분 잇기 UI로 Model↔GT 클래스 매핑"""
from PySide6.QtCore import Qt, QPointF, QRectF, Signal
from PySide6.QtGui import QPainter, QPen, QColor, QFont, QFontMetrics, QPainterPath
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QWidget, QPushButton,
    QTabWidget, QCheckBox, QComboBox, QLabel, QLineEdit,
    QScrollArea, QDialogButtonBox, QGroupBox, QGridLayout,
)

_ROW_H = 36
_DOT_R = 8
_SNAP_DIST = 24
_PAD_X = 16
_COL_W = 180
_LINE_COLOR = QColor("#89b4fa")
_DOT_COLOR_MODEL = QColor("#89b4fa")
_DOT_COLOR_GT = QColor("#a6e3a1")


class MappingWidget(QWidget):
    """Model(좌) ↔ GT(우) 선분 잇기 위젯"""

    mapping_changed = Signal()

    def __init__(self, gt_classes, model_classes, parent=None):
        super().__init__(parent)
        self.gt = gt_classes          # [(id, name), ...]
        self.model = model_classes    # [(id, name), ...]
        self.connections = []         # [(gt_idx, model_idx), ...]

        self._dragging = False
        self._drag_side = None
        self._drag_idx = -1
        self._drag_pos = QPointF()
        self._snap_target = None

        self.setMouseTracking(True)
        rows = max(len(self.gt), len(self.model), 1)
        self.setMinimumHeight(rows * _ROW_H + 20)
        self.setMinimumWidth(_COL_W * 2 + 200)

    def _dot_pos(self, side, idx):
        """Model dot = 좌측 컬럼 오른쪽 끝, GT dot = 우측 컬럼 왼쪽 끝"""
        y = 10 + idx * _ROW_H + _ROW_H // 2
        if side == "model":
            return QPointF(_COL_W + _PAD_X, y)
        else:  # gt
            return QPointF(self.width() - _COL_W - _PAD_X, y)

    def _hit_test(self, pos):
        for side, items in [("model", self.model), ("gt", self.gt)]:
            for i in range(len(items)):
                dp = self._dot_pos(side, i)
                if (pos - dp).manhattanLength() < _SNAP_DIST:
                    return (side, i)
        return None

    def set_connections(self, conns):
        self.connections = list(conns)
        self.update()

    def get_mapping(self):
        """model_class_id → gt_class_id"""
        m = {}
        for gi, mi in self.connections:
            m[self.model[mi][0]] = self.gt[gi][0]
        return m

    # ── 마우스 ──
    def mousePressEvent(self, e):
        if e.button() == Qt.RightButton:
            self._remove_nearest_connection(e.position())
            return
        hit = self._hit_test(e.position())
        if hit:
            self._dragging = True
            self._drag_side, self._drag_idx = hit
            self._drag_pos = e.position()
            self._snap_target = None
            self.update()

    def mouseMoveEvent(self, e):
        if self._dragging:
            self._drag_pos = e.position()
            target_side = "gt" if self._drag_side == "model" else "model"
            items = self.gt if target_side == "gt" else self.model
            best, best_dist = None, _SNAP_DIST * 1.5
            for i in range(len(items)):
                dp = self._dot_pos(target_side, i)
                d = (e.position() - dp).manhattanLength()
                if d < best_dist:
                    best_dist = d
                    best = (target_side, i)
            self._snap_target = best
            self.update()

    def mouseReleaseEvent(self, e):
        if not self._dragging:
            return
        self._dragging = False
        if self._snap_target:
            ts, ti = self._snap_target
            if ts != self._drag_side:
                if self._drag_side == "model":
                    gi, mi = ti, self._drag_idx
                else:
                    gi, mi = self._drag_idx, ti
                # gt_idx 결정
                if self._drag_side == "gt":
                    gi = self._drag_idx
                    mi = ti
                else:
                    mi = self._drag_idx
                    gi = ti
                conn = (gi, mi)
                if conn not in self.connections:
                    self.connections.append(conn)
                    self.mapping_changed.emit()
        self._snap_target = None
        self.update()

    def _remove_nearest_connection(self, pos):
        if not self.connections:
            return
        best_i, best_dist = -1, 30
        for i, (gi, mi) in enumerate(self.connections):
            p1 = self._dot_pos("model", mi)
            p2 = self._dot_pos("gt", gi)
            d = self._point_to_line_dist(pos, p1, p2)
            if d < best_dist:
                best_dist = d
                best_i = i
        if best_i >= 0:
            self.connections.pop(best_i)
            self.mapping_changed.emit()
            self.update()

    @staticmethod
    def _point_to_line_dist(p, a, b):
        ab = b - a
        ap = p - a
        t = max(0, min(1, QPointF.dotProduct(ap, ab) / max(QPointF.dotProduct(ab, ab), 1e-9)))
        proj = a + t * ab
        return (p - proj).manhattanLength()

    # ── 그리기 ──
    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        font = QFont("Segoe UI", 10)
        p.setFont(font)
        text_color = self.palette().text().color()

        # Model 라벨 (좌)
        for i, (cid, name) in enumerate(self.model):
            y = 10 + i * _ROW_H
            dp = self._dot_pos("model", i)
            p.setPen(text_color)
            p.drawText(QRectF(_PAD_X, y, _COL_W - _PAD_X * 2, _ROW_H),
                       Qt.AlignRight | Qt.AlignVCenter, f"{cid}: {name}")
            p.setBrush(_DOT_COLOR_MODEL)
            p.setPen(QPen(_DOT_COLOR_MODEL.darker(120), 1.5))
            p.drawEllipse(dp, _DOT_R, _DOT_R)

        # GT 라벨 (우)
        gt_x = self.width() - _COL_W
        for i, (cid, name) in enumerate(self.gt):
            y = 10 + i * _ROW_H
            dp = self._dot_pos("gt", i)
            p.setPen(text_color)
            p.drawText(QRectF(gt_x + _PAD_X, y, _COL_W - _PAD_X * 2, _ROW_H),
                       Qt.AlignLeft | Qt.AlignVCenter, f"{cid}: {name}")
            p.setBrush(_DOT_COLOR_GT)
            p.setPen(QPen(_DOT_COLOR_GT.darker(120), 1.5))
            p.drawEllipse(dp, _DOT_R, _DOT_R)

        # 연결선 — 직선, 인덱스별 색상 변화
        p.setBrush(Qt.NoBrush)
        for idx, (gi, mi) in enumerate(self.connections):
            p1 = self._dot_pos("model", mi)
            p2 = self._dot_pos("gt", gi)
            hue = (210 + idx * 37) % 360  # 파란 계열에서 시작, 조금씩 회전
            pen = QPen(QColor.fromHsv(hue, 160, 220), 2)
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)
            p.drawLine(p1, p2)

        # 드래그 중인 선
        if self._dragging:
            start = self._dot_pos(self._drag_side, self._drag_idx)
            end = self._drag_pos
            if self._snap_target:
                end = self._dot_pos(*self._snap_target)
            p.setPen(QPen(_LINE_COLOR, 2, Qt.DashLine))
            p.drawLine(start, end)
            if self._snap_target:
                p.setBrush(Qt.NoBrush)
                p.setPen(QPen(_LINE_COLOR, 3))
                p.drawEllipse(end, _DOT_R + 4, _DOT_R + 4)

        p.end()


class _NameEditorWidget(QWidget):
    """클래스명이 없는 모델용 이름 편집 위젯"""
    def __init__(self, class_ids, parent=None):
        super().__init__(parent)
        lay = QGridLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        self._edits = {}
        for i, cid in enumerate(sorted(class_ids)):
            lay.addWidget(QLabel(f"클래스 {cid}:"), i, 0)
            le = QLineEdit(f"class_{cid}")
            self._edits[cid] = le
            lay.addWidget(le, i, 1)

    def get_names(self):
        return {cid: le.text().strip() or f"class_{cid}" for cid, le in self._edits.items()}


class ClassMappingDialog(QDialog):
    """모델별 Model↔GT 클래스 매핑 다이얼로그"""

    def __init__(self, gt_classes, model_infos, parent=None, prev_mappings=None, prev_mapped_only=True):
        """
        gt_classes: [(id, name), ...]
        model_infos: [(model_name, model_type, [(id, name), ...]), ...]
        prev_mappings: 이전 매핑 {model_name: {model_id: gt_id}} (복원용)
        """
        super().__init__(parent)
        self.setWindowTitle("클래스 매핑")
        self.resize(750, 550)
        self._gt = gt_classes
        self._model_infos = model_infos
        self._prev_mappings = prev_mappings or {}
        self._widgets = {}
        self._mapped_only = prev_mapped_only
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)

        top = QHBoxLayout()
        self._chk_mapped = QCheckBox("매핑된 클래스만 평가")
        self._chk_mapped.setChecked(self._mapped_only)
        self._chk_mapped.toggled.connect(lambda v: setattr(self, '_mapped_only', v))
        top.addWidget(self._chk_mapped)
        top.addStretch()
        top.addWidget(QLabel("매핑 복사:"))
        self._combo_copy_src = QComboBox()
        for name, _, _ in self._model_infos:
            self._combo_copy_src.addItem(name)
        top.addWidget(self._combo_copy_src)
        btn_copy = QPushButton("→ 현재 탭에 복사")
        btn_copy.clicked.connect(self._copy_mapping)
        top.addWidget(btn_copy)
        root.addLayout(top)

        # 컬럼 헤더
        hdr = QHBoxLayout()
        lbl_m = QLabel("< Model 클래스")
        lbl_m.setAlignment(Qt.AlignCenter)
        hdr.addWidget(lbl_m)
        lbl_g = QLabel("GT 클래스 >")
        lbl_g.setAlignment(Qt.AlignCenter)
        hdr.addWidget(lbl_g)
        root.addLayout(hdr)

        self._tabs = QTabWidget()
        for name, mtype, model_classes in self._model_infos:
            if not model_classes:
                container = QWidget()
                vlay = QVBoxLayout(container)
                ids = [c[0] for c in self._gt]
                editor = _NameEditorWidget(ids)
                grp = QGroupBox("모델 클래스명 지정 (이름을 입력 후 '적용' 클릭)")
                grp_lay = QVBoxLayout(grp)
                grp_lay.addWidget(editor)
                btn_apply = QPushButton("클래스명 적용")
                grp_lay.addWidget(btn_apply)
                vlay.addWidget(grp)

                scroll = QScrollArea()
                scroll.setWidgetResizable(True)
                mw = MappingWidget(self._gt, [(i, f"class_{i}") for i in ids])
                scroll.setWidget(mw)
                vlay.addWidget(scroll, 1)

                def _make_apply(n=name, ed=editor, w=mw, sc=scroll):
                    def _apply():
                        names = ed.get_names()
                        new_classes = sorted(names.items())
                        w2 = MappingWidget(self._gt, new_classes)
                        w2.set_connections(w.connections)
                        sc.setWidget(w2)
                        self._widgets[n] = w2
                    return _apply

                btn_apply.clicked.connect(_make_apply())
                self._widgets[name] = mw
                self._tabs.addTab(container, name)
            else:
                scroll = QScrollArea()
                scroll.setWidgetResizable(True)
                mw = MappingWidget(self._gt, model_classes)
                # 이전 매핑 복원 또는 동일 ID 기본 매핑
                prev = self._prev_mappings.get(name, {})
                gt_ids = {c[0]: i for i, c in enumerate(self._gt)}
                model_ids = {c[0]: i for i, c in enumerate(model_classes)}
                if prev:
                    conns = []
                    for mid, gid in prev.items():
                        if mid in model_ids and gid in gt_ids:
                            conns.append((gt_ids[gid], model_ids[mid]))
                else:
                    conns = [(gt_ids[cid], mi) for mi, (cid, _) in enumerate(model_classes) if cid in gt_ids]
                mw.set_connections(conns)
                scroll.setWidget(mw)
                self._widgets[name] = mw
                self._tabs.addTab(scroll, name)

        root.addWidget(self._tabs, 1)

        hint = QLabel("좌클릭+드래그: 연결 | 우클릭: 삭제 | N:N 매핑 가능")
        hint.setStyleSheet("font-size: 11px; font-style: italic;")
        root.addWidget(hint)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

    def _copy_mapping(self):
        src_name = self._combo_copy_src.currentText()
        cur_idx = self._tabs.currentIndex()
        cur_name = self._model_infos[cur_idx][0]
        if src_name == cur_name:
            return
        src_w = self._widgets.get(src_name)
        dst_w = self._widgets.get(cur_name)
        if src_w and dst_w:
            src_map = src_w.get_mapping()
            dst_model_ids = {c[0]: i for i, c in enumerate(dst_w.model)}
            gt_ids = {c[0]: i for i, c in enumerate(dst_w.gt)}
            new_conns = []
            for mid, gid in src_map.items():
                if mid in dst_model_ids and gid in gt_ids:
                    new_conns.append((gt_ids[gid], dst_model_ids[mid]))
            dst_w.set_connections(new_conns)

    def get_result(self):
        mappings = {}
        for name, _, _ in self._model_infos:
            w = self._widgets.get(name)
            mappings[name] = w.get_mapping() if w else {}
        return mappings, self._mapped_only
