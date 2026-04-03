"""클래스 리매핑/병합/삭제: YOLO 라벨 일괄 변환"""
import os, glob
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QFileDialog, QProgressBar, QMessageBox, QGroupBox, QTableWidget,
    QTableWidgetItem, QComboBox, QCheckBox, QTextEdit, QHeaderView,
)


class _RemapWorker(QThread):
    progress = Signal(int, int)
    finished_ok = Signal(object)
    error = Signal(str)

    def __init__(self, lbl_dir, out_dir, mapping, auto_reindex):
        super().__init__()
        self.lbl_dir, self.out_dir = lbl_dir, out_dir
        self.mapping = mapping  # {src_id: (action, target_id)}  action: keep/merge/delete
        self.auto_reindex = auto_reindex

    def run(self):
        try:
            files = glob.glob(os.path.join(self.lbl_dir, "**", "*.txt"), recursive=True)
            if not files:
                self.error.emit("라벨 파일이 없습니다.")
                return
            os.makedirs(self.out_dir, exist_ok=True)
            # build reindex map
            used_ids = set()
            for src, (action, target) in self.mapping.items():
                if action != "delete":
                    used_ids.add(target)
            reindex = {}
            if self.auto_reindex:
                for i, cid in enumerate(sorted(used_ids)):
                    reindex[cid] = i
            stats = {"files": 0, "boxes_in": 0, "boxes_out": 0, "deleted": 0}
            for fi, fp in enumerate(files):
                lines_out = []
                with open(fp) as f:
                    for line in f:
                        p = line.strip().split()
                        if len(p) < 5:
                            continue
                        stats["boxes_in"] += 1
                        cid = int(p[0])
                        action, target = self.mapping.get(cid, ("keep", cid))
                        if action == "delete":
                            stats["deleted"] += 1
                            continue
                        new_id = reindex.get(target, target) if self.auto_reindex else target
                        lines_out.append(f"{new_id} {' '.join(p[1:])}\n")
                        stats["boxes_out"] += 1
                out_path = os.path.join(self.out_dir, os.path.basename(fp))
                with open(out_path, "w") as f:
                    f.writelines(lines_out)
                stats["files"] += 1
                self.progress.emit(fi + 1, len(files))
            self.finished_ok.emit(stats)
        except Exception as e:
            self.error.emit(str(e))


class ClassRemapper(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(QLabel("라벨 폴더:"))
        self._le_lbl = QLineEdit()
        top.addWidget(self._le_lbl, 1)
        b1 = QPushButton("찾아보기")
        b1.clicked.connect(lambda: self._browse(self._le_lbl))
        top.addWidget(b1)
        top.addWidget(QLabel("출력 폴더:"))
        self._le_out = QLineEdit()
        top.addWidget(self._le_out, 1)
        b2 = QPushButton("찾아보기")
        b2.clicked.connect(lambda: self._browse(self._le_out))
        top.addWidget(b2)
        self._btn_scan = QPushButton("스캔")
        self._btn_scan.clicked.connect(self._scan)
        top.addWidget(self._btn_scan)
        root.addLayout(top)

        # 매핑 테이블
        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["소스 ID", "박스 수", "액션", "대상 ID"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        root.addWidget(self._table)

        row_opt = QHBoxLayout()
        self._chk_reindex = QCheckBox("자동 재인덱싱 (0부터 연속)")
        self._chk_reindex.setChecked(True)
        row_opt.addWidget(self._chk_reindex)
        row_opt.addStretch()
        self._btn_run = QPushButton("적용")
        self._btn_run.clicked.connect(self._run)
        row_opt.addWidget(self._btn_run)
        root.addLayout(row_opt)

        self._prog = QProgressBar()
        self._prog.setMaximumHeight(18)
        root.addWidget(self._prog)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        root.addWidget(self._log, 1)

    def _browse(self, le):
        d = QFileDialog.getExistingDirectory(self, "폴더 선택")
        if d:
            le.setText(d)

    def _scan(self):
        d = self._le_lbl.text()
        if not os.path.isdir(d):
            return
        counts = {}
        for txt in glob.glob(os.path.join(d, "**", "*.txt"), recursive=True):
            with open(txt) as f:
                for line in f:
                    p = line.strip().split()
                    if p:
                        cid = int(p[0])
                        counts[cid] = counts.get(cid, 0) + 1
        self._table.setRowCount(len(counts))
        for i, cid in enumerate(sorted(counts)):
            self._table.setItem(i, 0, QTableWidgetItem(str(cid)))
            item_cnt = QTableWidgetItem(str(counts[cid]))
            item_cnt.setFlags(item_cnt.flags() & ~Qt.ItemIsEditable)
            self._table.setItem(i, 1, item_cnt)
            cb = QComboBox()
            cb.addItems(["유지", "병합", "삭제"])
            self._table.setCellWidget(i, 2, cb)
            self._table.setItem(i, 3, QTableWidgetItem(str(cid)))

    def _get_mapping(self):
        mapping = {}
        for i in range(self._table.rowCount()):
            src = int(self._table.item(i, 0).text())
            action_text = self._table.cellWidget(i, 2).currentText()
            target = int(self._table.item(i, 3).text())
            action = {"유지": "keep", "병합": "merge", "삭제": "delete"}[action_text]
            mapping[src] = (action, target)
        return mapping

    def _run(self):
        if not os.path.isdir(self._le_lbl.text()) or not self._le_out.text():
            QMessageBox.warning(self, "알림", "폴더를 지정하세요.")
            return
        self._btn_run.setEnabled(False)
        self._log.clear()
        self._worker = _RemapWorker(
            self._le_lbl.text(), self._le_out.text(),
            self._get_mapping(), self._chk_reindex.isChecked())
        self._worker.progress.connect(lambda c, t: self._prog.setValue(int(c / t * 100)))
        self._worker.finished_ok.connect(self._on_done)
        self._worker.error.connect(lambda e: (self._log.append(f"오류: {e}"), self._btn_run.setEnabled(True)))
        self._worker.start()

    def _on_done(self, stats):
        self._btn_run.setEnabled(True)
        self._prog.setValue(100)
        self._log.append(f"완료: {stats['files']}개 파일 처리")
        self._log.append(f"입력 박스: {stats['boxes_in']} → 출력: {stats['boxes_out']} (삭제: {stats['deleted']})")
