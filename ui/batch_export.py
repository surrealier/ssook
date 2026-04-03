"""배치 추론 & 결과 내보내기: 이미지 폴더 → 모델 추론 → txt/json/csv 저장"""
import os, glob, json, csv, time

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFileDialog, QComboBox, QDoubleSpinBox,
    QGroupBox, QProgressBar, QMessageBox, QCheckBox, QTextEdit,
)

from core.model_loader import load_model
from core.inference import (
    run_inference, run_classification, convert_darknet_to_unified,
    UNIFIED_NAMES,
)


class _BatchWorker(QThread):
    progress = Signal(int, int)
    log = Signal(str)
    finished_ok = Signal(int)
    error = Signal(str)

    def __init__(self, img_dir, model_path, model_type, conf, out_dir, fmt, save_vis):
        super().__init__()
        self.img_dir = img_dir
        self.model_path = model_path
        self.model_type = model_type
        self.conf = conf
        self.out_dir = out_dir
        self.fmt = fmt          # "yolo_txt" | "json" | "csv"
        self.save_vis = save_vis
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            mi = load_model(self.model_path, model_type=self.model_type)
            is_cls = mi.task_type == "classification"

            exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
            files = []
            for e in exts:
                files.extend(glob.glob(os.path.join(self.img_dir, e)))
            files.sort()
            if not files:
                self.error.emit("이미지가 없습니다.")
                return

            os.makedirs(self.out_dir, exist_ok=True)
            if self.save_vis:
                vis_dir = os.path.join(self.out_dir, "vis")
                os.makedirs(vis_dir, exist_ok=True)

            all_results = []  # for json/csv

            for i, fp in enumerate(files):
                if self._stop:
                    break
                frame = cv2.imread(fp)
                if frame is None:
                    continue
                stem = os.path.splitext(os.path.basename(fp))[0]
                h, w = frame.shape[:2]

                if is_cls:
                    res = run_classification(mi, frame)
                    entry = {"file": os.path.basename(fp), "class_id": res.class_id,
                             "confidence": round(res.confidence, 6),
                             "top_k": [(c, round(s, 6)) for c, s in res.top_k],
                             "infer_ms": round(res.infer_ms, 2)}
                    all_results.append(entry)

                    if self.fmt == "yolo_txt":
                        with open(os.path.join(self.out_dir, stem + ".txt"), "w") as f:
                            f.write(f"{res.class_id} {res.confidence:.6f}\n")

                    if self.save_vis:
                        vis = frame.copy()
                        label = mi.names.get(res.class_id, str(res.class_id))
                        cv2.putText(vis, f"{label}: {res.confidence:.3f}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.imwrite(os.path.join(vis_dir, stem + ".jpg"), vis)
                else:
                    res = run_inference(mi, frame, self.conf)
                    if self.model_type == "darknet":
                        res = convert_darknet_to_unified(res)

                    boxes_data = []
                    for box, score, cid in zip(res.boxes, res.scores, res.class_ids):
                        cid = int(cid)
                        x1, y1, x2, y2 = box
                        cx = ((x1 + x2) / 2) / w
                        cy = ((y1 + y2) / 2) / h
                        bw = (x2 - x1) / w
                        bh = (y2 - y1) / h
                        boxes_data.append({"class_id": cid, "cx": round(cx, 6), "cy": round(cy, 6),
                                           "w": round(bw, 6), "h": round(bh, 6),
                                           "confidence": round(float(score), 6)})

                    entry = {"file": os.path.basename(fp), "boxes": boxes_data,
                             "infer_ms": round(res.infer_ms, 2)}
                    all_results.append(entry)

                    if self.fmt == "yolo_txt":
                        with open(os.path.join(self.out_dir, stem + ".txt"), "w") as f:
                            for b in boxes_data:
                                f.write(f"{b['class_id']} {b['cx']} {b['cy']} {b['w']} {b['h']}\n")

                    if self.save_vis:
                        vis = frame.copy()
                        for box, score, cid in zip(res.boxes, res.scores, res.class_ids):
                            x1, y1, x2, y2 = [int(v) for v in box]
                            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(vis, f"{int(cid)}:{score:.2f}", (x1, max(y1 - 4, 14)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        cv2.imwrite(os.path.join(vis_dir, stem + ".jpg"), vis)

                self.progress.emit(i + 1, len(files))

            # 전체 결과 저장
            if self.fmt == "json":
                with open(os.path.join(self.out_dir, "results.json"), "w", encoding="utf-8") as f:
                    json.dump(all_results, f, ensure_ascii=False, indent=2)
            elif self.fmt == "csv":
                csv_path = os.path.join(self.out_dir, "results.csv")
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    if is_cls:
                        writer.writerow(["file", "class_id", "confidence", "infer_ms"])
                        for r in all_results:
                            writer.writerow([r["file"], r["class_id"], r["confidence"], r["infer_ms"]])
                    else:
                        writer.writerow(["file", "class_id", "cx", "cy", "w", "h", "confidence"])
                        for r in all_results:
                            for b in r["boxes"]:
                                writer.writerow([r["file"], b["class_id"], b["cx"], b["cy"],
                                                 b["w"], b["h"], b["confidence"]])

            self.finished_ok.emit(len(all_results))
        except Exception as e:
            self.error.emit(str(e))


class BatchExportTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)

        grp = QGroupBox("배치 추론 설정")
        g = QVBoxLayout(grp)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("모델:"))
        self._le_model = QLineEdit(); self._le_model.setReadOnly(True)
        row1.addWidget(self._le_model, 1)
        btn_m = QPushButton("찾아보기")
        btn_m.clicked.connect(self._browse_model)
        row1.addWidget(btn_m)
        self._combo_type = QComboBox()
        self._combo_type.addItems(["YOLO", "CenterNet"])
        row1.addWidget(self._combo_type)
        g.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("이미지 폴더:"))
        self._le_img = QLineEdit()
        row2.addWidget(self._le_img, 1)
        btn_i = QPushButton("찾아보기")
        btn_i.clicked.connect(lambda: self._browse_dir(self._le_img))
        row2.addWidget(btn_i)
        row2.addWidget(QLabel("출력 폴더:"))
        self._le_out = QLineEdit()
        row2.addWidget(self._le_out, 1)
        btn_o = QPushButton("찾아보기")
        btn_o.clicked.connect(lambda: self._browse_dir(self._le_out))
        row2.addWidget(btn_o)
        g.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Confidence:"))
        self._spin_conf = QDoubleSpinBox()
        self._spin_conf.setRange(0.01, 1.0); self._spin_conf.setValue(0.25); self._spin_conf.setSingleStep(0.05)
        row3.addWidget(self._spin_conf)
        row3.addWidget(QLabel("출력 형식:"))
        self._combo_fmt = QComboBox()
        self._combo_fmt.addItems(["YOLO txt", "JSON", "CSV"])
        row3.addWidget(self._combo_fmt)
        self._chk_vis = QCheckBox("시각화 이미지 저장")
        row3.addWidget(self._chk_vis)
        row3.addStretch()
        self._btn_run = QPushButton("추론 실행")
        self._btn_run.clicked.connect(self._run)
        row3.addWidget(self._btn_run)
        self._btn_stop = QPushButton("중지")
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._stop)
        row3.addWidget(self._btn_stop)
        g.addLayout(row3)

        self._prog = QProgressBar()
        g.addWidget(self._prog)
        root.addWidget(grp)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(200)
        root.addWidget(self._log, 1)

    def _browse_model(self):
        p, _ = QFileDialog.getOpenFileName(self, "모델 선택", "", "ONNX (*.onnx)")
        if p: self._le_model.setText(p)

    def _browse_dir(self, le):
        d = QFileDialog.getExistingDirectory(self, "폴더 선택")
        if d: le.setText(d)

    def _run(self):
        model_path = self._le_model.text()
        img_dir = self._le_img.text()
        out_dir = self._le_out.text()
        if not model_path or not os.path.isdir(img_dir) or not out_dir:
            QMessageBox.warning(self, "알림", "모델, 이미지 폴더, 출력 폴더를 모두 지정하세요.")
            return

        mtype = "yolo" if self._combo_type.currentIndex() == 0 else "darknet"
        fmts = ["yolo_txt", "json", "csv"]
        fmt = fmts[self._combo_fmt.currentIndex()]

        self._btn_run.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._prog.setValue(0)
        self._log.clear()
        self._log.append(f"추론 시작: {img_dir}")

        self._worker = _BatchWorker(img_dir, model_path, mtype,
                                    self._spin_conf.value(), out_dir, fmt, self._chk_vis.isChecked())
        self._worker.progress.connect(lambda c, t: self._prog.setValue(int(c / t * 100)))
        self._worker.log.connect(self._log.append)
        self._worker.finished_ok.connect(self._on_done)
        self._worker.error.connect(lambda e: (QMessageBox.critical(self, "오류", e),
                                              self._btn_run.setEnabled(True), self._btn_stop.setEnabled(False)))
        self._worker.start()

    def _stop(self):
        if hasattr(self, '_worker'):
            self._worker.stop()

    def _on_done(self, count):
        self._btn_run.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._prog.setValue(100)
        self._log.append(f"완료: {count}장 처리, 출력: {self._le_out.text()}")
