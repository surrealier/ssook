"""라벨 포맷 변환: YOLO ↔ COCO JSON ↔ Pascal VOC XML"""
import os, glob, json, cv2
import xml.etree.ElementTree as ET
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QFileDialog, QProgressBar, QMessageBox, QGroupBox, QComboBox,
    QTableWidget, QTableWidgetItem, QTextEdit, QHeaderView, QCheckBox,
)


class _ConvertWorker(QThread):
    progress = Signal(int, int)
    finished_ok = Signal(object)
    error = Signal(str)

    def __init__(self, img_dir, lbl_dir, out_dir, src_fmt, dst_fmt, class_names):
        super().__init__()
        self.img_dir, self.lbl_dir, self.out_dir = img_dir, lbl_dir, out_dir
        self.src_fmt, self.dst_fmt = src_fmt, dst_fmt
        self.class_names = class_names  # {id: name}

    def _read_image_size(self, stem):
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            fp = os.path.join(self.img_dir, stem + ext)
            if os.path.isfile(fp):
                img = cv2.imread(fp)
                if img is not None:
                    h, w = img.shape[:2]
                    return w, h, os.path.basename(fp)
        return None, None, None

    def _load_yolo(self):
        data = {}
        for txt in glob.glob(os.path.join(self.lbl_dir, "**", "*.txt"), recursive=True):
            stem = os.path.splitext(os.path.basename(txt))[0]
            boxes = []
            with open(txt) as f:
                for line in f:
                    p = line.strip().split()
                    if len(p) >= 5:
                        boxes.append((int(p[0]), *[float(x) for x in p[1:5]]))
            data[stem] = boxes
        return data

    def _load_coco(self):
        jf = None
        for f in glob.glob(os.path.join(self.lbl_dir, "**", "*.json"), recursive=True):
            jf = f
            break
        if not jf:
            self.error.emit("COCO JSON 파일을 찾을 수 없습니다.")
            return None
        with open(jf) as f:
            coco = json.load(f)
        img_map = {im["id"]: im for im in coco["images"]}
        data = {}
        for ann in coco["annotations"]:
            im = img_map[ann["image_id"]]
            stem = os.path.splitext(im["file_name"])[0]
            x, y, bw, bh = ann["bbox"]
            w, h = im["width"], im["height"]
            cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
            data.setdefault(stem, []).append((ann["category_id"], cx, cy, bw / w, bh / h))
        return data

    def _load_voc(self):
        data = {}
        name_to_id = {v: k for k, v in self.class_names.items()}
        for xml_f in glob.glob(os.path.join(self.lbl_dir, "**", "*.xml"), recursive=True):
            tree = ET.parse(xml_f)
            root = tree.getroot()
            stem = os.path.splitext(root.findtext("filename", ""))[0]
            if not stem:
                stem = os.path.splitext(os.path.basename(xml_f))[0]
            sz = root.find("size")
            w = int(sz.findtext("width", "1"))
            h = int(sz.findtext("height", "1"))
            boxes = []
            for obj in root.findall("object"):
                name = obj.findtext("name", "")
                cid = name_to_id.get(name, -1)
                bb = obj.find("bndbox")
                x1, y1 = float(bb.findtext("xmin", "0")), float(bb.findtext("ymin", "0"))
                x2, y2 = float(bb.findtext("xmax", "0")), float(bb.findtext("ymax", "0"))
                cx, cy = ((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h
                bw, bh = (x2 - x1) / w, (y2 - y1) / h
                boxes.append((cid, cx, cy, bw, bh))
            data[stem] = boxes
        return data

    def _save_yolo(self, data):
        os.makedirs(self.out_dir, exist_ok=True)
        for stem, boxes in data.items():
            with open(os.path.join(self.out_dir, stem + ".txt"), "w") as f:
                for cid, cx, cy, bw, bh in boxes:
                    f.write(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    def _save_coco(self, data):
        os.makedirs(self.out_dir, exist_ok=True)
        images, annotations = [], []
        ann_id = 1
        for img_id, (stem, boxes) in enumerate(data.items(), 1):
            w, h, fname = self._read_image_size(stem)
            if w is None:
                continue
            images.append(dict(id=img_id, file_name=fname, width=w, height=h))
            for cid, cx, cy, bw, bh in boxes:
                abs_w, abs_h = bw * w, bh * h
                abs_x, abs_y = (cx - bw / 2) * w, (cy - bh / 2) * h
                annotations.append(dict(id=ann_id, image_id=img_id, category_id=cid,
                                        bbox=[abs_x, abs_y, abs_w, abs_h],
                                        area=abs_w * abs_h, iscrowd=0))
                ann_id += 1
        cats = [dict(id=k, name=v) for k, v in self.class_names.items()]
        coco = dict(images=images, annotations=annotations, categories=cats)
        with open(os.path.join(self.out_dir, "annotations.json"), "w") as f:
            json.dump(coco, f, indent=2)

    def _save_voc(self, data):
        os.makedirs(self.out_dir, exist_ok=True)
        for stem, boxes in data.items():
            w, h, fname = self._read_image_size(stem)
            if w is None:
                continue
            root = ET.Element("annotation")
            ET.SubElement(root, "filename").text = fname
            sz = ET.SubElement(root, "size")
            ET.SubElement(sz, "width").text = str(w)
            ET.SubElement(sz, "height").text = str(h)
            ET.SubElement(sz, "depth").text = "3"
            for cid, cx, cy, bw, bh in boxes:
                obj = ET.SubElement(root, "object")
                ET.SubElement(obj, "name").text = self.class_names.get(cid, str(cid))
                bb = ET.SubElement(obj, "bndbox")
                ET.SubElement(bb, "xmin").text = str(int((cx - bw / 2) * w))
                ET.SubElement(bb, "ymin").text = str(int((cy - bh / 2) * h))
                ET.SubElement(bb, "xmax").text = str(int((cx + bw / 2) * w))
                ET.SubElement(bb, "ymax").text = str(int((cy + bh / 2) * h))
            ET.ElementTree(root).write(os.path.join(self.out_dir, stem + ".xml"), encoding="unicode")

    def run(self):
        try:
            loaders = {"YOLO": self._load_yolo, "COCO": self._load_coco, "VOC": self._load_voc}
            savers = {"YOLO": self._save_yolo, "COCO": self._save_coco, "VOC": self._save_voc}
            data = loaders[self.src_fmt]()
            if data is None:
                return
            self.progress.emit(1, 2)
            savers[self.dst_fmt](data)
            self.progress.emit(2, 2)
            self.finished_ok.emit(len(data))
        except Exception as e:
            self.error.emit(str(e))


class FormatConverter(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        # 폴더
        g1 = QGroupBox("입출력")
        gl = QVBoxLayout(g1)
        for label, attr in [("이미지 폴더:", "_le_img"), ("라벨 폴더:", "_le_lbl"), ("출력 폴더:", "_le_out")]:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            le = QLineEdit()
            setattr(self, attr, le)
            row.addWidget(le, 1)
            btn = QPushButton("찾아보기")
            btn.clicked.connect(lambda _, l=le: self._browse(l))
            row.addWidget(btn)
            gl.addLayout(row)
        row_fmt = QHBoxLayout()
        row_fmt.addWidget(QLabel("소스 포맷:"))
        self._cb_src = QComboBox()
        self._cb_src.addItems(["YOLO", "COCO", "VOC"])
        row_fmt.addWidget(self._cb_src)
        row_fmt.addWidget(QLabel("→ 대상 포맷:"))
        self._cb_dst = QComboBox()
        self._cb_dst.addItems(["COCO", "VOC", "YOLO"])
        row_fmt.addWidget(self._cb_dst)
        self._btn_scan = QPushButton("클래스 스캔")
        self._btn_scan.clicked.connect(self._scan_classes)
        row_fmt.addWidget(self._btn_scan)
        self._btn_run = QPushButton("변환")
        self._btn_run.clicked.connect(self._run)
        row_fmt.addWidget(self._btn_run)
        gl.addLayout(row_fmt)
        root.addWidget(g1)

        self._prog = QProgressBar()
        self._prog.setMaximumHeight(18)
        root.addWidget(self._prog)

        # 클래스 매핑
        self._class_table = QTableWidget(0, 2)
        self._class_table.setHorizontalHeaderLabels(["ID", "이름"])
        self._class_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self._class_table.setMaximumHeight(200)
        root.addWidget(self._class_table)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        root.addWidget(self._log, 1)

    def _browse(self, le):
        d = QFileDialog.getExistingDirectory(self, "폴더 선택")
        if d:
            le.setText(d)

    def _scan_classes(self):
        lbl_dir = self._le_lbl.text()
        if not os.path.isdir(lbl_dir):
            return
        ids = set()
        for txt in glob.glob(os.path.join(lbl_dir, "**", "*.txt"), recursive=True):
            with open(txt) as f:
                for line in f:
                    p = line.strip().split()
                    if p:
                        ids.add(int(p[0]))
        self._class_table.setRowCount(len(ids))
        for i, cid in enumerate(sorted(ids)):
            self._class_table.setItem(i, 0, QTableWidgetItem(str(cid)))
            self._class_table.setItem(i, 1, QTableWidgetItem(f"class_{cid}"))

    def _get_class_names(self):
        names = {}
        for i in range(self._class_table.rowCount()):
            cid = int(self._class_table.item(i, 0).text())
            name = self._class_table.item(i, 1).text()
            names[cid] = name
        return names

    def _run(self):
        if not os.path.isdir(self._le_lbl.text()) or not self._le_out.text():
            QMessageBox.warning(self, "알림", "폴더를 지정하세요.")
            return
        src, dst = self._cb_src.currentText(), self._cb_dst.currentText()
        if src == dst:
            QMessageBox.warning(self, "알림", "소스와 대상 포맷이 같습니다.")
            return
        self._btn_run.setEnabled(False)
        self._log.clear()
        self._log.append(f"{src} → {dst} 변환 시작...")
        self._worker = _ConvertWorker(
            self._le_img.text(), self._le_lbl.text(), self._le_out.text(),
            src, dst, self._get_class_names())
        self._worker.progress.connect(lambda c, t: self._prog.setValue(int(c / t * 100)))
        self._worker.finished_ok.connect(self._on_done)
        self._worker.error.connect(lambda e: (self._log.append(f"오류: {e}"), self._btn_run.setEnabled(True)))
        self._worker.start()

    def _on_done(self, count):
        self._btn_run.setEnabled(True)
        self._prog.setValue(100)
        self._log.append(f"변환 완료: {count}개 파일")
