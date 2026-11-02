"""
Klasifikasi Bentuk dan Warna (Tema 8) - UTS Praktikum Visi Komputer, Kelompok 8
Website Streamlit: unggah foto -> deteksi bentuk + warna -> tampil berdampingan dengan info hasil.

Jalankan lokal :  streamlit run app.py
Library        :  streamlit, opencv-python-headless, numpy (pipeline hanya OpenCV + NumPy, tanpa deep learning)
"""
from collections import Counter

import cv2
import numpy as np
import streamlit as st

# ---------------------------------------------------------------- konfigurasi
MAX_SISI = 1000                       # sisi terpanjang citra setelah resize
BENTUK = ["Segitiga", "Persegi", "Persegi panjang", "Lingkaran", "Bintang", "Love"]
WARNA  = ["Merah", "Kuning", "Hijau", "Biru", "Hitam", "Mint", "Ungu", "Pink", "Soft pink"]


# ------------------------------------------------------------- 1. praproses
def muat_citra(path, max_sisi=MAX_SISI):
    """Baca citra (OpenCV sudah menerapkan orientasi EXIF) lalu resize dengan rasio tetap."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    s = min(1.0, max_sisi / max(h, w))
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def praproses(img):
    """Gaussian blur lalu konversi BGR -> Lab (jarak warna di Lab mendekati persepsi mata)."""
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB).astype(np.float32)
    return blur, lab


# --------------------------------------------------------- 2. segmentasi/mask
def peta_skor(lab, latar):
    """Skor 'seberapa beda dari latar lokal'.
    kroma (a,b) -> objek berwarna ; 'gelap sekali' -> objek hitam.
    Bayangan hanya menurunkan L sedikit sehingga skornya kecil dan tidak ikut jadi objek."""
    d = lab - latar
    kroma = np.sqrt(d[:, :, 1] ** 2 + d[:, :, 2] ** 2)
    gelap = np.maximum(0, -d[:, :, 0] - 25)
    return np.maximum(kroma, gelap)


def estimasi_latar(lab, n_iter=4, sigma=35, thr=8):
    """Perkirakan warna latar tiap piksel (latar tidak rata: gradasi cahaya, lantai, bayangan).
    1) tebakan awal: median blur besar pada citra kecil (objek kecil terhapus)
    2) ulangi: objek yang sudah ketemu dikeluarkan, latar = rata-rata Gaussian piksel latar saja."""
    h, w = lab.shape[:2]
    kecil = cv2.resize(lab.astype(np.uint8), None, fx=0.125, fy=0.125, interpolation=cv2.INTER_AREA)
    latar = cv2.resize(cv2.medianBlur(kecil, 41), (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    for _ in range(n_iter):
        fg = cv2.dilate((peta_skor(lab, latar) > thr).astype(np.uint8), kern)
        bobot = (1 - fg).astype(np.float32)
        num = cv2.GaussianBlur(lab * bobot[:, :, None], (0, 0), sigma)
        den = cv2.GaussianBlur(bobot, (0, 0), sigma)[:, :, None]
        latar = np.where(den > 0.02, num / np.maximum(den, 1e-3), latar)
    return latar


def segmentasi(img, thr=8):
    """Mask objek = threshold pada peta skor, lalu opening untuk membuang garis tipis (nat lantai, tepi foto)."""
    blur, lab = praproses(img)
    latar = estimasi_latar(lab, thr=thr)
    skor = peta_skor(lab, latar)
    skor8 = np.clip(skor * 4, 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(skor8, thr * 4, 255, cv2.THRESH_BINARY)
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k7)
    return mask, skor, lab, latar


K5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def pisah_menempel(mask, img):
    """Pisahkan objek yang bersentuhan: marker = inti distance transform, lalu watershed."""
    hasil = np.zeros(mask.shape, np.int32)
    n, lbl, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    nomor = 1
    for i in range(1, n):
        x, y, w, h, a = st[i]
        x0, y0 = max(0, x - 5), max(0, y - 5)
        x1, y1 = min(mask.shape[1], x + w + 5), min(mask.shape[0], y + h + 5)
        comp = (lbl[y0:y1, x0:x1] == i).astype(np.uint8)
        dt = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
        inti = (dt > 0.62 * dt.max()).astype(np.uint8)
        nc, cl = cv2.connectedComponents(inti)
        besar = [k for k in range(1, nc) if (cl == k).sum() >= 40]
        if len(besar) >= 2:
            mk = np.zeros(comp.shape, np.int32)
            for j, k in enumerate(besar):
                mk[cl == k] = j + 2
            mk[comp == 0] = 1
            ws = cv2.watershed(blur[y0:y1, x0:x1].copy(), mk)
            for j in range(len(besar)):
                bagian = cv2.morphologyEx((ws == j + 2).astype(np.uint8), cv2.MORPH_OPEN, K5)
                hasil[y0:y1, x0:x1][bagian > 0] = nomor
                nomor += 1
        else:
            hasil[y0:y1, x0:x1][comp > 0] = nomor
            nomor += 1
    return hasil, nomor - 1


# ------------------------------------------------------------- 3. deteksi kontur
def luas_solid(c):
    a = cv2.contourArea(c)
    ha = cv2.contourArea(cv2.convexHull(c))
    return a, (a / ha if ha > 0 else 0)


def deteksi(img):
    """Mask -> pisah objek menempel -> findContours -> buang noise (kecil, besar, menempel tepi foto)."""
    mask, skor, lab, latar = segmentasi(img)
    H, W = mask.shape
    total = H * W
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, K5)
    label, n = pisah_menempel(mask, img)
    kontur = []
    for k in range(1, n + 1):
        cs, _ = cv2.findContours((label == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            continue
        c = max(cs, key=cv2.contourArea)
        a, sol = luas_solid(c)
        if a < total * 0.0009 or a > total * 0.12:
            continue                                   # terlalu kecil (noise/pantulan) / terlalu besar (bukan objek)
        x, y, w, h = cv2.boundingRect(c)
        nempel = x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2
        if nempel:
            (_, _), (rw, rh), _ = cv2.minAreaRect(c)
            asp = max(rw, rh) / max(1e-6, min(rw, rh))
            # objek yang terpotong tepi foto tetap boleh asal padat (cembung, tidak memanjang, tidak raksasa);
            # tepi meja, keyboard, tangan, dan bingkai hitam foto biasanya memanjang/tidak padat/besar.
            if not (sol > 0.9 and asp < 3 and a < total * 0.04):
                continue
        if sol < 0.45:
            continue
        kontur.append(c)
    return mask, kontur, lab, latar


# ----------------------------------------------------------- 4. klasifikasi bentuk
def fitur_bentuk(c, eps_halus=0.02, eps_kasar=0.04):
    a = cv2.contourArea(c)
    per = cv2.arcLength(c, True)
    hull = cv2.convexHull(c)
    sol = a / max(cv2.contourArea(hull), 1e-6)
    circ = 4 * np.pi * a / max(per * per, 1e-6)
    n_halus = len(cv2.approxPolyDP(c, eps_halus * per, True))
    n_kasar = len(cv2.approxPolyDP(c, eps_kasar * per, True))
    n_cekung = 0
    try:
        idx = cv2.convexHull(c, returnPoints=False)
        df = cv2.convexityDefects(c, idx)
        if df is not None:
            n_cekung = int(sum(1 for d in df[:, 0] if d[3] / 256.0 > 0.04 * np.sqrt(a)))
    except cv2.error:
        pass
    return dict(luas=a, keliling=per, solid=sol, circ=circ, n_halus=n_halus, n_kasar=n_kasar, n_cekung=n_cekung)


def iou_elips(c):
    """Seberapa mirip kontur dengan elips terbaik (lingkaran yang difoto miring = elips)."""
    if len(c) < 5:
        return 0.0
    (cx, cy), (MA, ma), ang = cv2.fitEllipse(c)
    x, y, w, h = cv2.boundingRect(c)
    pad = 10
    A = np.zeros((h + 2 * pad, w + 2 * pad), np.uint8)
    B = A.copy()
    cv2.drawContours(A, [c - [x - pad, y - pad]], -1, 255, -1)
    cv2.ellipse(B, ((cx - x + pad, cy - y + pad), (MA, ma), ang), 255, -1)
    return np.logical_and(A > 0, B > 0).sum() / max(np.logical_or(A > 0, B > 0).sum(), 1)


def rasio_sisi(c):
    """Rasio sisi segiempat. Dipakai 4 titik hasil approxPolyDP (rata-rata sisi berhadapan) agar
    tahan terhadap rotasi dan sudut miring; cadangan: minAreaRect."""
    per = cv2.arcLength(c, True)
    q = cv2.approxPolyDP(c, 0.04 * per, True)
    if len(q) != 4:
        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        return max(rw, rh) / max(1e-6, min(rw, rh))
    p = q[:, 0, :].astype(np.float32)
    s = [np.linalg.norm(p[i] - p[(i + 1) % 4]) for i in range(4)]
    a, b = (s[0] + s[2]) / 2, (s[1] + s[3]) / 2
    return max(a, b) / max(1e-6, min(a, b))


def klasifikasi_bentuk(c, eps_halus=0.02, eps_kasar=0.04, f=None):
    f = f or fitur_bentuk(c, eps_halus, eps_kasar)
    n2, n4 = f["n_halus"], f["n_kasar"]
    if f["solid"] < 0.80 and f["n_cekung"] >= 3:                 # banyak cekungan dalam -> bintang
        return "Bintang"
    if f["solid"] > 0.95 and n2 >= 6 and (f["circ"] > 0.85 or iou_elips(c) > 0.93):
        return "Lingkaran"
    if n4 == 3 and n2 <= 5 and f["solid"] > 0.88:
        return "Segitiga"
    if n2 <= 5 and n4 == 4 and f["solid"] > 0.92:
        return "Persegi" if rasio_sisi(c) < 1.4 else "Persegi panjang"
    if f["solid"] > 0.88 and n2 >= 6:                            # cembung, ada 1 lekukan -> love
        return "Love"
    if n4 == 3:
        return "Segitiga"
    if n4 == 4:
        return "Persegi" if rasio_sisi(c) < 1.4 else "Persegi panjang"
    return "Love"


# ----------------------------------------------------------- 5. klasifikasi warna
def mask_dalam(shape, c, erode=9):
    m = np.zeros(shape[:2], np.uint8)
    cv2.drawContours(m, [c], -1, 255, -1)
    er = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode, erode)))
    return er if er.sum() >= 255 * 30 else m          # buang piksel tepi (campur dengan latar)


def selisih_lab(lab, latar, c):
    """Median (dL, da, db) bagian dalam objek relatif terhadap latar lokal => kebal terhadap
    warna lampu dan bayangan karena latar ikut berubah bersama objek."""
    er = mask_dalam(lab.shape, c)
    return np.median((lab - latar)[er > 0], axis=0)


def klasifikasi_warna(d):
    dL, da, db = [float(x) for x in d]
    sudut = np.degrees(np.arctan2(db, da))            # arah warna (hue) di bidang a-b
    kroma = np.hypot(da, db)                          # kekuatan warna
    if dL < -55 and kroma < 16:
        return "Hitam"
    if kroma < 7:
        return "Abu-abu"
    if -140 <= sudut <= -85:
        return "Biru"
    if 80 <= sudut <= 112:
        return "Kuning"
    if -85 < sudut < -33:
        return "Ungu"
    if -33 <= sudut <= 45:
        if kroma > 35:
            return "Merah" if sudut > 5 else "Pink"
        return "Soft pink"
    if sudut > 112 or sudut < -140:
        return "Mint" if kroma < 25 else "Hijau"
    return "Oranye"


# ----------------------------------------------------------- pipeline lengkap
def klasifikasi_objek(kontur, lab, latar, eps_halus=0.02, eps_kasar=0.04):
    """Beri label (bentuk, warna) pada setiap kontur dan buang objek yang tidak masuk akal."""
    objek = []
    for c in sorted(kontur, key=lambda k: (cv2.boundingRect(k)[1] // 60, cv2.boundingRect(k)[0])):
        f = fitur_bentuk(c, eps_halus, eps_kasar)
        bentuk = klasifikasi_bentuk(c, eps_halus, eps_kasar, f)
        warna = klasifikasi_warna(selisih_lab(lab, latar, c))
        if warna == "Abu-abu":
            continue                                   # tidak berwarna = bukan potongan kertas
        if warna == "Hitam" and bentuk != "Lingkaran":
            continue                                   # benda hitam yang valid hanya lingkaran
        objek.append(dict(kontur=c, bentuk=bentuk, warna=warna, fitur=f))
    return objek


def analisis_citra(img, eps_halus=0.02, eps_kasar=0.04):
    mask, kontur, lab, latar = deteksi(img)
    objek = klasifikasi_objek(kontur, lab, latar, eps_halus, eps_kasar)
    return dict(mask=mask, kontur=kontur, objek=objek, lab=lab, latar=latar)


def gambar_hasil(img, objek):
    out = img.copy()
    for o in objek:
        c = o["kontur"]
        cv2.drawContours(out, [c], -1, (0, 255, 0), 2)
        cv2.drawContours(out, [cv2.convexHull(c)], -1, (255, 0, 255), 1)
        x, y, w, h = cv2.boundingRect(c)
        teks = f'{o["bentuk"]} - {o["warna"]}'
        pos = (x, max(y - 6, 14))
        cv2.putText(out, teks, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, teks, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ----------------------------------------------------------- mode objek besar (foto di luar dataset)
RENTANG_HSV = {
    "Merah":     [((0, 100, 60), (6, 255, 255)), ((176, 100, 60), (179, 255, 255))],
    "Pink":      [((158, 100, 60), (175, 255, 255))],
    "Soft pink": [((159, 25, 150), (179, 99, 255)), ((0, 25, 150), (8, 99, 255))],
    "Ungu":      [((120, 20, 40), (158, 120, 255))],
    "Kuning":    [((18, 100, 60), (35, 255, 255))],
    "Hijau":     [((36, 85, 20), (85, 255, 200))],
    "Mint":      [((30, 10, 120), (60, 84, 255))],
    "Biru":      [((86, 100, 40), (130, 255, 255))],
    "Hitam":     [((0, 0, 0), (179, 110, 74))],
}


def warna_hsv_inrange(hsv, c):
    """Warna = rentang HSV dengan cakupan terbesar (cv2.inRange) di dalam objek."""
    er = mask_dalam(hsv.shape, c)
    total = max((er > 0).sum(), 1)
    skor = {}
    for nama, daftar in RENTANG_HSV.items():
        m = np.zeros(er.shape, np.uint8)
        for bawah, atas in daftar:
            m |= cv2.inRange(hsv, np.array(bawah), np.array(atas))
        skor[nama] = ((m > 0) & (er > 0)).sum() / total
    return max(skor, key=skor.get)


def cari_seed(img, min_luas=0.001, maks=4):
    """Bagian yang sangat jenuh (S>=140) dan cerah, diurutkan dari yang terbesar."""
    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (7, 7), 0), cv2.COLOR_BGR2HSV)
    seed = cv2.inRange(hsv, np.array([0, 140, 140]), np.array([179, 255, 255]))
    seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    n, lbl, st, _ = cv2.connectedComponentsWithStats(seed)
    ids = [i for i in np.argsort(-st[1:, 4]) + 1 if st[i, 4] >= min_luas * seed.size][:maks]
    return ids, lbl, hsv


def analisis_objek_besar(img, min_luas_objek=0.20):
    """Mode objek besar (misalnya sticky note yang memenuhi sepertiga foto, latar ramai).
    Estimasi latar tidak cocok karena objek mendominasi foto, jadi dipakai region growing:
    seed = bagian paling jenuh; wilayah = piksel yang warnanya (Lab) dekat dengan warna seed.
    Hanya wilayah yang luasnya > 20% foto dan padat (solidity > 0,9) yang dianggap 'objek besar' (objek kecil ditangani pipeline normal)."""
    ids, lbl, hsv = cari_seed(img)
    blur = cv2.GaussianBlur(img, (7, 7), 0)
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB).astype(np.float32)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask_total = np.zeros(img.shape[:2], np.uint8)
    objek = []
    for i in ids:
        titik = np.argwhere(lbl == i)[0][::-1]                         # satu piksel seed (x, y)
        med = np.median(lab[lbl == i], axis=0)
        d = np.sqrt((lab[:, :, 1] - med[1]) ** 2 + (lab[:, :, 2] - med[2]) ** 2 + (0.5 * (lab[:, :, 0] - med[0])) ** 2)
        reg = (d < 30).astype(np.uint8) * 255
        reg = cv2.morphologyEx(reg, cv2.MORPH_CLOSE, k, iterations=2)  # tutup lubang tulisan tangan
        reg = cv2.morphologyEx(reg, cv2.MORPH_OPEN, k)
        cs, _ = cv2.findContours(reg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cs = [c for c in cs if cv2.pointPolygonTest(c, (float(titik[0]), float(titik[1])), False) >= 0]
        if not cs:
            continue
        c = max(cs, key=cv2.contourArea)
        if cv2.contourArea(c) < min_luas_objek * reg.size or luas_solid(c)[1] < 0.9:
            continue
        cv2.drawContours(mask_total, [c], -1, 255, -1)
        f = fitur_bentuk(c)
        objek.append(dict(kontur=c, bentuk=klasifikasi_bentuk(c, f=f), warna=warna_hsv_inrange(hsv, c), fitur=f))
    return dict(mask=mask_total, kontur=[o["kontur"] for o in objek], objek=objek, mode="besar")


def analisis_otomatis(img):
    """Ada objek yang menutupi >20% foto -> mode besar; selain itu pipeline normal."""
    hasil = analisis_objek_besar(img)
    if hasil["objek"]:
        return hasil
    hasil = analisis_citra(img)
    hasil["mode"] = "normal"
    return hasil

# =============================================================== WEBSITE (Streamlit)
def decode_upload(data):
    """Bytes file unggahan -> citra BGR (orientasi EXIF diterapkan OpenCV)."""
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def perkecil(img, max_sisi=MAX_SISI):
    h, w = img.shape[:2]
    s = min(1.0, max_sisi / max(h, w))
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img


def proses(data, mode):
    """Jalankan pipeline. mode: 'Otomatis' | 'Normal (potongan kecil)' | 'Objek besar'."""
    img = decode_upload(data)
    if img is None:
        return None
    img = perkecil(img)
    if mode.startswith("Normal"):
        hasil = analisis_citra(img)
        hasil["mode"] = "normal"
    elif mode.startswith("Objek besar"):
        hasil = analisis_objek_besar(img)
    else:
        hasil = analisis_otomatis(img)
    hasil["gambar"] = img
    hasil["hasil_img"] = gambar_hasil(img, hasil["objek"])
    return hasil


def bgr2rgb(x):
    return cv2.cvtColor(x, cv2.COLOR_BGR2RGB)


def tampil_gambar(img, **kw):
    """st.image selebar kolom; kompatibel dengan versi Streamlit lama (use_container_width) dan baru (width='stretch')."""
    try:
        st.image(img, width="stretch", **kw)
    except TypeError:
        st.image(img, use_container_width=True, **kw)


def tampil_tabel(baris):
    try:
        st.dataframe(baris, width="stretch", hide_index=True)
    except TypeError:
        st.dataframe(baris, use_container_width=True, hide_index=True)


def main():
    st.set_page_config(page_title="Klasifikasi Bentuk dan Warna", page_icon="🔷", layout="wide")
    st.title("🔷 Klasifikasi Bentuk dan Warna")
    st.caption("UTS Praktikum Visi Komputer - Kelompok 8 - Tema 8. Unggah foto benda datar (potongan kertas), "
               "program menamai tiap benda dengan bentuk dan warnanya, misalnya **Segitiga - Merah**.")

    with st.sidebar:
        st.header("Pengaturan")
        mode = st.radio(
            "Mode deteksi",
            ["Otomatis", "Normal (potongan kecil)", "Objek besar"],
            help="Otomatis memilih sendiri. Normal untuk potongan kertas kecil di atas latar polos. "
                 "Objek besar untuk benda yang memenuhi sebagian besar foto (misalnya sticky note).")
        tampil_mask = st.checkbox("Tampilkan mask dan detail", value=True)
        st.markdown("---")
        st.markdown("**Bentuk:** segitiga, persegi, persegi panjang, lingkaran, bintang, love  \n"
                    "**Warna:** merah, kuning, hijau, biru, hitam, mint, ungu, pink, soft pink")
        st.markdown("**Tips foto:** dari atas, latar polos, objek tidak saling menempel dan tidak menyentuh tepi foto.")

    berkas = st.file_uploader("Unggah gambar (JPG / PNG)", type=["jpg", "jpeg", "png"])
    if berkas is None:
        st.info("Silakan unggah foto untuk memulai.")
        return

    data = berkas.getvalue()
    with st.spinner("Memproses gambar..."):
        hasil = proses(data, mode)
    if hasil is None:
        st.error("File tidak bisa dibaca sebagai gambar. Coba format JPG atau PNG (foto HEIC dari iPhone perlu diubah ke JPG dulu).")
        return

    objek = hasil["objek"]

    # ---- gambar asli dan hasil berdampingan
    kol1, kol2 = st.columns(2)
    with kol1:
        st.subheader("Gambar asli")
        tampil_gambar(bgr2rgb(hasil["gambar"]))
    with kol2:
        st.subheader("Gambar hasil")
        tampil_gambar(bgr2rgb(hasil["hasil_img"]))

    # ---- informasi hasil
    st.markdown("---")
    st.subheader("Informasi hasil")
    jml_bentuk = Counter(o["bentuk"] for o in objek)
    jml_warna = Counter(o["warna"] for o in objek)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Jumlah objek", len(objek))
    m2.metric("Jenis bentuk", len(jml_bentuk))
    m3.metric("Jenis warna", len(jml_warna))
    m4.metric("Mode dipakai", "Objek besar" if hasil.get("mode") == "besar" else "Normal")

    if not objek:
        st.warning("Tidak ada objek yang terdeteksi. Pastikan potongan kertas terlihat jelas, latar polos, "
                   "dan objek tidak menyentuh tepi foto. Coba ganti mode di sidebar.")
        return

    t1, t2 = st.columns([3, 2])
    with t1:
        st.markdown("**Daftar objek**")
        st.table([{"No": i, "Bentuk": o["bentuk"], "Warna": o["warna"], "Label": f'{o["bentuk"]} - {o["warna"]}'}
                  for i, o in enumerate(objek, 1)])
    with t2:
        st.markdown("**Rekap per label**")
        rekap = Counter(f'{o["bentuk"]} - {o["warna"]}' for o in objek)
        st.table([{"Label": k, "Jumlah": v} for k, v in sorted(rekap.items(), key=lambda kv: (-kv[1], kv[0]))])
        st.markdown("**Rekap per bentuk**")
        st.table([{"Bentuk": k, "Jumlah": v} for k, v in sorted(jml_bentuk.items())])
        st.markdown("**Rekap per warna**")
        st.table([{"Warna": k, "Jumlah": v} for k, v in sorted(jml_warna.items())])

    if tampil_mask:
        with st.expander("Mask segmentasi dan detail"):
            tampil_gambar(hasil["mask"], caption="Mask (putih = bagian yang terdeteksi sebagai objek)")
            tampil_tabel([{"No": i, "Bentuk": o["bentuk"], "Warna": o["warna"],
                           "Luas (px)": int(o["fitur"]["luas"]),
                           "Solidity": round(float(o["fitur"]["solid"]), 2),
                           "Sirkularitas": round(float(o["fitur"]["circ"]), 2),
                           "Titik (eps 0,02)": o["fitur"]["n_halus"],
                           "Titik (eps 0,04)": o["fitur"]["n_kasar"],
                           "Lekukan dalam": o["fitur"]["n_cekung"]} for i, o in enumerate(objek, 1)])

    ok, buf = cv2.imencode(".png", hasil["hasil_img"])
    if ok:
        st.download_button("Unduh gambar hasil (PNG)", buf.tobytes(), file_name="hasil_klasifikasi.png", mime="image/png")


if __name__ == "__main__":
    main()
