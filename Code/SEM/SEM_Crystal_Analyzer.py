"""
======================================================
SEM Drug Crystal Analyzer
======================================================

출력 파일 (out_dir 폴더에 저장):
    Crystal_Analysis.png      — 8-패널 분석 결과 이미지
    Crystal_Measurements.csv  — 결정별 측정값 (CV + CNN 예측)
    Global_Metrics.csv        — 전체 이미지 수준 지표
    Loss_Curve.png            — CNN 학습 손실 곡선
    Crystal_CNN.pth           — 학습된 CNN 모델 가중치

═══════════════════════════════════════════════════════
표면 거칠기 지표 (grayscale 강도를 높이의 대용값으로 사용):
    Rq  : RMS 거칠기 (Root Mean Square roughness)
    Ra  : 산술 평균 거칠기 (Arithmetic Mean roughness)
    Rsk : 스큐니스 (분포 비대칭도 — 양수=봉우리, 음수=골)
    Rku : 초과 첨도 (분포 뾰족함 — 0=정규분포)
    Rv  : 최대 골 깊이 (Maximum Valley depth)
    Rp  : 최대 봉우리 높이 (Maximum Peak height)
    Rt  : 전체 높이 Rp+Rv (Total height)
    Rc  : 평균 봉우리 높이 (Mean Peak height, 양 편차 평균)

방위각/극각 지표 (이미지 기울기 → 면법선 벡터 기반):
    FPO  : 평균 극면 방향 (Mean Polar Facet Orientation, °)
    MFOV : 극면 방향 분산 (Variation of Polar Facet Orientation, °)
    FAD  : 방위각 지배 방향 (Dominant Azimuthal Facet Direction, °)
    MRV  : 평균 결과벡터 길이 (Mean Resultant Vector, 0~1)
    SA   : 표면적 (Surface Area, µm²)

형태 지표 (Shape Descriptors):
    circularity  : 원형도 4π*area/perimeter²  (1=완전 원)
    solidity     : 볼록 충실도 area/convex_area  (1=볼록 다각형)
    compactness  : 컴팩트 비율 equiv_diam/major_axis  (1=원)
"""

# ─── 표준 라이브러리 ─────────────────────────────────────────
import os, sys, argparse, warnings
import csv
import pandas as pd
import numpy as np
import cv2
from PIL import Image


# ─── 수치/이미지 처리 ────────────────────────────────────────
from scipy.ndimage import label as nd_label, distance_transform_edt
from scipy.stats import circmean, circstd
from skimage.measure import regionprops
from skimage.transform import radon, rescale
from skimage.filters import (threshold_otsu, threshold_local,
                              frangi, scharr_h, scharr_v)
from skimage.morphology import (remove_small_objects, closing, opening, disk)
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ─── PyTorch (딥러닝) ────────────────────────────────────────
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models

# ※ 위 항목들과 동일 — 중복 import 제거됨

# ─── GUI ─────────────────────────────────────────────────
import tkinter as tk
from tkinter import filedialog, messagebox

# ───불필요한 경고 억제 (skimage 버전 호환 경고 등)────────────────
warnings.filterwarnings("ignore")

# ───기본 디렉토리 (파일 위치) 지정───────────────────────────────
from pathlib import Path
_workspace_root = Path(__file__).resolve().parents[2]

# ───기본 입출력 변수 설정───────────────────────────────
class NGI_Feature:
    """
    프로젝트 전반에서 공유하는 컬럼명/피처명 목록을 정의하는 설정 클래스.
    마스터 CSV 파일(Data/Master/)에서 컬럼을 읽어 클래스 변수로 저장한다.
    CSV 파일이 없을 경우 빈 리스트로 대체하여 프로그램이 중단되지 않도록 처리.
    """
    # SEM 이미지에서 추출하는 표면 거칠기 + 방위각 지표 컬럼명 목록
    SEM_COLS = ["Rq", "Ra", "Rsk", "Rku", "Rv", "Rp", "Rt", "Rc",
                "FPO", "MFOV", "FAD", "MRV", "SA"]

    # 실험 조건 컬럼명 (혼합 시간, 유량 등)
    EXPERIMENT_CONDITIONS = ["Blending_Time", "LPM"]

    # ── CSV 로딩 (실패 시 빈 리스트로 대체) ──────────────────
    # Python 3에서 클래스 본문의 list comprehension은 클래스 스코프를
    # 참조할 수 없으므로, for 루프로 RESULT_COLS를 구성한다.
    try:
        # PXRD 선택 피처 목록 (첫 번째 열 = 인덱스이므로 제거)
        PXRD_COLS = pd.read_csv(
            _workspace_root / 'Data' / 'Master' / 'Selected_PXRD_Features.csv'
        ).columns.to_list()[1:]

        # NGI 측정 결과 컬럼 목록 (인덱스 열 제거)
        _all_result = pd.read_csv(
            _workspace_root / 'Data' / 'Master' / 'Nintendanib_NGI.csv'
        ).columns.to_list()[1:]

        # SEM / PXRD / 실험조건 컬럼을 제외한 순수 NGI 결과 컬럼만 추출
        # ※ for 루프 사용: 클래스 스코프 변수(SEM_COLS 등)를 안전하게 참조
        _exclude = set(SEM_COLS) | set(PXRD_COLS) | {"Blending_Time", "LPM"}
        RESULT_COLS = []
        for _col in _all_result:
            if _col not in _exclude:
                RESULT_COLS.append(_col)
        del _all_result, _exclude
        print("[NGI_Feature] 마스터 CSV 로드 완료.")

    except FileNotFoundError as _e:
        # 마스터 CSV가 없으면 빈 리스트로 초기화 (프로그램은 계속 실행)
        print(f"[NGI_Feature] 마스터 CSV 없음 → 빈 목록으로 대체: {_e}")
        PXRD_COLS   = []
        RESULT_COLS = []

# ========================================================
# 1. Choose Image file (GUI)
# ========================================================
class Image_Model:
    """
    SEM 이미지 분석 메인 클래스.
    고전적 컴퓨터 비전(CV) + CNN 딥러닝을 결합하여 결정(crystal) 특성을 추출한다.

    주요 파이프라인:
        choose_png_file()  → 사용자가 GUI로 이미지 파일을 선택
        crop_enhance_img() → 이미지 크롭 + 대비 강화
        segment_crystals() → 결정 영역 이진 분리 + 라벨링
    """

    def __init__(self, image_path, image_scale, footer_pixel, min_area_px,
                 z_arr, um_per_px, gray_patch, mask_patch):
        """
        Parameters
        ----------
        image_path   : 분석할 PNG 파일 경로 (str)
        image_scale  : 이미지 배율 (현재 미사용, 호환성 유지용)
        footer_pixel : SEM 이미지 하단 정보 표시줄 높이 (픽셀, 기본 150)
        min_area_px  : 결정으로 인정하는 최소 면적 (픽셀², 기본 30)
        z_arr        : 거칠기 계산용 높이 배열 (미사용, 호환성 유지)
        um_per_px    : 픽셀당 실제 길이 (µm/px) — 스케일 변환에 사용
        gray_patch   : 개별 결정 패치 gray 이미지 (미사용)
        mask_patch   : 개별 결정 패치 마스크 (미사용)
        """
        self.image_path    = image_path
        self.image_scale   = image_scale
        self.footer_pixel  = footer_pixel
        self.min_area_pixel = min_area_px
        self.z_arr         = z_arr
        self.um_per_px     = um_per_px
        self.gray_patch    = gray_patch
        self.mask_patch    = mask_patch
    
    # ── Step 0: 이미지 파일 선택 (GUI) ─────────────────────

    def choose_png_file(self):
        """
        tkinter GUI 창을 열어 사용자가 PNG 파일을 선택하도록 한다.
        파일 선택 후 스케일(pixels/µm)을 입력받아 반환한다.

        Returns
        -------
        image_path  : 선택된 이미지 파일의 절대 경로 (str)
        image_scale : 사용자가 입력한 pixels/µm 값 (float)

        ※ 메뉴에서는 이 메서드 대신 filedialog.askopenfilename()을 직접
           사용하므로 현재 이 메서드는 레거시 코드로 유지됨.
        """
        image_path: str | None = None
        image_scale: float | None = None

        def press1():
            nonlocal image_path
            f = filedialog.askopenfile(
                title="Choose Image File you want to convert!",
                filetypes=(("Image files", "*.png"), ("All Files", "*.*")),
            )
            if f is None:
                return
            
            image_path = f.name
            L2.configure(text="Your file is: " + image_path) # pyright: ignore[reportOperatorIssue]
            B2.grid(row=7, column=0, ipadx=40, padx=10, pady=20)

        def press2():
            window.destroy()

        def press3():
            window.destroy()
            sys.exit()

        def press_ok():
            nonlocal image_scale
            try:
                image_scale = float(E1.get())
                window.destroy()
            except ValueError:
                messagebox.showwarning("Input Error", "Please enter a valid number for pixels/µm")

        window = tk.Tk()
        window.title("SEM Image Analysis Setup")

        tk.Label(window, text="1. Select SEM Image File", font=("Arial", 12, "bold")).grid(row=0, column=0, pady=10)
        
        B1 = tk.Button(window, text="Browse File", command=press1)
        B1.grid(row=1, column=0, pady=5)

        L2 = tk.Label(window, text="No file selected", fg="red")
        L2.grid(row=2, column=0, pady=5)

        tk.Label(window, text="2. Enter Scale (pixels/µm):", font=("Arial", 10)).grid(row=3, column=0, pady=10)
        E1 = tk.Entry(window)
        E1.insert(0, "8.1")  # Default value
        E1.grid(row=4, column=0, pady=5)

        B2 = tk.Button(window, text="Run Analysis", state="disabled", command=press_ok)
        B2.grid(row=5, column=0, pady=20)

        B3 = tk.Button(window, text="Exit", command=press3)
        B3.grid(row=6, column=0, pady=5)

        # Update press function to enable OK button
        _old_press = press1
        def press():
            _old_press()
            if image_path:
                B2.config(state="normal")


        window = tk.Tk()
        window.title("Choose SEM Image file")

        L1 = tk.Label(window, text="Choose Image File you want to convert", font=20) # type: ignore
        L1.grid(row=1, column=0, padx=200, pady=30)

        B1 = tk.Button(window, text="Find File", font=12, command=press) # type: ignore
        B1.grid(row=5, column=0, ipadx=40, padx=10, pady=20)

        L2 = tk.Label(window, text="", font=12, height=3) # type: ignore
        L2.grid(row=6, column=0, padx=10, pady=20)

        B2 = tk.Button(window, text="OK", font=12, command=press2) # type: ignore
        B3 = tk.Button(window, text="Cancel", font=12, command=press3) # type: ignore
        B3.grid(row=8, column=0, ipadx=40, padx=10, pady=20)

        window.mainloop()

        return image_path, image_scale # type: ignore
    
    # ── Step 1: 전처리 ──────────────────────────────────────

    def crop_enhance_img(self, image, footer_px=150):
        """
        SEM 이미지를 불러와 하단 정보 표시줄을 제거하고 대비를 강화한다.

        Parameters
        ----------
        image     : 이미지 파일 경로 (str)
        footer_px : 하단 정보 표시줄 높이 (픽셀). SEM 기기마다 다름, 기본 150px.

        Returns
        -------
        gray : 원본 그레이스케일 이미지 (uint8 ndarray) — 시각화에 사용
        enh  : CLAHE + Gaussian 처리된 강화 이미지 — 세그먼테이션에 사용

        처리 단계:
            1) 하단 footer_px 픽셀 제거 (SEM 메타데이터 표시줄)
            2) BGR → Grayscale 변환
            3) CLAHE (Contrast Limited Adaptive Histogram Equalization):
               clipLimit=3.0, tileGridSize=(8,8) — 국소 대비 향상
            4) Gaussian 블러 (3×3): 고주파 잡음 제거
        """
        FOOTER_PX = footer_px
        path = image
        bgr = cv2.imread(path)
        h, w = bgr.shape[:2]
        bgr = bgr[:h - FOOTER_PX]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        # CLAHE
        c1 = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)

        # Light Gaussian denoise
        enh = cv2.GaussianBlur(c1, (3, 3), 0)
        return gray, enh
    
    # ── Step 2: 세그먼테이션 (결정 영역 분리) ────────────────

    def segment_crystals(self, enh, min_area_px=30):
        """
        다단계 세그먼테이션으로 결정(crystal) 영역을 이진 마스크로 분리한다.

        Parameters
        ----------
        enh         : crop_enhance_img()가 반환한 강화 이미지 (uint8 ndarray)
        min_area_px : 결정으로 인정하는 최소 면적 (픽셀²). 잡음 제거 역할.

        Returns
        -------
        binary  : 결정=1, 배경=0 인 이진 마스크 (uint8 ndarray)
        labeled : 각 결정 영역에 고유 정수 라벨을 부여한 배열

        처리 단계:
            1) Frangi vesselness 필터:
               바늘/막대 모양의 능선(ridge) 구조를 강조.
               sigmas=1~3: 검출할 능선 두께 범위 (픽셀 단위).
               black_ridges=False: 밝은 능선(결정) 검출.
            2) 강화 이미지 + Frangi 결과를 55:45 비율로 혼합(blend):
               단순 이진화보다 결정 경계가 더 선명하게 분리됨.
            3) Otsu 임계값 이진화:
               히스토그램 이중봉 분리 최적 임계값 자동 계산.
            4) 열림(opening) 연산: 돌출 잡음 제거.
               닫힘(closing) 연산: 결정 내부 빈 구멍 메우기.
               remove_small_objects: min_area_px 미만 소형 잡음 제거.
            5) Watershed 분리:
               서로 닿아 있는 결정을 개별 영역으로 분리.
               - 거리 변환(distance_transform_edt): 각 픽셀의 배경까지 거리
               - peak_local_max: 거리 맵에서 극대점 → 각 결정의 씨앗(seed)
               - watershed: 씨앗에서 물을 채우듯 영역을 확장하여 분리
        """
        img_f  = enh.astype(np.float32) / 255.0          # [0,1] 정규화
        ridge  = frangi(img_f, sigmas=range(1, 4), black_ridges=False)
        r8     = (ridge / (ridge.max() + 1e-9) * 255).astype(np.uint8)

        # 강화 이미지(55%)와 Frangi 능선 맵(45%)을 혼합
        blend  = cv2.addWeighted(enh, 0.55, r8, 0.45, 0)

        # Otsu on blend
        thr    = threshold_otsu(blend)
        binary = (blend > thr).astype(bool)

        # Clean
        binary = opening(binary, disk(1))
        binary = closing(binary, disk(2))
        binary = remove_small_objects(binary, min_size=min_area_px)

        # Watershed to separate touching blobs
        dist   = distance_transform_edt(binary)
        # local max as seeds
        coords = peak_local_max(dist, min_distance=6, labels=binary)
        mask   = np.zeros(dist.shape, dtype=bool) # type: ignore
        mask[tuple(coords.T)] = True
        seeds, _ = nd_label(mask) # type: ignore
        labeled  = watershed(-dist, seeds, mask=binary) # type: ignore

        return binary.astype(np.uint8), labeled
    

# ══════════════════════════════════════════════════════════
# 3.  SURFACE METRICS  (module-level)
# ══════════════════════════════════════════════════════════
def roughness_metrics(z_arr, um_per_px):
    """
    grayscale 강도(intensity)를 표면 높이(height)의 대용값으로 사용하여
    ISO 표면 거칠기 파라미터를 계산한다.

    Parameters
    ----------
    z_arr     : 1-D 또는 2-D float 배열. 픽셀 강도값(0~255).
                결정 패치 영역의 마스크된 강도값을 입력.
    um_per_px : 픽셀당 µm 변환 계수. 강도 단위의 거칠기를 µm 단위로 환산.

    Returns
    -------
    dict : {Rq, Ra, Rsk, Rku, Rv, Rp, Rt, Rc} — 각 값의 단위는 µm
           (단, Rsk·Rku는 무차원 비율이므로 스케일링 없음)

    계산 방법:
        dev = z - mean(z)           ← 평균에서의 편차 프로파일
        Rq  = sqrt(mean(dev²))      ← RMS 거칠기 (표준편차와 동일)
        Ra  = mean(|dev|)           ← 산술 평균 거칠기
        Rsk = mean(dev³)/Rq³        ← 스큐니스: 양수→봉우리 우세, 음수→골 우세
        Rku = mean(dev⁴)/Rq⁴ - 3   ← 초과 첨도: 0=정규분포, 양수=뾰족
        Rp  = max(dev)              ← 최대 봉우리 높이
        Rv  = -min(dev)             ← 최대 골 깊이
        Rt  = Rp + Rv               ← 전체 높이
        Rc  = mean(dev > 0 인 dev)  ← 평균 봉우리 높이
    """
    z   = z_arr.astype(np.float64).ravel()   # 1-D 평탄화
    zm  = z.mean()                            # 평균 강도
    dev = z - zm                              # 평균 편차 프로파일

    Rq  = np.sqrt((dev**2).mean())            # RMS 거칠기
    Ra  = np.abs(dev).mean()                  # 산술 평균 거칠기
    Rsk = (dev**3).mean() / (Rq**3 + 1e-12)  # 스큐니스 (0으로 나누기 방지용 1e-12)
    Rku = (dev**4).mean() / (Rq**4 + 1e-12) - 3.0  # 초과 첨도
    Rp  = float(dev.max())                    # 최대 봉우리
    Rv  = float(-dev.min())                   # 최대 골
    Rt  = Rp + Rv                             # 전체 높이

    # Rc: 평균선 위 편차(양의 편차)의 평균 → 평균 봉우리 높이
    pos = dev[dev > 0]
    Rc  = float(pos.mean()) if len(pos) else 0.0

    # 강도 단위 → µm 변환 (um_per_px를 스케일 계수로 사용)
    scale = um_per_px
    return dict(Rq=Rq*scale, Ra=Ra*scale, Rsk=Rsk, Rku=Rku,
                Rv=Rv*scale, Rp=Rp*scale, Rt=Rt*scale, Rc=Rc*scale)


def orientation_metrics(gray_patch, mask_patch, um_per_px):
    """
    Scharr 기울기를 이용해 결정 표면의 방위각/극각 지표를 계산한다.
    grayscale 강도를 높이 z로 가정하여 면법선(facet normal) 방향을 추정한다.

    Parameters
    ----------
    gray_patch : 결정 바운딩 박스 내 gray 이미지 패치 (float32 ndarray)
    mask_patch : 결정 영역만 True인 boolean 마스크 (같은 크기)
    um_per_px  : 픽셀당 µm — 표면적(SA) 계산에 사용

    Returns
    -------
    dict : {FPO, MFOV, FAD, MRV, SA}

    원리:
        gx = scharr_v(z)  : x 방향 기울기 ∂z/∂x  (수직 Scharr 필터)
        gy = scharr_h(z)  : y 방향 기울기 ∂z/∂y  (수평 Scharr 필터)
        면법선 n ≈ (-gx, -gy, 1) / |n|

        극각 θ = arctan2(|∇z|, 1)   [0°=수평, 90°=수직 벽면]
        FPO  = mean(θ)   — 평균 극면 방향
        MFOV = std(θ)    — 극면 방향 분산 (균일할수록 작음)

        방위각 φ = arctan2(gy, gx)   [-π, π]
        2φ 트릭: 방위각은 180° 주기이므로 2φ로 원형 통계 처리
        FAD  = (arctan2(mean(sin2φ), mean(cos2φ)) / 2) % 180°  — 지배 방위
        MRV  = sqrt(mean(sin2φ)² + mean(cos2φ)²)               — 방향 정렬도

        SA = Σ sqrt(1 + gx² + gy²) · px_area   — 실제 표면적 (µm²)

    필터링:
        - 마스크 내부 픽셀만 사용
        - 기울기 크기 상위 40%만 사용 (배경 잡음 제거)
        - 유효 픽셀 < 3이면 NaN 반환
    """
    z      = gray_patch.astype(np.float32)
    gx     = scharr_v(z)   # ∂z/∂x
    gy     = scharr_h(z)   # ∂z/∂y

    # Only consider pixels inside crystal mask with significant gradient
    if mask_patch is not None:
        m = mask_patch.astype(bool)
    else:
        m = np.ones(z.shape, dtype=bool)

    mag = np.sqrt(gx**2 + gy**2)
    # Threshold: top 40 % gradient pixels
    thr = np.percentile(mag[m], 60) if m.sum() > 10 else 0
    m2  = m & (mag > thr)

    if m2.sum() < 3:
        return dict(FPO=np.nan, MFOV=np.nan, FAD=np.nan, MRV=np.nan, SA=np.nan)

    gxm = gx[m2];  gym = gy[m2]
    mag_m = mag[m2]

    # Polar angle (0° = flat, 90° = vertical wall)
    theta = np.degrees(np.arctan2(mag_m, 1.0))          # shape (N,)
    FPO   = float(theta.mean())
    MFOV  = float(theta.std())

    # Azimuthal angle (axial → fold to [0°,180°))
    phi_rad = np.arctan2(gym, gxm)                       # [-π, π]
    # Axial data: use 2φ trick for circular stats
    phi2    = 2 * phi_rad
    FAD_rad = np.arctan2(np.sin(phi2).mean(), np.cos(phi2).mean()) / 2
    FAD     = float(np.degrees(FAD_rad) % 180)
    # MRV = length of mean resultant vector of 2φ
    MRV     = float(np.sqrt(np.sin(phi2).mean()**2 + np.cos(phi2).mean()**2))

    # Surface area  SA = Σ sqrt(1 + gx² + gy²) · px_area
    px_area = um_per_px ** 2
    SA_mask = mask_patch if mask_patch is not None else np.ones_like(z, dtype=bool)
    all_gx  = gx[SA_mask.astype(bool)]
    all_gy  = gy[SA_mask.astype(bool)]
    SA      = float(np.sqrt(1 + all_gx**2 + all_gy**2).sum() * px_area)

    return dict(FPO=round(FPO,3), MFOV=round(MFOV,3),
                FAD=round(FAD,3), MRV=round(MRV,4), SA=round(SA,4))


# ══════════════════════════════════════════════════════════
# 4.  CRYSTAL MEASUREMENT  (per region)
# ══════════════════════════════════════════════════════════
def measure_crystals(gray_enh, labeled, um_per_px,
                     max_aspect=12.0, max_len_um=40.0, min_area_px=30):
    """
    라벨링된 각 결정 영역에 대해 형태/표면/방위각 지표를 계산한다.

    Parameters
    ----------
    gray_enh    : 전처리된 그레이 이미지 (uint8 ndarray) — 강도 정보 소스
    labeled     : 각 결정 영역에 정수 라벨이 부여된 배열 (segment_crystals 출력)
    um_per_px   : 픽셀 → µm 변환 계수
    max_aspect  : 허용 최대 종횡비 (기본 12). 이 초과 → 잡음으로 제외
    max_len_um  : 허용 최대 길이 (µm, 기본 40). 이 초과 → 비현실적 크기로 제외
    min_area_px : 허용 최소 면적 (픽셀², 기본 30). 미만 → 잡음으로 제외

    Returns
    -------
    records : list of dict — 결정당 다음 키를 포함:
        label, centroid_r/c : 라벨 번호, 무게중심 좌표
        area_um2, diam_um   : 면적(µm²), 등가 직경(µm)
        length_um, width_um : 장축/단축 길이(µm)
        aspect_ratio        : 장축/단축 비율
        azimuth_deg         : 방위각 φ ∈ [0°, 180°) — 결정 장축의 평면 내 방향
        polar_deg           : 극각 θ — 결정의 면외 기울기 추정치
        az_sin2, az_cos2    : sin(2φ), cos(2φ) — 방위각의 주기성 인코딩
        Rq, Ra, ... Rc      : 표면 거칠기 지표 (roughness_metrics 결과)
        FPO, MFOV, FAD, MRV, SA : 방위각/표면적 지표 (orientation_metrics 결과)
    """
    props   = regionprops(labeled, intensity_image=gray_enh)
    records = []
    skipped = 0

    for p in props:
        if p.area < min_area_px:    # 최소 면적 미만: 잡음 픽셀
            continue

        maj_px  = p.major_axis_length                               # 장축 길이 (픽셀)
        min_px  = p.minor_axis_length if p.minor_axis_length > 1 else 1.0  # 단축 (0 방지)
        aspect  = maj_px / min_px                                   # 종횡비

        # ── 필터: 비현실적 형태 제외 ─────────────────────
        if aspect > max_aspect:     # 지나치게 가는 섬유 → 잡음
            skipped += 1;  continue
        maj_um = maj_px * um_per_px
        if maj_um > max_len_um:
            skipped += 1;  continue

        min_um   = min_px  * um_per_px
        diam_um  = p.equivalent_diameter * um_per_px

        # Azimuthal φ: skimage orientation = angle of major-axis w.r.t. row-axis
        phi_rad  = np.pi / 2 - p.orientation
        phi_deg  = float(np.degrees(phi_rad) % 180)

        # 극각 θ 추정: 종횡비가 클수록 결정이 평면에 눕혀진 것으로 간주 (→ 90°)
        # arctan(1/aspect)=0이면 θ=90° (완전히 눕혀진 바늘)
        theta_deg = float(90.0 - np.degrees(np.arctan(1.0 / aspect)))

        # ── 결정 패치 추출 → 표면/방위 지표 계산 ─────────
        r0, c0, r1, c1 = p.bbox                               # 바운딩 박스
        patch_gray = gray_enh[r0:r1, c0:c1].astype(np.float32)
        patch_mask = labeled[r0:r1, c0:c1] == p.label        # 해당 결정 픽셀만 True

        # 거칠기 계산 시 마스크를 곱해 다른 결정 픽셀의 영향 제거
        rough = roughness_metrics(patch_gray * patch_mask, um_per_px)
        ormet = orientation_metrics(patch_gray, patch_mask, um_per_px)

        records.append({
            "label"        : p.label,           # 결정 고유 번호
            "centroid_r"   : p.centroid[0],     # 무게중심 행(row) 좌표 (픽셀)
            "centroid_c"   : p.centroid[1],     # 무게중심 열(col) 좌표 (픽셀)
            "area_um2"     : round(p.area * um_per_px**2, 4),  # 면적 (µm²)
            "diam_um"      : round(diam_um,  3),               # 등가 직경 (µm)
            "length_um"    : round(maj_um,   3),               # 장축 길이 (µm)
            "width_um"     : round(min_um,   3),               # 단축 길이 (µm)
            "aspect_ratio" : round(aspect,   3),               # 종횡비 (무차원)
            "azimuth_deg"  : round(phi_deg,  2),               # 방위각 φ (°)
            "polar_deg"    : round(theta_deg,2),               # 극각 θ (°)
            "az_sin2"      : round(np.sin(2*phi_rad), 4),      # sin(2φ) - 방향 인코딩
            "az_cos2"      : round(np.cos(2*phi_rad), 4),      # cos(2φ) - 방향 인코딩
            # roughness_metrics 결과 언팩 (Rq, Ra, Rsk, Rku, Rv, Rp, Rt, Rc)
            **{k: round(v, 4) if isinstance(v, float) else v
               for k, v in rough.items()},
            # orientation_metrics 결과 언팩 (FPO, MFOV, FAD, MRV, SA)
            **{k: round(v, 4) if isinstance(v, float) and not np.isnan(v) else v
               for k, v in ormet.items()},
        })

    print(f"  Crystals passed filters: {len(records)}  |  excluded: {skipped}")
    return records


# ══════════════════════════════════════════════════════════
# 5.  GLOBAL METRICS
# ══════════════════════════════════════════════════════════
def global_metrics(gray_enh, binary_mask, um_per_px):
    """
    이미지 전체(결정 영역만)에 대한 전역 표면 거칠기 + 방위각 지표를 계산한다.
    binary_mask를 곱하여 배경 픽셀을 0으로 처리한 뒤 roughness/orientation 계산.

    Returns : roughness_metrics + orientation_metrics 결과를 합친 dict
    """
    # 결정 영역 픽셀만 강도값 유지, 배경=0 처리
    rough = roughness_metrics(
        gray_enh.astype(np.float32) * binary_mask, um_per_px)
    ormet = orientation_metrics(gray_enh, binary_mask, um_per_px)
    return {**rough, **ormet}


# ══════════════════════════════════════════════════════════
# 6.  GLOBAL TEXTURE  (FFT + Radon)
# ══════════════════════════════════════════════════════════
def fft_dominant_angle(enh):
    """
    2D FFT 파워 스펙트럼에서 이미지의 지배적 방향각을 추출한다.

    원리:
        - 이미지에 2D FFT를 적용하면 파워 스펙트럼이 결정의 반복 방향에
          수직한 방향으로 에너지가 집중된다.
        - 각도별로 중심을 지나는 선을 따라 파워를 적산(radial integral)하여
          최대 파워 방향을 찾는다.
        - FFT 피크 방향은 결정 장축에 수직이므로 +90° 보정 후 반환한다.

    Returns : (dominant_angle_deg, power_spectrum_image)
    """
    f  = np.fft.fft2(enh.astype(np.float32))    # 2D 푸리에 변환
    ps = np.log1p(np.abs(np.fft.fftshift(f))**2) # 로그 파워 스펙트럼 (중심 이동)
    h, w = ps.shape
    cv2.circle(ps, (w//2, h//2), 5, 0, -1)       # DC 성분(중심) 마스킹
    angles = np.arange(0, 180)
    power  = []
    for ang in angles:
        rad = np.deg2rad(ang)
        # 각도 ang 방향으로 중심을 지나는 선 위의 픽셀 좌표 계산
        xs = (w//2 + np.arange(-w//3, w//3)*np.cos(rad)).astype(int)
        ys = (h//2 + np.arange(-w//3, w//3)*np.sin(rad)).astype(int)
        ok = (xs>=0)&(xs<w)&(ys>=0)&(ys<h)       # 이미지 경계 내 픽셀만
        power.append(ps[ys[ok], xs[ok]].mean())   # 해당 방향 평균 파워
    # 최대 파워 방향 + 90° = 결정 장축 방향 (FFT와 실제 방향은 수직)
    dom = (angles[np.argmax(power)] + 90) % 180
    return float(dom), ps


def radon_dominant_angle(enh):
    """
    Radon 변환의 사이노그램 분산에서 이미지의 지배적 방향각을 추출한다.

    원리:
        - Radon 변환: 각도 θ 방향으로 적분 투영(projection)한 결과 = 사이노그램
        - 결정이 많이 배열된 방향으로 투영하면 분산이 최대가 됨
        - 계산 속도를 위해 이미지를 0.25배로 축소(rescale)하여 처리

    Returns : dominant_angle_deg (float)
    """
    small    = rescale(enh.astype(float), 0.25, anti_aliasing=True)
    angles   = np.linspace(0, 180, 180, endpoint=False)
    sinogram = radon(small, theta=angles, circle=False)
    dom      = float(angles[np.argmax(sinogram.var(axis=0))])
    return dom


# ══════════════════════════════════════════════════════════
# 7.  CNN MODEL
# ══════════════════════════════════════════════════════════
# CNN이 예측하는 15개 출력 레이블 이름
# 거칠기(8개) + 방위각(5개) + 방향 인코딩(2개)
LABEL_KEYS = ["Rq","Ra","Rsk","Rku","Rv","Rp","Rt","Rc",
              "FPO","MFOV","FAD","MRV","SA","az_sin2","az_cos2"]

# 각 레이블의 (최솟값, 최댓값) — 2kx SEM 이미지에서의 물리적 범위
# CNN 출력을 [0,1]로 정규화/역정규화할 때 사용
LABEL_NORM = {
    "Rq": (0, 5),   "Ra": (0, 4),   "Rsk": (-5, 5),  "Rku": (-5, 15),
    "Rv": (0, 10),  "Rp": (0, 10),  "Rt":  (0, 20),  "Rc":  (0, 8),
    "FPO":  (0, 90),  "MFOV": (0, 45), "FAD": (0, 180),
    "MRV":  (0, 1),   "SA":   (0, 50),
    "az_sin2": (-1, 1), "az_cos2": (-1, 1),
}

# CNN 입력 패치 크기 (픽셀). 각 결정 중심에서 64×64 영역을 잘라 입력으로 사용.
PATCH_SIZE = 64

# 학습/추론 디바이스: CUDA GPU가 있으면 GPU, 없으면 CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CrystalPatchDataset(Dataset):
    """
    PyTorch Dataset — 각 결정의 64×64 패치를 추출하여 CNN 학습/추론에 제공한다.

    동작:
        - 결정 무게중심을 기준으로 PATCH_SIZE×PATCH_SIZE 영역을 잘라냄
        - 이미지 경계에 걸치는 경우 zero-padding으로 보완
        - 그레이스케일 이미지를 3채널로 복제 (ResNet 호환용)
        - LABEL_KEYS 중 NaN을 포함한 샘플은 자동 제외
    """
    def __init__(self, gray, records):
        self.gray    = gray
        # NaN 레이블이 있는 결정은 CNN 학습에서 제외 (안정성)
        self.records = [r for r in records
                        if all(not (isinstance(r.get(k), float) and np.isnan(r.get(k,0)))
                               for k in LABEL_KEYS)]
        self.H, self.W = gray.shape
        # 데이터 증강: 좌우/상하 반전 (결정 방향이 임의적이므로 유효한 증강)
        # ToTensor: [H,W] → [1,H,W] float32, Normalize: [0,1] → [-1,1]
        self.tf = T.Compose([
            T.RandomHorizontalFlip(),  # 50% 확률로 좌우 반전
            T.RandomVerticalFlip(),    # 50% 확률로 상하 반전
            T.ToTensor(),              # PIL/ndarray → Tensor
            T.Normalize([0.5], [0.5]), # 정규화: (x-0.5)/0.5 → [-1,1]
        ])

    def __len__(self):
        return len(self.records)  # 유효 결정 수

    def _norm(self, rec):
        """레이블 값을 LABEL_NORM 범위에 따라 [0, 1]로 정규화."""
        out = []
        for k in LABEL_KEYS:
            lo, hi = LABEL_NORM[k]
            # (값 - 최솟값) / (범위 + ε) 후 [0,1] 클리핑
            v = float(np.clip((rec[k] - lo) / (hi - lo + 1e-9), 0, 1))
            out.append(v)
        return torch.tensor(out, dtype=torch.float32)

    def __getitem__(self, idx):
        """idx번 결정의 패치 이미지(Tensor)와 정규화된 레이블(Tensor)을 반환."""
        rec  = self.records[idx]
        r, c = int(rec["centroid_r"]), int(rec["centroid_c"])  # 무게중심 픽셀 좌표
        half = PATCH_SIZE // 2                                  # 패치 절반 크기 = 32

        # 무게중심 ± half 범위의 패치 추출 (이미지 경계 클리핑)
        r0,r1 = max(r-half,0), min(r+half, self.H)
        c0,c1 = max(c-half,0), min(c+half, self.W)
        patch = self.gray[r0:r1, c0:c1]
        ph,pw = patch.shape

        # 경계 근처 결정은 64×64보다 작을 수 있으므로 zero-padding
        if ph < PATCH_SIZE or pw < PATCH_SIZE:
            pad = np.zeros((PATCH_SIZE, PATCH_SIZE), np.uint8)
            pad[:ph,:pw] = patch;  patch = pad

        # 그레이→PIL→Tensor 변환 후 1채널을 3채널로 복제 (모델 입력 형식 맞춤)
        x = self.tf(Image.fromarray(patch)).repeat(3,1,1) # type: ignore
        return x, self._norm(rec)


class CrystalCNN(nn.Module):
    """
    경량 맞춤형 CNN — CPU에서도 빠르게 동작 (파라미터 약 25만 개).

    구조:
        encoder : Conv→BN→ReLU→MaxPool 블록 × 4  (채널: 1→32→64→128→128)
                  AdaptiveAvgPool2d(1)            → [B, 128, 1, 1]
        head    : Flatten → Linear(128→64) → ReLU → Dropout(0.25)
                  → Linear(64→15) → Sigmoid     → [B, 15] in [0,1]

    입력 : [B, 1 또는 3, 64, 64]  (3채널이면 평균내어 1채널로 변환)
    출력 : [B, 15]  — LABEL_KEYS 순서로 정규화된 예측값
    """
    N_OUT = len(LABEL_KEYS)  # 15개 출력

    def __init__(self):
        super().__init__()
        def block(ci, co):
            """Conv(3×3) + BatchNorm + ReLU + MaxPool(2×2) 블록."""
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1, bias=False),  # 공간 크기 유지
                nn.BatchNorm2d(co),   # 배치 정규화: 학습 안정화
                nn.ReLU(True),        # 비선형 활성화
                nn.MaxPool2d(2))      # 해상도 절반으로 축소

        # 인코더: 64×64 → 32→16→8→4 → 1×1 (AdaptiveAvgPool)
        self.encoder = nn.Sequential(
            block(1, 32), block(32, 64),
            block(64, 128), block(128, 128),
            nn.AdaptiveAvgPool2d(1))  # 임의 크기 입력을 1×1로 압축

        # 회귀 헤드: 128차원 특징 → 15개 출력
        self.head = nn.Sequential(
            nn.Flatten(),                           # [B,128,1,1] → [B,128]
            nn.Linear(128, 64), nn.ReLU(True),
            nn.Dropout(0.25),                       # 과적합 방지
            nn.Linear(64, self.N_OUT), nn.Sigmoid()) # Sigmoid: 출력 [0,1]

    def forward(self, x):
        # 3채널 입력인 경우 채널 평균으로 1채널 변환 (그레이스케일 처리)
        if x.size(1) == 3:
            x = x.mean(dim=1, keepdim=True)
        return self.head(self.encoder(x))


def train_model(model, loader, n_epochs):
    """
    CrystalCNN을 CV 레이블(roughness + orientation)로 학습시킨다.

    Parameters
    ----------
    model    : CrystalCNN 인스턴스
    loader   : CrystalPatchDataset을 감싼 DataLoader
    n_epochs : 전체 학습 에포크 수 (기본 50)

    Returns
    -------
    history : 에포크별 평균 Huber Loss 리스트 (그래프 출력에 사용)

    학습 설정:
        - 옵티마이저 : AdamW (lr=0.001, weight_decay=0.0001)
          → Adam + 가중치 감쇠로 과적합 억제
        - 스케줄러  : CosineAnnealingLR
          → lr을 코사인 곡선으로 감소시켜 후반부 수렴 안정화
        - 손실 함수 : HuberLoss (δ=1 기본)
          → L1과 L2의 절충: 이상치에 강건하면서 작은 오차엔 부드럽게 작동
        - 그래디언트 클리핑 (max_norm=1.0): 폭발적 그래디언트 방지
    """
    model.to(DEVICE)
    opt     = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched   = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    loss_fn = nn.HuberLoss()    # 이상치에 강건한 회귀 손실
    history = []
    model.train()

    for ep in range(1, n_epochs+1):
        tot = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()                                     # 이전 그래디언트 초기화
            loss = loss_fn(model(xb), yb)                      # 순전파 + 손실 계산
            loss.backward()                                     # 역전파
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 그래디언트 클리핑
            opt.step()                                          # 파라미터 업데이트
            tot += loss.item() * xb.size(0)                    # 배치 가중 손실 누적
        sched.step()                                            # lr 스케줄 업데이트
        avg = tot / len(loader.dataset)                         # 에포크 평균 손실
        history.append(avg)
        if ep % 10 == 0 or ep == 1:
            print(f"  Epoch {ep:3d}/{n_epochs}  Huber: {avg:.5f}")
    return history


def decode_pred(pred_t, dataset):
    """
    CNN의 정규화된 출력([0,1])을 실제 물리 단위로 역변환한다.

    Parameters
    ----------
    pred_t  : CNN forward()의 출력 Tensor [1, 15] or [B, 15]
    dataset : CrystalPatchDataset (미사용이지만 호환성 유지)

    Returns
    -------
    dict : {LABEL_KEY: 역정규화된 값} — 단위는 µm, ° 등 물리 단위

    역변환 공식: value = pred_norm × (hi - lo) + lo
    """
    p   = pred_t.cpu().numpy().flatten()  # numpy로 변환 + 1-D
    out = {}
    for i, k in enumerate(LABEL_KEYS):
        lo, hi = LABEL_NORM[k]
        out[k] = float(p[i] * (hi - lo) + lo)  # 역정규화
    return out


# ══════════════════════════════════════════════════════════
# 8.  VISUALISATION
# ══════════════════════════════════════════════════════════
BG = "#0d1117"  # 배경색: 다크 네이비 (GitHub Dark 스타일)
FG = "white"    # 전경색: 흰색 (텍스트/축)

def save_figure(gray, records, binary, fft_angle, radon_angle, um_per_px, path):
    """
    8-패널 분석 결과 이미지를 PNG로 저장한다.

    패널 구성 (2행 × 4열):
        [0] 원본 SEM 이미지 + 스케일 바
        [1] 결정 방위각 오버레이 (HSV 색상 코딩)
        [2] 세그먼테이션 결과 (Frangi + Watershed)
        [3] 결정 크기 분포 히스토그램 (Length / Width)
        [4] Rq(RMS 거칠기) 히트맵
        [5] 방위각 Rose 다이어그램 + FFT/Radon 지배 방향
        [6] Rsk-Rku 산점도 (결정 크기별 색상)
        [7] FPO-MRV 산점도 (표면적 SA 색상)

    Parameters
    ----------
    gray       : 원본 그레이스케일 이미지
    records    : measure_crystals() 반환 리스트
    binary     : 이진 세그먼테이션 마스크
    fft_angle  : FFT 분석 지배 방향각 (°)
    radon_angle: Radon 분석 지배 방향각 (°)
    um_per_px  : 픽셀당 µm (스케일 바 계산에 사용)
    path       : 저장 경로 (.png)
    """
    fig = plt.figure(figsize=(20, 12), facecolor=BG)
    spec = fig.add_gridspec(2, 4, hspace=0.38, wspace=0.35)
    axes_list = [fig.add_subplot(spec[r, c]) for r in range(2) for c in range(4)]
    for ax in axes_list:
        ax.set_facecolor(BG)
        ax.tick_params(colors=FG, labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor("#444")

    # ── 0: SEM original ──────────────────────────────────
    ax = axes_list[0]
    ax.imshow(gray, cmap="gray")
    bar_px = int(20 / um_per_px)
    ax.annotate("", xy=(45+bar_px, gray.shape[0]-22),
                xytext=(45, gray.shape[0]-22),
                arrowprops=dict(arrowstyle="|-|", color="#ffd60a", lw=2))
    ax.text(45+bar_px//2, gray.shape[0]-10, "20 µm",
            color="#ffd60a", ha="center", fontsize=8)
    ax.set_title("Original SEM  (30 kV / 2.01 kx)", color=FG, fontsize=9)
    ax.axis("off")

    # ── 1: Orientation overlay ───────────────────────────
    ax = axes_list[1]
    ax.imshow(gray, cmap="gray", alpha=0.45)
    norm_c = Normalize(0, 180)
    cmap_h = plt.cm.hsv # type: ignore
    for rec in records:
        r, c   = rec["centroid_r"], rec["centroid_c"]
        phi_r  = np.radians(rec["azimuth_deg"])
        hl     = (rec["length_um"] / um_per_px) / 2
        dr, dc = hl*np.sin(phi_r), hl*np.cos(phi_r)
        col    = cmap_h(norm_c(rec["azimuth_deg"]))
        ax.plot([c-dc, c+dc], [r-dr, r+dr], color=col, lw=1.4, alpha=0.9)
    sm = ScalarMappable(cmap=cmap_h, norm=norm_c)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.038, pad=0.02)
    cb.set_label("Azimuth φ (°)", color=FG, fontsize=7)
    cb.ax.tick_params(colors=FG, labelsize=7)
    ax.set_title(f"Crystal Orientations  (n={len(records)})", color=FG, fontsize=9)
    ax.axis("off")

    # ── 2: Segmentation ──────────────────────────────────
    ax = axes_list[2]
    ax.imshow(binary, cmap="hot")
    ax.set_title("Segmentation (Frangi + Watershed)", color=FG, fontsize=9)
    ax.axis("off")

    # ── 3: Size distribution ─────────────────────────────
    ax = axes_list[3]
    lengths = [r["length_um"] for r in records]
    widths  = [r["width_um"]  for r in records]
    ax.hist(lengths, bins=22, color="#4cc9f0", alpha=0.85, label="Length")
    ax.hist(widths,  bins=22, color="#f72585", alpha=0.65, label="Width")
    ax.set_xlabel("Size (µm)", color=FG, fontsize=8)
    ax.set_ylabel("Count",     color=FG, fontsize=8)
    ax.set_title("Size Distribution", color=FG, fontsize=9)
    ax.legend(facecolor="#222", labelcolor=FG, fontsize=8)

    # ── 4: Roughness heatmap (Rq) ────────────────────────
    ax = axes_list[4]
    rq_img = np.zeros(gray.shape, dtype=np.float32)
    for rec in records:
        r, c = int(rec["centroid_r"]), int(rec["centroid_c"])
        rq_val = rec.get("Rq", 0)
        rad_px = max(2, int(rec.get("diam_um", 2) / um_per_px / 2))
        r0 = max(0, r-rad_px); r1 = min(gray.shape[0], r+rad_px)
        c0 = max(0, c-rad_px); c1 = min(gray.shape[1], c+rad_px)
        rq_img[r0:r1, c0:c1] = np.maximum(rq_img[r0:r1, c0:c1], rq_val)
    ax.imshow(gray, cmap="gray", alpha=0.35)
    im = ax.imshow(rq_img, cmap="plasma", alpha=0.65,
                   vmin=0, vmax=np.percentile(rq_img[rq_img>0], 98)+1e-9) # type: ignore
    cb2 = fig.colorbar(im, ax=ax, fraction=0.038, pad=0.02)
    cb2.set_label("Rq (µm)", color=FG, fontsize=7)
    cb2.ax.tick_params(colors=FG, labelsize=7)
    ax.set_title("Rq  (RMS roughness per crystal)", color=FG, fontsize=9)
    ax.axis("off")

    # ── 5: Rose diagram ──────────────────────────────────
    ax5 = axes_list[5]
    ax5.remove()
    ax5 = fig.add_subplot(spec[1, 1], polar=True, facecolor=BG)
    ax5.set_theta_zero_location("E");  ax5.set_theta_direction(1) # type: ignore
    phi_rad = np.radians([r["azimuth_deg"] for r in records])
    bins_r  = np.linspace(0, np.pi, 25)
    cnts, _ = np.histogram(phi_rad % np.pi, bins=bins_r)
    bw      = bins_r[1]-bins_r[0]
    bc      = 0.5*(bins_r[:-1]+bins_r[1:])
    ax5.bar(bc, cnts, width=bw, bottom=0, color=plt.cm.hsv(bc/np.pi), alpha=0.85) # type: ignore
    for ang, lbl, lc in [(np.radians(fft_angle),   "FFT",   "#00f5d4"),
                          (np.radians(radon_angle), "Radon", "#ffd60a")]:
        ax5.axvline(ang % np.pi, color=lc, lw=2, ls="--", label=lbl)
        ax5.axvline((ang+np.pi)%(2*np.pi), color=lc, lw=2, ls="--")
    ax5.tick_params(colors=FG, labelsize=7)
    ax5.set_facecolor(BG)
    ax5.legend(loc="lower right", bbox_to_anchor=(1.45,-0.1),
               facecolor="#222", labelcolor=FG, fontsize=7)
    ax5.set_title("Azimuth Rose Diagram", color=FG, fontsize=9, pad=12)

    # ── 6: Rsk / Rku scatter ─────────────────────────────
    ax = axes_list[6]
    rsk = [r.get("Rsk", 0) for r in records]
    rku = [r.get("Rku", 0) for r in records]
    sc = ax.scatter(rsk, rku, c=[r["length_um"] for r in records],
                    cmap="viridis", s=25, alpha=0.8)
    ax.axhline(0, color="#555", lw=0.8); ax.axvline(0, color="#555", lw=0.8)
    ax.set_xlabel("Rsk  (skewness)", color=FG, fontsize=8)
    ax.set_ylabel("Rku  (excess kurtosis)", color=FG, fontsize=8)
    ax.set_title("Roughness Character per Crystal", color=FG, fontsize=9)
    cb3 = fig.colorbar(sc, ax=ax, fraction=0.038, pad=0.02)
    cb3.set_label("Length (µm)", color=FG, fontsize=7)
    cb3.ax.tick_params(colors=FG, labelsize=7)

    # ── 7: FPO vs MRV ────────────────────────────────────
    ax = axes_list[7]
    fpo = [r.get("FPO", float("nan")) for r in records]
    mrv = [r.get("MRV", float("nan")) for r in records]
    sc2 = ax.scatter(fpo, mrv, c=[r.get("SA",0) for r in records],
                     cmap="magma", s=25, alpha=0.8)
    ax.set_xlabel("FPO  (°)", color=FG, fontsize=8)
    ax.set_ylabel("MRV  (orientation order)", color=FG, fontsize=8)
    ax.set_title("Polar Orientation vs Order", color=FG, fontsize=9)
    cb4 = fig.colorbar(sc2, ax=ax, fraction=0.038, pad=0.02)
    cb4.set_label("SA (µm²)", color=FG, fontsize=7)
    cb4.ax.tick_params(colors=FG, labelsize=7)

    plt.suptitle(
        f"SEM Drug Crystal Analysis  —  {um_per_px:.4f} µm/px  |  n={len(records)} crystals",
        color=FG, fontsize=12, fontweight="bold", y=1.01
    )
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[SAVED] {path}")


def save_loss_curve(history, path):
    fig, ax = plt.subplots(figsize=(7,4), facecolor=BG)
    ax.set_facecolor(BG)
    ax.plot(history, color="#4cc9f0", lw=2)
    ax.set_xlabel("Epoch", color=FG); ax.set_ylabel("Huber Loss", color=FG)
    ax.set_title("CrystalCNN Training Loss", color=FG)
    ax.tick_params(colors=FG)
    for sp in ax.spines.values(): sp.set_edgecolor("#444")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[SAVED] {path}")


# ══════════════════════════════════════════════════════════
# 9.  CSV 출력
# ══════════════════════════════════════════════════════════
# CSV 컬럼 그룹 정의
SIZE_KEYS   = ["diam_um","length_um","width_um","aspect_ratio","area_um2",
               "Rq","Ra","Rsk","Rku","Rv","Rp","Rt","Rc"]   # 크기 + 거칠기 지표
ORIENT_KEYS = ["azimuth_deg","polar_deg","FPO","MFOV","FAD","MRV","SA"]  # 방향 지표

def save_csv_crystals(records, cnn_preds, path):
    """
    결정별 측정값을 CSV로 저장한다.

    컬럼 구조:
        id          : 결정 순번 (1-based)
        cv_*        : 고전 CV로 측정한 값 (SIZE_KEYS + ORIENT_KEYS)
        cnn_*       : CNN이 예측한 값 (LABEL_KEYS 15개)

    encoding='utf-8-sig': Excel에서 한글/특수문자 깨짐 방지 (BOM 포함)
    """
    header = (["id"] +
              ["cv_"+k for k in SIZE_KEYS+ORIENT_KEYS] +   # CV 측정값 컬럼
              ["cnn_"+k for k in LABEL_KEYS])               # CNN 예측값 컬럼
    with open(path, "w", newline="", encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(header)
        for i, (rec, pred) in enumerate(zip(records, cnn_preds)):
            row = ([i+1] +
                   [rec.get(k, "") for k in SIZE_KEYS+ORIENT_KEYS] +
                   [round(pred.get(k, float("nan")), 4) for k in LABEL_KEYS])
            w.writerow(row)
    print(f"[SAVED] {path}")


def save_csv_global(g_metrics, fft_angle, radon_angle, n_crystals, um_per_px, path):
    """
    이미지 전체 수준의 전역 지표를 CSV로 저장한다.

    행 구조: metric | value | unit
        - n_crystals    : 검출된 결정 수
        - um_per_px     : 스케일 계수
        - Rq, Ra, ...   : 전역 거칠기/방위 지표
        - FFT/Radon 각도: 이미지 지배 방향
    """
    with open(path, "w", newline="", encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(["metric", "value", "unit"])
        w.writerow(["n_crystals", n_crystals, "—"])
        w.writerow(["um_per_px",  um_per_px,  "µm/px"])
        for k, v in g_metrics.items():
            # 지표 종류별 단위 자동 지정
            unit = "µm"  if k in ("Rq","Ra","Rv","Rp","Rt","Rc") else \
                   "°"   if k in ("FPO","MFOV","FAD") else \
                   "µm²" if k == "SA" else "—"
            w.writerow([k, round(v,4) if isinstance(v,float) else v, unit])
        w.writerow(["FFT_dominant_angle",   round(fft_angle,2),   "°"])
        w.writerow(["Radon_dominant_angle", round(radon_angle,2), "°"])
    print(f"[SAVED] {path}")


# ══════════════════════════════════════════════════════════
# 10.  PIPELINE  (공통 분석 함수)
# ══════════════════════════════════════════════════════════
def _run_pipeline(image_path, um_per_px, out_dir, n_epochs=50, load_model_path=None):
    """
    SEM 이미지 분석 공통 파이프라인 — 메뉴 옵션 2·3에서 호출한다.

    Parameters
    ----------
    image_path      : 분석할 PNG 파일 경로
    um_per_px       : 픽셀당 µm (= 1 / 스케일[px/µm])
    out_dir         : 결과 파일 저장 폴더 (없으면 자동 생성)
    n_epochs        : CNN 학습 에포크 수 (옵션 2에서만 사용, 기본 50)
    load_model_path : None이면 CNN을 새로 학습(옵션 2),
                      경로를 주면 해당 모델 로드 후 추론만 수행(옵션 3)

    처리 순서:
        Step 1: 전처리 — 크롭 + CLAHE + Gaussian
        Step 2: 세그먼테이션 — Frangi + Otsu + Watershed
        Step 3: 결정별 측정 — 형태 + 거칠기 + 방위각 지표
        Step 4: 전역 지표 — FFT / Radon 지배 방향각
        Step 5: CNN 학습(옵션2) 또는 저장 모델 추론(옵션3)
        Step 6: CNN 추론 (옵션2일 때)
        Step 7: 결과 저장 — PNG + 2개 CSV
    """
    os.makedirs(out_dir, exist_ok=True)
    # Image_Model 인스턴스 생성 (크롭/세그먼테이션 메서드 접근용)
    img_model = Image_Model(image_path, None, 150, 30, None, um_per_px, None, None)

    # ── 1. Pre-processing ────────────────────────────────
    print("\n=== Step 1: Pre-processing ===")
    gray, enh = img_model.crop_enhance_img(image_path, footer_px=150)
    print(f"  Image: {gray.shape}  |  Device: {DEVICE}")

    # ── 2. Segmentation ──────────────────────────────────
    print("\n=== Step 2: Segmentation ===")
    binary, labeled = img_model.segment_crystals(enh, min_area_px=30)
    print(f"  Watershed regions: {labeled.max()}")

    # ── 3. Per-crystal metrics ───────────────────────────
    print("\n=== Step 3: Per-crystal metrics ===")
    records = measure_crystals(gray, labeled, um_per_px)

    if len(records) < 2:
        print("[WARN] Very few crystals — trying relaxed segmentation.")
        _, bw = cv2.threshold(enh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = remove_small_objects((bw > 0), min_size=20).astype(np.uint8)
        labeled, _ = nd_label(binary)  # type: ignore
        records = measure_crystals(gray, labeled, um_per_px,
                                   max_aspect=18.0, max_len_um=60.0, min_area_px=15)

    # ── 4. Summary statistics ────────────────────────────
    if records:
        for metric, label in [
            ("length_um", "Length"), ("width_um", "Width"),
            ("aspect_ratio", "Aspect"), ("Rq", "Rq"), ("Ra", "Ra"),
            ("Rsk", "Rsk"), ("Rku", "Rku"), ("FPO", "FPO"), ("MRV", "MRV"),
        ]:
            vals = np.array([r[metric] for r in records
                             if not np.isnan(r.get(metric, float("nan")))])
            if vals.size:
                print(f"  {label:<12}: {vals.mean():.3f} ± {vals.std():.3f}"
                      f"  [{vals.min():.3f} – {vals.max():.3f}]")

    # ── 5. Global texture ────────────────────────────────
    print("\n=== Step 4: Global texture ===")
    g_metrics = global_metrics(gray, binary, um_per_px)
    fft_angle, _ = fft_dominant_angle(enh)
    radon_angle = radon_dominant_angle(enh)
    print(f"  FFT dominant   : {fft_angle:.1f}°")
    print(f"  Radon dominant : {radon_angle:.1f}°")
    print(f"  Global Rq      : {g_metrics['Rq']:.4f} µm")
    print(f"  Global FPO     : {g_metrics['FPO']:.2f}°   MRV: {g_metrics['MRV']:.4f}")
    print(f"  Global SA      : {g_metrics['SA']:.2f} µm²")

    # ── 6. CNN ───────────────────────────────────────────
    cnn_preds = [{}] * len(records)
    dataset = CrystalPatchDataset(gray, records)

    if load_model_path:
        # 옵션 3: 저장된 모델로 추론만 수행
        print("\n=== Step 5: CNN inference (saved model) ===")
        model = CrystalCNN()
        model.load_state_dict(torch.load(load_model_path, map_location=DEVICE,
                                         weights_only=True))
        model.to(DEVICE).eval()
        if len(dataset) >= 1:
            all_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
            cnn_preds = []
            with torch.no_grad():
                for xb, _ in all_loader:
                    preds = model(xb.to(DEVICE))
                    for p in preds:
                        cnn_preds.append(decode_pred(p.unsqueeze(0), dataset))

    elif len(dataset) >= 4:
        # 옵션 2: CNN 학습 후 추론
        print("\n=== Step 5: CrystalCNN training ===")
        loader = DataLoader(dataset, batch_size=min(16, len(dataset)),
                            shuffle=True, num_workers=0)
        model = CrystalCNN()
        print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")
        history = train_model(model, loader, n_epochs)
        save_loss_curve(history, os.path.join(out_dir, "loss_curve.png"))

        print("\n=== Step 6: CNN inference ===")
        model.eval()
        all_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
        cnn_preds = []
        with torch.no_grad():
            for xb, _ in all_loader:
                preds = model(xb.to(DEVICE))
                for p in preds:
                    cnn_preds.append(decode_pred(p.unsqueeze(0), dataset))

        print(f"\n  {'ID':>3}  {'CV_Rq':>8}  {'CNN_Rq':>8}"
              f"  {'CV_FPO':>8}  {'CNN_FPO':>9}  {'CV_len':>8}")
        for i, (rec, pred) in enumerate(zip(records[:5], cnn_preds[:5])):
            print(f"  {i+1:>3}  {rec.get('Rq', 0):>8.3f}  {pred.get('Rq', 0):>8.3f}"
                  f"  {rec.get('FPO', 0):>8.2f}  {pred.get('FPO', 0):>9.2f}"
                  f"  {rec['length_um']:>8.2f}")

        model_save = os.path.join(out_dir, "crystal_cnn.pth")
        torch.save(model.state_dict(), model_save)
        print(f"[SAVED] {model_save}")
    else:
        print("  Skipping CNN (too few valid crystals).")

    # ── 7. Save outputs ──────────────────────────────────
    print("\n=== Step 7: Saving outputs ===")
    save_figure(gray, records, binary, fft_angle, radon_angle, um_per_px,
                os.path.join(out_dir, "crystal_analysis.png"))
    save_csv_crystals(records, cnn_preds,
                      os.path.join(out_dir, "crystal_measurements.csv"))
    save_csv_global(g_metrics, fft_angle, radon_angle,
                    len(records), um_per_px,
                    os.path.join(out_dir, "global_metrics.csv"))

    print(f"\n[Done]  {len(records)} crystals  →  {out_dir}")

# ========================================================
# 11. Data Merge
# ========================================================
def merge_data():
    """
    Data/SEM/SEM_Results_*/crystal_measurements.csv 파일들을 통합하여
    결정형별 집계 피처를 계산하고 Data/SEM/Crystal_Measurements.csv로 저장한다.

    폴더명에서 결정형(Raw/Needle/Rod/Small_Square)을 추출하고,
    같은 결정형의 모든 이미지를 날짜 구분 없이 하나의 샘플로 집계한다.
    sample_id = NE_{결정형}  (총 4개)

    집계 방법:
        diam_um / length_um / width_um / area_um2 : D10, D50, mean, D90, Span
        aspect_ratio                               : median, mean
        azimuth_deg / polar_deg / FPO / FAD        : sin·cos 분해 후 평균
        MFOV / MRV / SA                            : mean, max, min
    """
    import re

    sem_dir  = _workspace_root / "Data" / "SEM"
    out_path = _workspace_root / "Data" / "Master"/ "Crystal_Measurements.csv"

    # 폴더명 패턴: SEM_Results_{날짜}_NE_{결정형}_SEM[...]
    folder_pat = re.compile(
        r"SEM_Results_\d+_NE_(Raw|Needle|Rod|Small_Square)_SEM"
    )

    # crystal_type → [DataFrame, ...]
    groups: dict = {}

    for folder in sorted(sem_dir.iterdir()):
        if not folder.is_dir():
            continue
        m = folder_pat.match(folder.name)
        if not m:
            continue
        csv_path = folder / "crystal_measurements.csv"
        if not csv_path.exists():
            print(f"  [SKIP] {folder.name} — crystal_measurements.csv 없음")
            continue

        crystal_type = m.group(1)
        df           = pd.read_csv(csv_path)
        groups.setdefault(crystal_type, []).append(df)
        print(f"  [LOAD] {folder.name}  ({len(df)} crystals)")

    if not groups:
        print("[ERROR] SEM_Results 폴더에서 데이터를 찾을 수 없습니다.")
        return

    rows = []
    for crystal_type, dfs in sorted(groups.items()):
        df_all    = pd.concat(dfs, ignore_index=True)
        sample_id = f"NE_{crystal_type}"
        row: dict = {"sample_id": sample_id}

        # ── diam_um / length_um / width_um / area_um2 ────────────
        for col in ["cv_diam_um", "cv_length_um", "cv_width_um", "cv_area_um2"]:
            if col not in df_all.columns:
                continue
            vals  = df_all[col].dropna().to_numpy(dtype=float)
            name  = col.replace("cv_", "")
            d10   = float(np.percentile(vals, 10))
            d50   = float(np.percentile(vals, 50))
            d90   = float(np.percentile(vals, 90))
            mean_ = float(vals.mean())
            span  = (d90 - d10) / d50 if d50 != 0 else np.nan
            row.update({
                f"{name}_d10":  round(d10,    4),
                f"{name}_d50":  round(d50,    4),
                f"{name}_mean": round(mean_,  4),
                f"{name}_d90":  round(d90,    4),
                f"{name}_span": round(float(span), 4) if not np.isnan(span) else np.nan,
            })

        # ── aspect_ratio ──────────────────────────────────────────
        if "cv_aspect_ratio" in df_all.columns:
            vals = df_all["cv_aspect_ratio"].dropna().to_numpy(dtype=float)
            row["aspect_ratio_median"] = round(float(np.median(vals)), 4)
            row["aspect_ratio_mean"]   = round(float(vals.mean()),     4)

        # ── 각도 컬럼: sin/cos 분해 후 평균·표준편차 ────────────
        for col in ["cv_azimuth_deg", "cv_polar_deg", "cv_FPO", "cv_FAD"]:
            if col not in df_all.columns:
                continue
            vals = df_all[col].dropna().to_numpy(dtype=float)
            name = col.replace("cv_", "")
            rad  = np.radians(vals)
            s, c = np.sin(rad), np.cos(rad)
            row[f"{name}_sin_mean"] = round(float(s.mean()), 4)
            row[f"{name}_sin_std"]  = round(float(s.std()),  4)
            row[f"{name}_cos_mean"] = round(float(c.mean()), 4)
            row[f"{name}_cos_std"]  = round(float(c.std()),  4)

        # ── MFOV / MRV / SA ───────────────────────────────────────
        for col in ["cv_MFOV", "cv_MRV", "cv_SA"]:
            if col not in df_all.columns:
                continue
            vals = df_all[col].dropna().to_numpy(dtype=float)
            name = col.replace("cv_", "")
            row[f"{name}_mean"] = round(float(vals.mean()), 4)
            row[f"{name}_max"]  = round(float(vals.max()),  4)
            row[f"{name}_min"]  = round(float(vals.min()),  4)

        rows.append(row)
        print(f"  [AGG] {sample_id}  ({len(df_all)} crystals, {len(dfs)} 이미지)")

    result_df = pd.DataFrame(rows)
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[SAVED] {out_path}  ({len(result_df)} rows × {len(result_df.columns)} cols)")

# ========================================================
# CUDA 가용성 확인 유틸리티
# ========================================================
def check_cuda():
    """
    PyTorch를 통해 CUDA(GPU) 가용성을 확인한다.
    학습 전에 GPU 사용 가능 여부를 사용자에게 알려주는 진단 함수.
    """
    return ("CUDA Available!."
            if torch.cuda.is_available()
            else "CUDA Not Available. Run by CPU.")

# ========================================================
# 실제 실행 부분 — 프로그램 진입점
# ========================================================

if __name__ == "__main__":
    while True:
        print("\n!Choose what to do!")
        print("=" * 60)
        print("1. Check CUDA (GPU) Acceleration Available")
        print("2. Import & Train Model")
        print("3. Analyze Image from Trained Model")
        print("4. Merge Data")
        print("5. Exit Program")
        print("=" * 60)
        _work = input("Input Number: ").strip()

        # ── 1. CUDA 확인 ──────────────────────────────────
        if _work == '1':
            print(check_cuda())

        # ── 2. 이미지 학습 ────────────────────────────────
        elif _work == '2':
            # tkinter 루트 창을 만들고 숨긴 뒤 파일 다이얼로그를 호출한다.
            # attributes('-topmost', True) : 다이얼로그가 VSCode/터미널 뒤에
            # 숨지 않도록 항상 맨 앞에 표시되게 설정
            _root = tk.Tk()
            _root.withdraw()                      # 빈 루트 창 숨기기
            _root.attributes('-topmost', True)    # 파일 선택창을 맨 앞으로
            _root.update()                        # 이벤트 루프 한 번 갱신
            _image_path = filedialog.askopenfilename(
                parent=_root,
                title="SEM 이미지 파일 선택 (학습용)",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
            )
            _root.destroy()

            if not _image_path:
                print("[WARN] 파일을 선택하지 않았습니다.")
                continue
                
            # 파일명 추출 (확장자 제외) 및 폴더명 동적 생성
            image_name = os.path.splitext(os.path.basename(_image_path))[0]
            _DEFAULT_OUT = str(_workspace_root / "Data" / "SEM" / f"SEM_Results_{image_name}")

            try:
                _s = input("Scale (pixels/µm) 입력 [기본값(Scale bar = 20µm): 8.1]: ").strip()
                _px_per_um = float(_s) if _s else 8.1
                _e = input("학습 Epoch 수 입력 [기본값: 100]: ").strip()
                _epochs = int(_e) if _e else 100
            except ValueError:
                print("[WARN] 잘못된 입력 — 기본값 사용 (px/µm=8.1, epoch=50)")
                _px_per_um, _epochs = 8.1, 100

            _run_pipeline(_image_path, 1.0 / _px_per_um, _DEFAULT_OUT,
                          n_epochs=_epochs)

        # ── 3. 저장된 모델로 분석 ─────────────────────────
        elif _work == '3':
            _root = tk.Tk()
            _root.withdraw()
            _root.attributes('-topmost', True)
            _root.update()
            _image_path = filedialog.askopenfilename(
                parent=_root,
                title="SEM 이미지 파일 선택 (분석용)",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
            )
            _root.destroy()

            if not _image_path:
                print("[WARN] 파일을 선택하지 않았습니다.")
                continue

            # 파일명 추출 및 폴더명/모델 경로 동적 생성
            image_name = os.path.splitext(os.path.basename(_image_path))[0]
            _DEFAULT_OUT = str(_workspace_root / "Data" / "SEM" / f"SEM_Results_{image_name}")
            _MODEL_PATH  = os.path.join(_DEFAULT_OUT, "crystal_cnn.pth")

            if not os.path.exists(_MODEL_PATH):
                print(f"[ERROR] 해당 이미지에 대해 학습된 모델을 찾을 수 없습니다: {_MODEL_PATH}")
                print("  옵션 2를 먼저 실행하여 모델을 학습시켜 주세요.")
                continue

            try:
                _s = input("Scale (pixels/µm) 입력 [기본값: 8.1]: ").strip()
                _px_per_um = float(_s) if _s else 8.1
            except ValueError:
                print("[WARN] 잘못된 입력 — 기본값 사용 (px/µm=8.1)")
                _px_per_um = 8.1

            _run_pipeline(_image_path, 1.0 / _px_per_um, _DEFAULT_OUT,
                          load_model_path=_MODEL_PATH)

        # ── 4. 데이터 통합 ────────────────────────────────
        elif _work == '4':
            print("\n=== Data Merge: SEM 결정 측정값 통합 ===")
            merge_data()

        # ── 5. 종료 ───────────────────────────────────────
        elif _work == '5':
            print("프로그램을 종료합니다.")
            sys.exit(0)

        else:
            print("잘못된 입력입니다. 1~5 중에서 선택해 주세요.")
