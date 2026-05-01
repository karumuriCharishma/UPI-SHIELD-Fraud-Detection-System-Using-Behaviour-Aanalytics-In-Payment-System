import io
from urllib.parse import parse_qs, urlparse

from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import pandas as pd
import numpy as np
import streamlit as st

from app_core import analyze_transaction
from database import init_db, get_recent_transactions, get_stats

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None

try:
    from pyzbar.pyzbar import decode as pyzbar_decode
except Exception:
    pyzbar_decode = None


st.set_page_config(
    page_title="UPI Shield",
    page_icon="U",
    layout="wide",
)

init_db()


def verdict_style(verdict: str) -> str:
    mapping = {
        "SAFE": "ok",
        "SUSPICIOUS": "warn",
        "FRAUD": "risk",
    }
    return mapping.get(verdict, "warn")


def render_metric_card(label: str, value: str, help_text: str = ""):
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=5)
def load_stats() -> dict:
    return get_stats()


@st.cache_data(ttl=5)
def load_history(limit: int = 20) -> list[dict]:
    rows = get_recent_transactions(limit=limit)
    return [
        {
            "id": r[0],
            "upi_id": r[1],
            "amount": r[2],
            "timestamp": r[3],
            "risk_score": r[4],
            "verdict": r[5],
        }
        for r in rows
    ]


def normalize_history_rows(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    rename_map = {
        "id": "ID",
        "upi_id": "UPI ID",
        "amount": "Amount",
        "timestamp": "Timestamp",
        "risk_score": "Risk Score",
        "verdict": "Verdict",
    }
    df = df.rename(columns=rename_map)

    ordered_columns = ["ID", "UPI ID", "Amount", "Timestamp", "Risk Score", "Verdict"]
    df = df.reindex(columns=ordered_columns)
    if "Risk Score" in df.columns:
        df["Risk Score"] = pd.to_numeric(df["Risk Score"], errors="coerce").round(1)
    return df


def _center_crop_square(image: Image.Image, ratio: float = 0.8) -> Image.Image:
    width, height = image.size
    side = int(min(width, height) * ratio)
    if side <= 0:
        return image
    left = max((width - side) // 2, 0)
    top = max((height - side) // 2, 0)
    return image.crop((left, top, left + side, top + side))


def _grid_crop_squares(image: Image.Image, ratio: float = 0.65) -> list[Image.Image]:
    """
    Generate square crops from a 3x3 grid.

    This helps for app screenshots where the QR is not centered.
    """
    width, height = image.size
    side = int(min(width, height) * ratio)
    if side <= 0:
        return [image]

    xs = [0, max((width - side) // 2, 0), max(width - side, 0)]
    ys = [0, max((height - side) // 2, 0), max(height - side, 0)]

    crops: list[Image.Image] = []
    for top in ys:
        for left in xs:
            crop = image.crop((left, top, left + side, top + side))
            if crop.size[0] > 0 and crop.size[1] > 0:
                crops.append(crop)
    return crops


def _custom_crop(
    image: Image.Image,
    left_pct: float,
    top_pct: float,
    right_pct: float,
    bottom_pct: float,
) -> Image.Image:
    width, height = image.size
    left = int(width * left_pct)
    top = int(height * top_pct)
    right = int(width * right_pct)
    bottom = int(height * bottom_pct)

    left = max(min(left, width - 1), 0)
    top = max(min(top, height - 1), 0)
    right = max(min(right, width), left + 1)
    bottom = max(min(bottom, height), top + 1)
    return image.crop((left, top, right, bottom))


def _autocrop_square(image: Image.Image) -> Image.Image:
    """
    Try to find the densest dark region, which often corresponds to the QR area
    in screenshots that include app chrome around the code.
    """
    gray = image.convert("L")
    arr = np.array(gray)
    if arr.size == 0:
        return image

    threshold = np.percentile(arr, 45)
    mask = arr < threshold
    ys, xs = np.where(mask)
    if len(xs) < 50 or len(ys) < 50:
        return image

    pad_x = max(int(image.width * 0.08), 10)
    pad_y = max(int(image.height * 0.08), 10)
    left = max(int(xs.min()) - pad_x, 0)
    right = min(int(xs.max()) + pad_x, image.width - 1)
    top = max(int(ys.min()) - pad_y, 0)
    bottom = min(int(ys.max()) + pad_y, image.height - 1)

    if right - left < 40 or bottom - top < 40:
        return image

    cropped = image.crop((left, top, right + 1, bottom + 1))
    side = min(cropped.width, cropped.height)
    left = max((cropped.width - side) // 2, 0)
    top = max((cropped.height - side) // 2, 0)
    return cropped.crop((left, top, left + side, top + side))


def _prepare_pil_variants(image: Image.Image, screenshot_mode: bool = False) -> list[Image.Image]:
    variants: list[Image.Image] = []

    base = image.convert("RGB")
    variants.append(base)
    variants.append(base.convert("L"))
    variants.append(ImageOps.autocontrast(base.convert("L")))
    variants.append(ImageOps.invert(base.convert("L")))
    variants.append(ImageOps.autocontrast(ImageOps.invert(base.convert("L"))))
    variants.append(ImageEnhance.Sharpness(base).enhance(2.0))
    variants.append(ImageEnhance.Contrast(base).enhance(2.0))
    variants.append(base.filter(ImageFilter.SHARPEN))

    crop_ratios = (0.98, 0.92, 0.85, 0.75) if screenshot_mode else (0.95, 0.85, 0.75)
    for ratio in crop_ratios:
        cropped = _center_crop_square(base, ratio=ratio)
        variants.append(cropped)
        variants.append(cropped.convert("L"))
        variants.append(ImageOps.autocontrast(cropped.convert("L")))
        variants.append(ImageOps.invert(cropped.convert("L")))
        variants.append(ImageOps.autocontrast(ImageOps.invert(cropped.convert("L"))))
        variants.append(ImageEnhance.Sharpness(cropped).enhance(2.0))

    if screenshot_mode:
        for crop in _grid_crop_squares(base, ratio=0.75):
            variants.append(crop)
            variants.append(crop.convert("L"))
            variants.append(ImageOps.autocontrast(crop.convert("L")))
            variants.append(ImageOps.invert(crop.convert("L")))
            variants.append(ImageOps.autocontrast(ImageOps.invert(crop.convert("L"))))
            variants.append(ImageEnhance.Contrast(crop).enhance(2.0))

        auto = _autocrop_square(base)
        variants.append(auto)
        variants.append(auto.convert("L"))
        variants.append(ImageOps.autocontrast(auto.convert("L")))
        variants.append(ImageOps.invert(auto.convert("L")))
        variants.append(ImageOps.autocontrast(ImageOps.invert(auto.convert("L"))))
        variants.append(ImageEnhance.Contrast(auto).enhance(2.0))

    return variants


def _get_cv2_detector():
    """
    Return the best available OpenCV QR detector.
    Tries both API styles and catches ALL exceptions (not just AttributeError).
    """
    # Style 2: dot-submodule (some older contrib builds)
    try:
        det = cv2.wechat_qrcode.WeChatQRCode()
        return det, "wechat"
    except Exception:
        pass

    # Fallback: basic detector (weak, fails on logo QRs like PhonePe)
    return cv2.QRCodeDetector(), "basic"


def _wechat_try(detector, img_bgr: np.ndarray) -> str | None:
    """
    Try WeChatQRCode on a single BGR image.
    WeChatQRCode MUST receive a color BGR image — never grayscale.
    It does its own internal grayscale conversion + detection.
    """
    try:
        if img_bgr is None or img_bgr.size == 0:
            return None
        # Ensure 3-channel BGR — WeChatQRCode rejects 1-channel images
        if len(img_bgr.shape) == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        texts, _ = detector.detectAndDecode(img_bgr)
        # texts can be a tuple of strings or a plain string depending on build
        if isinstance(texts, str):
            texts = [texts]
        for t in (texts or []):
            if t and str(t).strip():
                return str(t).strip()
    except Exception:
        pass
    return None


def _basic_try(detector, img: np.ndarray) -> str | None:
    """Try basic cv2.QRCodeDetector on one image."""
    try:
        data, _, _ = detector.detectAndDecode(img)
        if data and data.strip():
            return data.strip()
    except Exception:
        pass
    return None


def _pyzbar_decode_variants(image_bytes: bytes, screenshot_mode: bool = False) -> str | None:
    """
    Fallback decoder using pyzbar.
    This is much weaker than the OpenCV/WeChat path, but it can still help for
    some clean QR images if OpenCV fails.
    """
    if pyzbar_decode is None:
        return None

    try:
        base = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return None

    variants: list[Image.Image] = [
        base,
        base.convert("L"),
        ImageOps.autocontrast(base.convert("L")),
        ImageOps.invert(base.convert("L")),
        ImageOps.autocontrast(ImageOps.invert(base.convert("L"))),
    ]

    if screenshot_mode:
        variants.extend(_prepare_pil_variants(base, screenshot_mode=True))

    variants.extend(
        [
            base.resize((base.width * 2, base.height * 2)),
            base.resize((base.width * 3, base.height * 3)),
        ]
    )

    for variant in variants:
        try:
            decoded_items = pyzbar_decode(variant)
        except Exception:
            continue

        for item in decoded_items:
            payload = item.data.decode("utf-8", errors="ignore").strip()
            if payload:
                return payload

    return None


def decode_qr_text(image_bytes: bytes, screenshot_mode: bool = False) -> str | None:
    """
    Decode a QR code from raw image bytes.

    Strategy (WeChatQRCode path — the common case):
      1. Try the raw full image directly — works for clean standalone QR images.
      2. Auto-detect portrait screenshots (PhonePe/GPay are tall) and try
         vertical-band crops so the QR region fills the frame.
      3. Try center-square crops at shrinking ratios.
      4. Try fixed resize targets (helps when image is huge or tiny).

    WeChatQRCode does NOT need grayscale/threshold preprocessing —
    it has its own internal pipeline. Feeding it grayscale HURTS accuracy.
    """
    if cv2 is None:
        return _pyzbar_decode_variants(image_bytes, screenshot_mode=screenshot_mode)

    # Primary: load via PIL then convert to BGR numpy array.
    # This is more robust for web-uploaded images (handles WEBP, RGBA PNGs, etc.)
    # and mirrors what cv2.imread() does when reading from disk.
    img = None
    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        img = None

    # Fallback: cv2.imdecode from raw bytes
    if img is None:
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if img is None:
        return _pyzbar_decode_variants(image_bytes, screenshot_mode=screenshot_mode)

    # Ensure the array is writable (some OpenCV ops require it)
    img = img.copy()

    detector, mode = _get_cv2_detector()
    h, w = img.shape[:2]

    if mode == "wechat":
        # ── Step 1: raw full image ────────────────────────────────
        result = _wechat_try(detector, img)
        if result:
            return result

        # ── Step 2: portrait screenshot handling ──────────────────
        # PhonePe/GPay screenshots are tall (aspect > 1.3).
        # The QR is usually in the bottom 60% of the screen.
        # We slice the image into horizontal bands and try each.
        aspect = h / max(w, 1)
        if aspect > 1.2 or screenshot_mode:
            bands = [
                (0.0,  0.5),   # top half
                (0.2,  0.7),   # upper-middle
                (0.35, 0.85),  # lower-middle  ← PhonePe QR usually here
                (0.5,  1.0),   # bottom half
                (0.1,  0.9),   # almost full
            ]
            for y0_pct, y1_pct in bands:
                y0, y1 = int(h * y0_pct), int(h * y1_pct)
                band = img[y0:y1, :]
                bh, bw = band.shape[:2]
                # Center-square crop within the band
                side = min(bh, bw)
                cx = (bw - side) // 2
                square = band[:side, cx:cx + side]
                if square.size == 0:
                    continue
                result = _wechat_try(detector, square)
                if result:
                    return result
                # Also try 2× upscale — helps when QR is small in the band
                if side < 600:
                    up = cv2.resize(square, None, fx=2, fy=2,
                                    interpolation=cv2.INTER_CUBIC)
                    result = _wechat_try(detector, up)
                    if result:
                        return result

        # ── Step 3: shrinking center-square crops ─────────────────
        for ratio in (0.98, 0.90, 0.80, 0.70, 0.60, 0.50):
            side = int(min(w, h) * ratio)
            if side < 80:
                continue
            x0 = (w - side) // 2
            y0 = (h - side) // 2
            crop = img[y0:y0 + side, x0:x0 + side]
            result = _wechat_try(detector, crop)
            if result:
                return result
            if side < 600:
                up = cv2.resize(crop, None, fx=2, fy=2,
                                interpolation=cv2.INTER_CUBIC)
                result = _wechat_try(detector, up)
                if result:
                    return result

        # ── Step 4: fixed-size rescale targets ────────────────────
        # Helps when the image is extremely large or very tiny
        for target in (800, 600, 400, 1200):
            resized = cv2.resize(img, (target, target),
                                 interpolation=cv2.INTER_CUBIC)
            result = _wechat_try(detector, resized)
            if result:
                return result

    else:
        # Basic detector — try raw + a few simple variants
        for attempt in [
            img,
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
            cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),
        ]:
            result = _basic_try(detector, attempt)
            if result:
                return result

    # ── Final fallback: pyzbar ────────────────────────────────────
    return _pyzbar_decode_variants(image_bytes, screenshot_mode=screenshot_mode)




def parse_upi_payload(payload: str) -> dict[str, str]:
    raw = payload.strip()
    result = {"upi_id": "", "payee_name": ""}

    if not raw:
        return result

    if raw.lower().startswith("upi://"):
        parsed = urlparse(raw)
        params = parse_qs(parsed.query)
        result["upi_id"] = params.get("pa", [""])[0].strip()
        result["payee_name"] = params.get("pn", [""])[0].strip()
        return result

    if "@" in raw:
        result["upi_id"] = raw
        return result

    return result


st.markdown(
    """
    <style>
      .stApp {
        background:
          radial-gradient(circle at top left, rgba(16,185,129,0.14), transparent 28%),
          radial-gradient(circle at top right, rgba(59,130,246,0.10), transparent 24%),
          linear-gradient(180deg, #08111f 0%, #0b1728 45%, #0f1d32 100%);
        color: #eef2ff;
      }
      .stApp,
      .stApp label,
      .stApp p,
      .stApp span,
      .stApp li,
      .stApp td,
      .stApp th,
      .stApp small,
      .stApp div[data-testid="stMarkdownContainer"],
      .stApp div[data-testid="stCaptionContainer"] {
        color: #eef2ff;
      }
      .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
      }
      h1, h2, h3, h4, p, label, div, span, li, td, th, small {
        font-family: "Segoe UI", "Inter", sans-serif;
      }
      .stApp [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
      }
      .stApp [data-testid="stMetricValue"] {
        color: #f8fafc !important;
      }
      .stApp [data-testid="stMetricDelta"] {
        color: #cbd5e1 !important;
      }
      .stApp input,
      .stApp textarea,
      .stApp [data-baseweb="input"] input,
      .stApp [data-baseweb="textarea"] textarea,
      .stApp [data-baseweb="select"] div,
      .stApp [data-baseweb="select"] input {
        color: #f8fafc !important;
        -webkit-text-fill-color: #f8fafc !important;
      }
      .stApp [data-baseweb="input"],
      .stApp [data-baseweb="textarea"],
      .stApp [data-baseweb="select"] {
        background: rgba(15, 23, 42, 0.9) !important;
      }
      .stApp [data-testid="stWidgetLabel"] p {
        color: #e2e8f0 !important;
      }
      .stApp [data-testid="stFileUploaderDropzone"] {
        background: rgba(15, 23, 42, 0.72);
        border-color: rgba(148,163,184,0.18);
      }
      .stApp [data-testid="stFileUploaderDropzone"] * {
        color: #f8fafc !important;
      }
      .stApp [data-testid="stDataFrame"] * {
        color: #0f172a !important;
      }
      .stApp [data-testid="stTable"] * {
        color: #0f172a !important;
      }
      .stApp [data-testid="stAlert"] {
        color: #f8fafc !important;
      }
      .stApp [data-testid="stAlert"] * {
        color: inherit !important;
      }
      .hero {
        padding: 1.25rem 1.4rem;
        border: 1px solid rgba(148,163,184,0.18);
        border-radius: 20px;
        background: rgba(6, 10, 19, 0.65);
        box-shadow: 0 20px 60px rgba(0,0,0,0.25);
        margin-bottom: 1rem;
      }
      .hero h1 {
        margin: 0;
        font-size: 2.2rem;
        line-height: 1.1;
      }
      .hero p {
        margin: 0.35rem 0 0 0;
        color: #cbd5e1;
      }
      .metric-card {
        border-radius: 18px;
        padding: 1rem 1.1rem;
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(148,163,184,0.14);
      }
      .metric-label {
        font-size: 0.82rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }
      .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 0.25rem;
      }
      .metric-help {
        font-size: 0.9rem;
        color: #cbd5e1;
        margin-top: 0.15rem;
      }
      .result-box {
        border-radius: 20px;
        padding: 1rem 1.1rem;
        border: 1px solid rgba(148,163,184,0.18);
        background: rgba(6, 10, 19, 0.74);
      }
      .result-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        font-weight: 700;
        letter-spacing: 0.04em;
      }
      .result-badge.SAFE { background: rgba(34,197,94,0.16); color: #22c55e; }
      .result-badge.SUSPICIOUS { background: rgba(245,158,11,0.16); color: #f59e0b; }
      .result-badge.FRAUD { background: rgba(239,68,68,0.16); color: #ef4444; }
      .tip-item {
        padding: 0.55rem 0.7rem;
        border-radius: 12px;
        background: rgba(15,23,42,0.72);
        border: 1px solid rgba(148,163,184,0.12);
        margin-bottom: 0.45rem;
        color: #f8fafc;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>UPI Shield</h1>
      <p>Streamlit fraud-risk checker for UPI transactions with manual entry or QR code scan.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

stats = {"total": 0, "fraud": 0, "suspicious": 0, "avg_risk": 0}
try:
    stats = load_stats()
except Exception as exc:
    st.error(f"Could not load dashboard stats from the local database: {exc}")

col1, col2, col3, col4 = st.columns(4)
with col1:
    render_metric_card("Total Scans", str(stats["total"]), "Transactions analyzed so far")
with col2:
    render_metric_card("Fraud", str(stats["fraud"]), "Flagged as high risk")
with col3:
    render_metric_card("Suspicious", str(stats["suspicious"]), "Needs manual review")
with col4:
    render_metric_card("Avg Risk", f'{stats["avg_risk"]:.1f}/100', "Mean score across scans")

st.write("")

last_result = st.session_state.get("last_result")

top_left, top_right = st.columns([1.05, 0.95], gap="large")

with top_left:
    st.subheader("Analyze Transaction")
    input_mode = st.radio(
        "Input method",
        ["Type UPI ID", "Scan QR code"],
        horizontal=True,
        label_visibility="collapsed",
    )

    qr_details = {"upi_id": "", "payee_name": ""}

    if input_mode == "Scan QR code":
        qr_file = st.file_uploader(
            "Upload QR code image",
            type=["png", "jpg", "jpeg", "webp"],
            help="Upload a UPI QR image. We will try to extract the payment address from it.",
        )
        if qr_file is not None:
            st.caption(f"Accepted file type: {qr_file.type or 'unknown'}")
            qr_bytes = qr_file.getvalue()
            st.success("Image loaded successfully.")
            st.image(
                qr_bytes,
                caption="Uploaded QR code preview",
                use_container_width=True,
            )

            # Show diagnostics so the user knows which decoder is active
            with st.expander("🔍 Decoder diagnostics", expanded=False):
                if cv2 is not None:
                    st.write(f"**OpenCV version:** {cv2.__version__}")
                    det_name = "WeChatQRCode (contrib)" if hasattr(cv2, "wechat_qrcode") else "QRCodeDetector (basic)"
                    st.write(f"**Detector:** {det_name}")
                else:
                    st.write("**OpenCV:** not available — using pyzbar fallback")
                st.write(f"**pyzbar:** {'available' if pyzbar_decode is not None else 'not available'}")
                # Verify PIL can load the image
                try:
                    _pil_check = Image.open(io.BytesIO(qr_bytes))
                    st.write(f"**PIL image:** {_pil_check.size[0]}×{_pil_check.size[1]} px, mode={_pil_check.mode}, format={_pil_check.format}")
                except Exception as _e:
                    st.write(f"**PIL load failed:** {_e}")

            decoded_text = decode_qr_text(qr_bytes, screenshot_mode=False)
            if not decoded_text:
                # Screenshots often need the more aggressive crop-and-resize path.
                decoded_text = decode_qr_text(qr_bytes, screenshot_mode=True)
            if decoded_text:
                qr_details = parse_upi_payload(decoded_text)
                st.success("QR code decoded successfully.")
                with st.expander("Decoded QR payload", expanded=False):
                    st.code(decoded_text)
                if qr_details["upi_id"]:
                    st.success(f"Detected UPI ID: {qr_details['upi_id']}")
                    if qr_details["payee_name"]:
                        st.info(f"Payee name: {qr_details['payee_name']}")
                else:
                    st.error("QR code was detected, but it did not contain a UPI address.")
            else:
                st.warning(
                    "Could not decode the QR from the uploaded image. "
                    "Please upload a clearer QR image or enter the UPI ID directly."
                )
                st.info("Image loaded successfully, but QR decoding failed.")
    with st.form("analyze_form", clear_on_submit=False):
        upi_id_placeholder = "name@okhdfcbank"
        if input_mode == "Scan QR code" and qr_details["upi_id"]:
            upi_id_placeholder = qr_details["upi_id"]
        upi_id = st.text_input("UPI ID", placeholder=upi_id_placeholder, value=qr_details["upi_id"])
        payee_name = st.text_input("Payee Name", value=qr_details["payee_name"], placeholder="Optional")
        amount = st.number_input("Amount (INR)", min_value=1.0, step=100.0, value=500.0)
        submitted = st.form_submit_button("Analyze Risk")

    if submitted:
        try:
            if not upi_id.strip():
                st.error("Please type a UPI ID or upload a QR code with a payment address.")
            else:
                result = analyze_transaction(
                    upi_id=upi_id,
                    amount=float(amount),
                    payee_name=payee_name,
                )
                st.session_state["last_result"] = {
                    "upi_id": upi_id,
                    "amount": float(amount),
                    "payee_name": payee_name,
                    **result,
                }
                load_stats.clear()
                load_history.clear()
                st.rerun()
        except Exception as exc:
            st.error(str(exc))

with top_right:
    st.subheader("Risk Score")
    if last_result:
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        badge_class = verdict_style(last_result["verdict"])
        st.markdown(
            f"""
            <div class="result-badge {badge_class}">{last_result["verdict"]}</div>
            <h3 style="margin:0.65rem 0 0.15rem 0;">Risk Score: {last_result["risk_score"]:.1f}/100</h3>
            <p style="margin:0;color:#cbd5e1;">UPI ID: {last_result["upi_id"]} | Amount: INR {last_result["amount"]:,.2f}</p>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(last_result["risk_score"] / 100.0, 1.0))
        st.markdown("**Tips**")
        for tip in last_result["tips"]:
            st.markdown(f'<div class="tip-item">{tip}</div>', unsafe_allow_html=True)
        st.markdown("**Explanation**")
        st.json(last_result["explanation"])
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Run an analysis to see the risk score here.")

st.write("")

bottom_left, bottom_right = st.columns([1.1, 0.9], gap="large")

with bottom_left:
    st.subheader("Recent Transactions")
    rows = []
try:
    rows = load_history()
except Exception as exc:
    st.error(f"Could not load recent transactions from the local database: {exc}")
if rows:
    history_df = normalize_history_rows(rows)
    st.dataframe(history_df, use_container_width=True, hide_index=True)
else:
    st.info("No transactions yet. Run your first scan.")

with bottom_right:
    st.subheader("Quick Signals")
    if last_result:
        signals = pd.DataFrame(
            [
                ("Handle Known", "Yes" if last_result["explanation"]["handle_known"] else "No"),
                ("Odd Hour", "Yes" if last_result["explanation"]["odd_hour"] else "No"),
                ("High Velocity", "Yes" if last_result["explanation"]["high_velocity"] else "No"),
                ("Large Amount", "Yes" if last_result["explanation"]["amount_flag"] else "No"),
                ("Velocity Count", str(last_result["explanation"]["velocity_count"])),
            ],
            columns=["Signal", "Value"],
        )
        st.dataframe(signals, use_container_width=True, hide_index=True)
    else:
        st.caption("Analyze a transaction to see feature-level signals.")
