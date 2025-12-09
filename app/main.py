from dotenv import load_dotenv
import os
import cv2
import numpy as np
import requests
from flask import Flask, request, jsonify, abort
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from collections import Counter
import traceback
import tempfile
from typing import Dict, List, Tuple
from functools import wraps
import logging
from datetime import datetime, timedelta
import re
import time

# ----------------------------------
# إعدادات التسجيل
# ----------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ----------------------------------
# إعدادات المشروع
# ----------------------------------
load_dotenv()  # يحمّل كل القيم من ملف .env

HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
GMAIL_SENDER = os.getenv("GMAIL_SENDER", "").strip()
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "").strip()

# التحقق من المتغيرات البيئية
REQUIRED_ENV_VARS = ["HF_TOKEN", "GMAIL_SENDER", "GMAIL_APP_PASSWORD"]
missing_vars = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
if missing_vars:
    logger.warning(f"⚠️ متغيرات بيئية مفقودة: {missing_vars}")

# ----------------------------------
# دوال مساعدة - الآن تقبل بايتات (bytes) أو مصفوفة (np.ndarray)
# ----------------------------------


def validate_image_file_content(file_bytes: bytes, filename: str) -> Tuple[bool, str]:
    """التحقق من صحة محتوى ملف الصورة في الذاكرة"""
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    if not file_bytes:
        return False, "لم يتم توفير محتوى الملف"

    filename = (filename or '').lower().strip()
    if not filename:
        return False, "اسم الملف فارغ"

    _, ext = os.path.splitext(filename)

    if not ext or ext not in ALLOWED_EXTENSIONS:
        allowed = ', '.join(ALLOWED_EXTENSIONS)
        return False, f"نوع الملف غير مدعوم. المسموح: {allowed}"

    file_size = len(file_bytes)
    if file_size == 0:
        return False, "الملف فارغ"
    if file_size > 10 * 1024 * 1024:
        size_mb = file_size / (1024 * 1024)
        return False, f"حجم الملف كبير جداً ({size_mb:.1f}MB). الحد الأقصى: 10MB"

    return True, ""


def load_and_preprocess_image(image_bytes: bytes) -> np.ndarray:
    """تحميل البايتات ومعالجة مسبقة للصورة في الذاكرة"""
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("فشل في قراءة الصورة من الذاكرة")

    # تصغير الصورة إذا كانت كبيرة جداً للحفاظ على الأداء
    height, width = img.shape[:2]
    max_dimension = 1200

    if height > max_dimension or width > max_dimension:
        scale = max_dimension / max(height, width)
        new_size = (int(width * scale), int(height * scale))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

    return img


def rgb_to_hex(r, g, b):
    """تحويل RGB إلى Hex"""
    return f"#{r:02x}{g:02x}{b:02x}".upper()


def get_color_name(rgb_values):
    """الحصول على اسم تقريبي للون"""
    r, g, b = rgb_values

    # قاعدة بيانات ألوان مبسطة
    colors_db = [
        ((255, 0, 0), "أحمر"),
        ((0, 255, 0), "أخضر"),
        ((0, 0, 255), "أزرق"),
        ((255, 255, 0), "أصفر"),
        ((255, 0, 255), "أرجواني"),
        ((0, 255, 255), "سماوي"),
        ((255, 255, 255), "أبيض"),
        ((0, 0, 0), "أسود"),
        ((128, 128, 128), "رمادي"),
        ((255, 165, 0), "برتقالي"),
        ((128, 0, 128), "بنفسجي"),
        ((165, 42, 42), "بني"),
        ((255, 192, 203), "وردي"),
    ]

    # البحث عن أقرب لون
    min_distance = float('inf')
    closest_color = "غير معروف"

    for color_rgb, color_name in colors_db:
        # حساب المسافة الإقليدية (Euclidean Distance)
        distance = np.sqrt((r - color_rgb[0])**2 +
                           (g - color_rgb[1])**2 +
                           (b - color_rgb[2])**2)
        if distance < min_distance:
            min_distance = distance
            closest_color = color_name

    return closest_color


def analyze_colors(img_bytes: bytes) -> List[Dict]:
    """تحليل الألوان الأساسية في الرسم بدون sklearn"""
    try:
        img = load_and_preprocess_image(img_bytes)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # تقليل دقة الصورة لتسريع التجميع
        img_small = cv2.resize(img_rgb, (100, 100),
                               interpolation=cv2.INTER_AREA)
        pixels = img_small.reshape(-1, 3)

        # تجميع الألوان يدوياً
        color_bins = {}
        bin_size = 32  # تجميع كل 32 قيمة لون في bin واحد

        for pixel in pixels:
            r, g, b = pixel
            # تقريب الألوان
            r_bin = (r // bin_size) * bin_size
            g_bin = (g // bin_size) * bin_size
            b_bin = (b // bin_size) * bin_size

            key = (r_bin, g_bin, b_bin)
            color_bins[key] = color_bins.get(key, 0) + 1

        # الحصول على أكثر 3 ألوان انتشاراً
        sorted_colors = sorted(
            color_bins.items(), key=lambda x: x[1], reverse=True)[:3]

        total_pixels = len(pixels)
        colors_info = []
        for (r, g, b), count in sorted_colors:
            color_hex = rgb_to_hex(r, g, b)
            color_name = get_color_name((r, g, b))
            percentage = (count / total_pixels) * 100

            colors_info.append({
                "rgb": f"({r}, {g}, {b})",
                "hex": color_hex,
                "name": color_name,
                "percentage": round(percentage, 1)
            })

        return colors_info

    except Exception as e:
        logger.error(f"خطأ في تحليل الألوان: {e}")
        return [{"rgb": "(خطأ)", "hex": "#000000", "name": "غير متاح", "percentage": 0}]


def analyze_emotion_from_colors(colors):
    """تحليل المشاعر من الألوان"""
    emotion_map = {
        'أحمر': 'طاقة، حماس، عاطفة',
        'أخضر': 'توازن، نمو، أمان',
        'أزرق': 'هدوء، استقرار، ثقة',
        'أصفر': 'سعادة، تفاؤل، إبداع',
        'برتقالي': 'دفء، نشاط، حماس',
        'أرجواني': 'إبداع، غموض، روحية',
        'وردي': 'حب، رقة، عطف',
        'بني': 'استقرار، أمان، واقعية',
        'أسود': 'قوة، جدية، غموض',
        'أبيض': 'نقاء، بساطة، سلام',
        'رمادي': 'حيادية، توازن، رسمية'
    }

    emotions = []
    for color_info in colors[:3]:  # أول 3 ألوان
        color_name = color_info.get("name", "")
        if color_name in emotion_map:
            emotions.append(f"{color_name}: {emotion_map[color_name]}")

    return emotions[:3]


def analyze_lines(img_bytes: bytes) -> Dict:
    """تحليل اتجاه الخطوط"""
    try:
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)

        if img is None:
            return {"avg_angle": None, "horizontal": 0, "vertical": 0, "diagonal": 0, "total_lines": 0, "pattern": "فشل التحميل"}

        # تحسين الصورة
        img = cv2.GaussianBlur(img, (5, 5), 0)

        # كشف الحواف
        edges = cv2.Canny(img, 50, 150)

        # اكتشاف الخطوط
        lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                                threshold=50,
                                minLineLength=30,
                                maxLineGap=10)

        if lines is not None:
            angles = []
            horizontal = 0
            vertical = 0
            diagonal = 0

            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx = x2 - x1
                dy = y2 - y1

                # حساب الزاوية
                angle = np.degrees(np.arctan2(dy, dx))
                # تطبيع الزاوية بين -90 و 90
                if angle > 90:
                    angle -= 180
                elif angle < -90:
                    angle += 180

                angles.append(angle)

                # تصنيف الخطوط
                abs_angle = abs(angle)
                if abs_angle < 15 or abs_angle > 165:  # أفقي
                    horizontal += 1
                elif abs_angle > 75 and abs_angle < 105:  # عمودي
                    vertical += 1
                else:
                    diagonal += 1

            if angles:
                avg_angle = np.mean(angles)

                # تحليل نمط الخطوط
                total = horizontal + vertical + diagonal
                line_pattern = "متنوع"
                if total > 0:
                    # نحدد النمط المسيطر بناءً على النسبة
                    h_pct = horizontal / total
                    v_pct = vertical / total
                    d_pct = diagonal / total

                    if h_pct > 0.4 and h_pct > v_pct and h_pct > d_pct:
                        line_pattern = "أفقي"
                    elif v_pct > 0.4 and v_pct > h_pct and v_pct > d_pct:
                        line_pattern = "عمودي"
                    elif d_pct > 0.4 and d_pct > h_pct and d_pct > v_pct:
                        line_pattern = "مائل"
                    elif total < 5:
                        line_pattern = "قليل"

                return {
                    "avg_angle": round(float(avg_angle), 1),
                    "horizontal": horizontal,
                    "vertical": vertical,
                    "diagonal": diagonal,
                    "total_lines": len(lines),
                    "pattern": line_pattern,
                }

        return {"avg_angle": None, "horizontal": 0, "vertical": 0, "diagonal": 0,
                "total_lines": 0, "pattern": "قليل/غير محدد"}

    except Exception as e:
        logger.error(f"خطأ في تحليل الخطوط: {e}")
        return {"error": str(e), "total_lines": 0, "pattern": "خطأ"}


def analyze_shapes(img_bytes: bytes) -> Dict:
    """تحليل الأشكال في الرسم"""
    try:
        img = load_and_preprocess_image(img_bytes)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # تحسين الصورة
        gray = cv2.medianBlur(gray, 5)

        # العتبة التكيفية
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        # تنظيف الصورة
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # إيجاد الكنتورات
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        shape_counts = Counter()
        # مساحة كافية (مثلاً 0.01% من إجمالي البيكسلات)
        min_area = img.size * 0.0001

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
            sides = len(approx)

            # حساب الدائرية
            circularity = (4 * np.pi * area) / (perimeter *
                                                perimeter) if perimeter > 0 else 0

            shape_type = "other"

            if sides == 3:
                shape_type = "triangle"
            elif sides == 4:
                (x, y, w, h) = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / h if h > 0 else 1

                if 0.85 <= aspect_ratio <= 1.15:
                    shape_type = "square"
                else:
                    shape_type = "rectangle"
            elif sides > 4:
                if circularity > 0.80:
                    shape_type = "circle"
                elif circularity > 0.65 and sides > 5:
                    shape_type = "ellipse"
                elif sides >= 5:
                    shape_type = "polygon"

            shape_counts[shape_type] += 1

        # إيجاد الشكل المسيطر
        dominant_shape = shape_counts.most_common(
            1)[0][0] if shape_counts else "غير محدد"

        result = dict(shape_counts)
        result["dominant"] = dominant_shape
        result["total"] = len(contours)

        return result

    except Exception as e:
        logger.error(f"خطأ في تحليل الأشكال: {e}")
        return {"error": str(e), "total": 0, "dominant": "خطأ"}


def analyze_composition(img_bytes: bytes) -> Dict:
    """تحليل تركيب الصورة (قاعدة الثلاثيات)"""
    try:
        img = load_and_preprocess_image(img_bytes)
        height, width = img.shape[:2]

        # تقسيم الصورة إلى شبكة 3x3
        grid_h = height // 3
        grid_w = width // 3

        brightness_grid = []

        # تحويل إلى تدرج رمادي
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        for i in range(3):
            for j in range(3):
                y_start = i * grid_h
                y_end = (i + 1) * grid_h
                x_start = j * grid_w
                x_end = (j + 1) * grid_w

                cell = gray_img[y_start:y_end, x_start:x_end]

                if cell.size > 0:
                    # متوسط سطوع الخلية
                    brightness = np.mean(cell)
                    brightness_grid.append(brightness)
                else:
                    brightness_grid.append(0)

        # حساب مؤشرات التركيب
        if brightness_grid:
            brightness_variance = np.var(brightness_grid)

            composition_type = "متوازن"
            if brightness_variance > 1500:
                composition_type = "درامي/متباين"
            elif brightness_variance > 500:
                composition_type = "مركّز"

            # تحديد المنطقة الأكثر إشراقاً (منطقة التركيز)
            max_brightness_idx = np.argmax(brightness_grid)
            focus_area = ["أعلى-يسار", "أعلى-وسط", "أعلى-يمين",
                          "وسط-يسار", "مركزي", "وسط-يمين",
                          "أسفل-يسار", "أسفل-وسط", "أسفل-يمين"][max_brightness_idx]

            return {
                "composition_type": composition_type,
                "focus_area": focus_area,
                "balance": "جيد" if brightness_variance < 1000 else "متوسط",
                "brightness_variance": round(float(brightness_variance), 1),
            }

        return {"composition_type": "غير محدد", "focus_area": "غير محدد", "balance": "غير محدد"}

    except Exception as e:
        logger.error(f"خطأ في تحليل التركيب: {e}")
        return {"error": str(e)}


def analyze_complexity(img_bytes: bytes) -> Dict:
    """تحليل تعقيد الرسم"""
    try:
        img = load_and_preprocess_image(img_bytes)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # حساب كثافة الحواف (Edge Density)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        # حساب تباين الصورة (Contrast)
        contrast = np.std(gray)

        # حساب تفاصيل الصورة (Detail Level)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        detail_level = np.var(laplacian)

        # تقييم التعقيد
        complexity_score = (edge_density * 100) + \
            (contrast / 10) + (detail_level / 1000)

        if complexity_score > 50:
            complexity_level = "عالي"
        elif complexity_score > 20:
            complexity_level = "متوسط"
        else:
            complexity_level = "بسيط"

        return {
            "edge_density": round(float(edge_density * 100), 1),
            "contrast": round(float(contrast), 1),
            "detail_level": round(float(detail_level), 1),
            "complexity_score": round(float(complexity_score), 1),
            "complexity_level": complexity_level
        }

    except Exception as e:
        logger.error(f"خطأ في تحليل التعقيد: {e}")
        return {"error": str(e)}


def blip_caption(image_bytes: bytes) -> str:
    """وصف الصورة باستخدام Hugging Face API"""
    try:
        if not HF_TOKEN:
            logger.error("❌ HF_TOKEN غير متوفر")
            return "وصف غير متاح - يرجى التحقق من إعدادات النظام"

        headers = {"Authorization": f"Bearer {HF_TOKEN}"}

        # استخدام نموذج موثوق (nlpconnect/vit-gpt2-image-captioning)
        HF_API_URL = "https://api-inference.huggingface.co/models/nlpconnect/vit-gpt2-image-captioning"

        response = requests.post(
            HF_API_URL,
            headers=headers,
            data=image_bytes,  # تمرير البايتات مباشرة
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                caption = result[0].get(
                    "generated_text") or result[0].get("caption")
                if caption:
                    # **ملاحظة:** هنا يجب إضافة دالة ترجمة (مثل Google Translate) لتحويل الإنجليزية إلى العربية
                    # حالياً سنعيد النص كما هو
                    return caption

        # جرب نموذج Salesforce/blip-image-captioning-base كبديل
        logger.warning(
            f"⚠️ فشل النموذج الأول ({response.status_code}). تجربة النموذج الثاني...")
        HF_API_URL_2 = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"

        response = requests.post(
            HF_API_URL_2,
            headers=headers,
            data=image_bytes,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                caption = result[0].get("generated_text")
                if caption:
                    return caption

        return "رسم طفل بألوان متنوعة وخطوط معبرة"

    except requests.exceptions.Timeout:
        logger.warning("انتهت مهلة الاتصال بـ Hugging Face")
        return "رسم طفل إبداعي"
    except Exception as e:
        logger.error(f"❌ خطأ في وصف الصورة: {e}")
        return "رسم معبر من طفل"


def generate_psychological_analysis(analysis_results: Dict) -> List[str]:
    """توليد تحليل نفسي بناءً على النتائج"""
    psychological_notes = []

    # تحليل الألوان
    colors = analysis_results.get("colors", [])
    color_emotions = analyze_emotion_from_colors(colors)
    psychological_notes.extend(color_emotions)

    # تحليل الخطوط
    lines = analysis_results.get("lines", {})
    line_pattern = lines.get("pattern", "")
    if line_pattern == "أفقي":
        psychological_notes.append(
            "الخطوط الأفقية تدل على الاستقرار والهدوء النفسي")
    elif line_pattern == "عمودي":
        psychological_notes.append(
            "الخطوط العمودية تشير إلى الطموح والثقة بالنفس")
    elif line_pattern == "مائل":
        psychological_notes.append(
            "الخطوط المائلة تعبر عن الحركة والطاقة والرغبة في التغيير")
    elif line_pattern == "قليل/غير محدد" and lines.get("total_lines", 0) < 5:
        psychological_notes.append(
            "قلة الخطوط قد تشير إلى الحذر أو بساطة في التعبير")

    # تحليل الأشكال
    shapes = analysis_results.get("shapes", {})
    dominant_shape = shapes.get("dominant", "")

    if dominant_shape == "circle":
        psychological_notes.append("الدوائر تعبر عن المرونة والانسجام العاطفي")
    elif dominant_shape == "triangle":
        psychological_notes.append(
            "المثلثات تشير إلى الطموح والتوجه نحو الأهداف")
    elif dominant_shape == "square" or dominant_shape == "rectangle":
        psychological_notes.append(
            "الأشكال المضلعة (مربعات/مستطيلات) تعبر عن النظام والتفكير المنطقي")
    elif shapes.get("total", 0) == 0:
        psychological_notes.append(
            "عدم وجود أشكال واضحة قد يدل على التركيز على الحركة واللون أكثر من التفاصيل")

    # تحليل التركيب
    composition = analysis_results.get("composition", {})
    composition_type = composition.get("composition_type", "")
    if composition_type == "متوازن":
        psychological_notes.append(
            "التركيب المتوازن يدل على شخصية منظمة ومرتاحة")
    elif composition_type == "درامي/متباين":
        psychological_notes.append(
            "التباين العالي في السطوع قد يعبر عن مشاعر قوية أو صراع داخلي")

    if composition.get("focus_area") == "مركزي":
        psychological_notes.append(
            "وضع العناصر في المنتصف يدل على الثقة بالنفس والاهتمام بالنفس")

    # تحليل التعقيد
    complexity = analysis_results.get("complexity", {})
    complexity_level = complexity.get("complexity_level", "")
    if complexity_level == "عالي":
        psychological_notes.append(
            "التعقيد العالي يشير إلى تفكير متعدد الأبعاد وقدرة على الملاحظة الدقيقة")
    elif complexity_level == "بسيط":
        psychological_notes.append(
            "البساطة تدل على الوضوح والصراحة والتركيز على الفكرة الأساسية")

    # إزالة التكرارات
    return list(dict.fromkeys(psychological_notes))[:5]


def generate_educational_advice(analysis_results: Dict) -> List[str]:
    """توليد نصائح تربوية"""
    advice = []

    # نصائح عامة
    advice.append("شجعوا الطفل على الاستمرار في الرسم والتعبير الفني")
    advice.append("وفرا له أدوات رسم متنوعة الألوان والأنواع")
    advice.append(
        "ناقشوا معه رسوماته واسألوه عن معانيها وماذا يشعر عندما يرسمها")

    # نصائح بناءً على التحليل
    shapes = analysis_results.get("shapes", {})
    if shapes.get("total", 0) < 3:
        advice.append(
            "شجع الطفل على رسم أشكال أكثر تنوعاً والتركيز على التفاصيل")

    colors = analysis_results.get("colors", [])
    if len(colors) < 3:
        advice.append(
            "قدم للطفل ألواناً جديدة لاستكشافها وتجربة تراكيب لونية مختلفة")

    if any(c['name'] in ['أسود', 'رمادي'] and c['percentage'] > 30 for c in colors):
        advice.append(
            "إذا كان يغلب على الرسم الألوان الداكنة، تأكد من قضاء الطفل وقتاً كافياً في اللعب بالخارج")

    complexity = analysis_results.get("complexity", {})
    if complexity.get("complexity_level") == "بسيط":
        advice.append("شجع الطفل على إضافة تفاصيل أكثر لرسوماته")

    return advice[:5]


def generate_report(image_bytes: bytes, child_id: str, child_name: str = "", child_age: str = "") -> Dict:
    """توليد تقرير شامل وإرجاع البيانات الخام"""

    logger.info(f"📊 بدء تحليل رسم الطفل {child_id}")

    # جمع كل التحليلات
    analysis_results = {
        "colors": analyze_colors(image_bytes),
        "lines": analyze_lines(image_bytes),
        "shapes": analyze_shapes(image_bytes),
        "composition": analyze_composition(image_bytes),
        "complexity": analyze_complexity(image_bytes),
        "caption": blip_caption(image_bytes)
    }

    # توليد التحليلات النفسية والتربوية
    psychological_notes = generate_psychological_analysis(analysis_results)
    educational_advice = generate_educational_advice(analysis_results)

    # بناء التقرير النصي (مقسم لتسهيل القراءة)
    child_info = child_name if child_name else f"الطفل {child_id}"
    if child_age:
        child_info += f" (العمر: {child_age})"

    report_sections = []
    report_sections.append(f"""
تقرير تحليل الرسم الفني
========================

🎨 الطفل: {child_info}
📅 تاريخ التحليل: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'-' * 50}

🔍 الوصف العام:
{analysis_results['caption']}
""")

    report_sections.append("🎨 تحليل الألوان:")
    for i, color in enumerate(analysis_results['colors'][:3], 1):
        report_sections.append(
            f"{i}. {color['name']} - {color['hex']} ({color['percentage']}%)")

    lines_info = analysis_results['lines']
    report_sections.append(f"""
📏 تحليل الخطوط:
• النمط المسيطر: {lines_info.get('pattern', 'غير محدد')}
• الخطوط الأفقية: {lines_info.get('horizontal', 0)}
• الخطوط العمودية: {lines_info.get('vertical', 0)}
• الخطوط المائلة: {lines_info.get('diagonal', 0)}
• المجموع: {lines_info.get('total_lines', 0)} خط
""")

    shapes_info = analysis_results['shapes']
    report_sections.append(f"""
🔷 تحليل الأشكال:
• الشكل المسيطر: {shapes_info.get('dominant', 'غير محدد')}
• دوائر: {shapes_info.get('circle', 0)}
• مثلثات: {shapes_info.get('triangle', 0)}
• مربعات/مستطيلات: {shapes_info.get('square', 0) + shapes_info.get('rectangle', 0)}
• أشكال أخرى: {shapes_info.get('other', 0)}
""")

    report_sections.append(f"""
{'-' * 50}

💡 التحليل النفسي:
""")
    for i, note in enumerate(psychological_notes, 1):
        report_sections.append(f"• {note}")

    report_sections.append(f"""
{'-' * 50}

📚 نصائح تربوية:
""")
    for i, advice in enumerate(educational_advice, 1):
        report_sections.append(f"{i}. {advice}")

    report_sections.append(f"""
{'-' * 50}

✨ ملاحظات إضافية:
• تعقيد الرسم: {analysis_results['complexity'].get('complexity_level', 'غير محدد')} (الدرجة: {analysis_results['complexity'].get('complexity_score', 0)} / 100)
• تركيب الصورة: {analysis_results['composition'].get('composition_type', 'غير محدد')}
• منطقة التركيز: {analysis_results['composition'].get('focus_area', 'غير محدد')}

{'-' * 50}

💝 ختاماً:
نشجعكم على متابعة موهبة طفلكم الفنية، فالرسم بوابة للتعبير عن المشاعر
وتنمية الخيال والإبداع. كل رسمة هي نافذة إلى عالم الطفل الداخلي.

مع أطيب التمنيات،
فريق عمل أفق للتحليل النفسي الفني
📧 support@afaq.com
""")

    full_report_text = "\n".join(report_sections).strip()

    # دمج نتائج التحليل والنصائح في قاموس واحد
    full_analysis = {
        "report_text": full_report_text,
        "psychological_notes": psychological_notes,
        "educational_advice": educational_advice,
        "raw_analysis": analysis_results
    }

    return full_analysis


def send_email_gmail(parent_email: str, subject: str, analysis_text: str) -> bool:
    """إرسال إيميل بسيط يعمل في معظم الحالات"""
    try:
        logger.info(f"📧 بدء إرسال إيميل إلى: {parent_email}")

        if not GMAIL_SENDER or not GMAIL_APP_PASSWORD or not parent_email or '@' not in parent_email:
            logger.error("❌ بيانات البريد ناقصة أو إيميل غير صالح")
            return False

        # إنشاء الرسالة
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = GMAIL_SENDER
        msg['To'] = parent_email

        # المحتوى النصي
        text_content = analysis_text

        # المحتوى HTML
        html_content = f"""
<!DOCTYPE html>
<html dir="rtl">
<head>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: 'Arial', sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 700px; margin: 0 auto; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }}
        .header {{ background: #4CAF50; color: white; padding: 20px; text-align: center; border-radius: 8px 8px 0 0;}}
        .content {{ padding: 20px; background: #f9f9f9; white-space: pre-wrap; }}
        .footer {{ margin-top: 20px; padding: 10px; text-align: center; color: #666; font-size: 12px; border-top: 1px solid #ddd; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h2>تقرير تحليل الرسم الفني</h2>
            <p>مشروع أفاق للتحليل النفسي للأطفال</p>
        </div>
        <div class="content">
            <pre style="white-space: pre-wrap; font-family: inherit;">{analysis_text}</pre>
        </div>
        <div class="footer">
            <p>تم إنشاء هذا التقرير تلقائياً. © 2024 مشروع أفاق</p>
        </div>
    </div>
</body>
</html>
        """

        # إرفاق المحتوى
        part1 = MIMEText(text_content, 'plain', 'utf-8')
        part2 = MIMEText(html_content, 'html', 'utf-8')
        msg.attach(part1)
        msg.attach(part2)

        # إرسال الإيميل
        server = None
        try:
          
            server = smtplib.SMTP("smtp.gmail.com", 587, timeout=10)
            server.starttls()
            server.login(GMAIL_SENDER, GMAIL_APP_PASSWORD)
            server.send_message(msg)
            logger.info("✅ تم إرسال الإيميل بنجاح (587/TLS)")
            return True
        except Exception as e1:
            logger.warning(f"⚠️ فشل الإرسال عبر 587: {e1}. تجربة 465 (SSL)...")
            if server:
                server.quit()  

            
            try:
                server = smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=10)
                server.login(GMAIL_SENDER, GMAIL_APP_PASSWORD)
                server.send_message(msg)
                logger.info("✅ تم إرسال الإيميل بنجاح (465/SSL)")
                return True
            except Exception as e2:
                logger.error(f"❌ فشل الإرسال عبر 465 أيضاً: {e2}")
                raise

        finally:
            if server:
                server.quit()

    except Exception as e:
        logger.error(f"❌ فشل إرسال الإيميل نهائياً: {e}")
        return False


# ----------------------------------
# خادم Flask
# ----------------------------------
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB
app.config['JSON_AS_ASCII'] = False  # دعم العربية في JSON

# Rate limiting
request_log = {}


def rate_limit(max_per_minute=10):
    """محدد للطلبات"""
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            ip = request.remote_addr
            now = datetime.now()

            # تنظيف الطلبات القديمة
            if ip in request_log:
                request_log[ip] = [req_time for req_time in request_log[ip]
                                   if now - req_time < timedelta(minutes=1)]

            # التحقق من الحد
            if ip in request_log and len(request_log[ip]) >= max_per_minute:
                logger.warning(f"⏰ تجاوز الحد للـ IP: {ip}")
                return jsonify({
                    "status": "error",
                    "message": "تم تجاوز الحد المسموح للطلبات. الرجاء الانتظار دقيقة."
                }), 429

            # تسجيل الطلب
            if ip not in request_log:
                request_log[ip] = []
            request_log[ip].append(now)

            return f(*args, **kwargs)
        return wrapped
    return decorator


@app.route('/')
def home():
    return jsonify({
        "status": "running",
        "service": "Afaq Drawing Analysis API",
        "version": "3.1 (In-Memory Refactor)",
        "endpoints": {
            "/health": "فحص حالة الخادم",
            "/analyze": "تحليل الرسم وإرسال التقرير (POST)",
            "/analyze-only": "تحليل الرسم فقط (POST)",
            "/test": "فحص الاتصالات"
        },
        "documentation": "https://github.com/afaq-project/docs"
    })


@app.route('/health')
def health_check():
    """فحص حالة الخادم"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "hf_token_configured": bool(HF_TOKEN),
        "email_configured": bool(GMAIL_SENDER and GMAIL_APP_PASSWORD),
    }
    return jsonify(health_status)


@app.route('/test', methods=['GET'])
def test_connections():
    """فحص الاتصالات الخارجية"""
    tests = {
        "huggingface": "Skipped (need image)",
        "gmail": False,
    }

    # فحص Gmail
    try:
        if GMAIL_SENDER and GMAIL_APP_PASSWORD:
            server = smtplib.SMTP("smtp.gmail.com", 587, timeout=5)
            server.quit()
            tests["gmail"] = True
        else:
            tests["gmail"] = "no_credentials"
    except Exception as e:
        tests["gmail"] = f"Failed: {str(e)}"

    return jsonify({
        "status": "success",
        "tests": tests,
        "message": "تم فحص الاتصالات (HuggingFace يتطلب إرسال صورة)"
    })


def common_analysis_logic(send_email: bool = True):
    """منطق مشترك لنقاط النهاية /analyze و /analyze-only"""
    request_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{abs(hash(request.remote_addr))}"
    logger.info(
        f"📥 بدء طلب جديد [ID: {request_id}] من {request.remote_addr} (Email: {send_email})")

    if 'image' not in request.files:
        abort(400, "لم يتم العثور على ملف الصورة (المتوقع 'image')")

    file = request.files['image']
    child_id = request.form.get('child_id', f"Anon_{request_id}")
    child_name = request.form.get('child_name', '')
    child_age = request.form.get('child_age', '')
    parent_email = request.form.get('parent_email', '')

    file_content = file.read()  # قراءة محتوى الملف بالكامل في الذاكرة

    # 1. التحقق من الصحة
    is_valid, message = validate_image_file_content(
        file_content, file.filename)
    if not is_valid:
        logger.warning(f"❌ ملف غير صالح: {message}")
        abort(400, message)

    analysis_start = time.time()

    try:
        # 2. توليد التقرير
        analysis_data = generate_report(
            file_content,
            child_id,
            child_name,
            child_age
        )
        report_text = analysis_data['report_text']

        analysis_time = time.time() - analysis_start
        logger.info(f"✅ تحليل الرسم اكتمل في {analysis_time:.2f} ثانية")

        if send_email:
            # 3. إرسال الإيميل
            email_subject = f"تقرير رسم الطفل {child_id} - مشروع أفق"
            email_sent = send_email_gmail(
                parent_email, email_subject, report_text)

            email_status = "✅ تم الإرسال بنجاح" if email_sent else "❌ فشل الإرسال"
            logger.info(f"📧 حالة الإيميل: {email_status}")

            return jsonify({
                "status": "success",
                "request_id": request_id,
                "child_id": child_id,
                "email_status": email_status,
                "analysis_time_s": round(analysis_time, 2),
                "report_summary": analysis_data['psychological_notes'][:2] + analysis_data['educational_advice'][:2],
                # "full_report_text": report_text # يمكن حذفه للحفاظ على نظافة الـ JSON
            }), 200
        else:
            # 4. إرجاع النتيجة فقط
            return jsonify({
                "status": "success",
                "request_id": request_id,
                "child_id": child_id,
                "analysis_time_s": round(analysis_time, 2),
                "analysis_data": analysis_data  # إرجاع البيانات الخام والنص
            }), 200

    except ValueError as ve:
        logger.error(f"❌ خطأ في معالجة الصورة: {ve}")
        return jsonify({
            "status": "error",
            "message": f"خطأ في معالجة الصورة: {str(ve)}"
        }), 400
    except Exception as e:
        logger.error(f"❌ خطأ غير متوقع [ID: {request_id}]: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error",
            "message": f"حدث خطأ غير متوقع: {str(e)}",
            "traceback": traceback.format_exc().splitlines()
        }), 500


@app.route('/analyze', methods=['POST'])
@rate_limit(max_per_minute=5)
def analyze_and_notify():
    """نقطة النهاية الرئيسية للتحليل وإرسال الإيميل"""
    return common_analysis_logic(send_email=True)


@app.route('/analyze-only', methods=['POST'])
@rate_limit(max_per_minute=15)
def analyze_only():
    """تحليل الرسم وإرجاع النتيجة في الـ JSON فقط (بدون إيميل)"""
    return common_analysis_logic(send_email=False)


 
