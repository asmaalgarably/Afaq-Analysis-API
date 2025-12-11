from dotenv import load_dotenv
import os
import cv2
import numpy as np
import requests
from flask import Flask, request, jsonify, abort, render_template_string
from collections import Counter
import traceback
from typing import Dict, List, Tuple
from functools import wraps
import logging
from datetime import datetime, timedelta
import re
import time
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from bson.json_util import dumps

# ----------------------------------
# إعدادات التسجيل
# ----------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ----------------------------------
# إعدادات المشروع (متغيرات البيئة)
# ----------------------------------
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
GMAIL_SENDER = os.getenv("GMAIL_SENDER", "").strip()
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY", "").strip()
MONGO_URI = os.getenv("MONGO_URI", "").strip()

REQUIRED_ENV_VARS = ["HF_TOKEN", "GMAIL_SENDER",
                     "SENDGRID_API_KEY", "MONGO_URI"]
missing_vars = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
if missing_vars:
    logger.warning(f"متغيرات بيئية مفقودة: {missing_vars}")

# ----------------------------------
# إعداد قاعدة البيانات
# ----------------------------------

MONGO_CLIENT = None
MONGO_DB = None
REPORT_COLLECTION = None


def setup_mongo():
    """تهيئة اتصال MongoDB"""
    global MONGO_CLIENT, MONGO_DB, REPORT_COLLECTION

    if not MONGO_URI:
        logger.error("MONGO_URI غير متوفر. لن يتم حفظ التقارير.")
        return False

    try:
        # إعداد العميل والاتصال
        MONGO_CLIENT = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10000)
        # محاولة عمل ping للتحقق من الاتصال
        MONGO_CLIENT.admin.command('ping')

        MONGO_DB = MONGO_CLIENT.get_database(
            "AfaqAnalysisDB")  # اسم قاعدة البيانات
        REPORT_COLLECTION = MONGO_DB.get_collection(
            "reports")   
        logger.info("تم الاتصال بقاعدة بيانات MongoDB بنجاح.")
        return True
    except ConnectionFailure as e:
        logger.error(f"فشل الاتصال بقاعدة بيانات MongoDB: {e}")
        return False
    except Exception as e:
        logger.error(f"خطأ غير متوقع في إعداد MongoDB: {e}")
        return False


# استدعاء دالة التهيئة عند بدء تشغيل التطبيق
setup_mongo()


# ----------------------------------
# دوال مساعدة وتحليل الصور (CV)
# ----------------------------------

def validate_image_file_content(file_bytes: bytes, filename: str) -> Tuple[bool, str]:
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    if not file_bytes:
        return False, "لم يتم توفير محتوى الملف"

    filename = (filename or '').lower().strip()
    _, ext = os.path.splitext(filename)

    if not ext or ext not in ALLOWED_EXTENSIONS:
        allowed = ', '.join(ALLOWED_EXTENSIONS)
        return False, f"نوع الملف غير مدعوم. المسموح: {allowed}"

    file_size = len(file_bytes)
    if file_size == 0:
        return False, 
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

    min_distance = float('inf')
    closest_color = "غير معروف"

    for color_rgb, color_name in colors_db:
        distance = np.sqrt((r - color_rgb[0])**2 +
                           (g - color_rgb[1])**2 +
                           (b - color_rgb[2])**2)
        if distance < 50 and distance < min_distance:   
            min_distance = distance
            closest_color = color_name
        elif min_distance == float('inf'):
            
            if r > 200 and g < 50 and b < 50:
                closest_color = "أحمر ساطع"
            elif r < 50 and g > 200 and b < 50:
                closest_color = "أخضر ساطع"
            elif r < 50 and g < 50 and b > 200:
                closest_color = "أزرق ساطع"
            elif r > 200 and g > 200 and b < 50:
                closest_color = "أصفر ساطع"
            elif r > 150 and g > 150 and b > 150:
                closest_color = "لون فاتح"
            elif r < 100 and g < 100 and b < 100:
                closest_color = "لون داكن"

    return closest_color


def analyze_colors(img_bytes: bytes) -> List[Dict]:
    try:
        img = load_and_preprocess_image(img_bytes)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # تقليل دقة الصورة لتسريع التجميع
        img_small = cv2.resize(img_rgb, (100, 100),
                               interpolation=cv2.INTER_AREA)
        pixels = img_small.reshape(-1, 3)

        # تجميع الألوان يدوياً
        color_bins = {}
        bin_size = 32

        for pixel in pixels:
            r, g, b = pixel
            # تقريب الألوان
            r_bin = (r // bin_size) * bin_size
            g_bin = (g // bin_size) * bin_size
            b_bin = (b // bin_size) * bin_size

            key = (r_bin, g_bin, b_bin)
            color_bins[key] = color_bins.get(key, 0) + 1

        # الحصول على أكثر 5 ألوان انتشاراً
        sorted_colors = sorted(
            color_bins.items(), key=lambda x: x[1], reverse=True)[:5]

        total_pixels = len(pixels)
        colors_info = []
        for (r, g, b), count in sorted_colors:
            color_hex = rgb_to_hex(r, g, b)
            avg_r, avg_g, avg_b = r + bin_size//2, g + bin_size//2, b + bin_size//2
            color_name = get_color_name((avg_r, avg_g, avg_b))
            percentage = (count / total_pixels) * 100

            colors_info.append({
                "rgb": f"({avg_r}, {avg_g}, {avg_b})",
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
        'أحمر': 'طاقة، حماس، عاطفة قوية',
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
    for color_info in colors:
        color_name = color_info.get("name", "")
        if color_name in emotion_map and color_info.get("percentage", 0) > 10:
            emotions.append(f"{color_name}: {emotion_map[color_name]}")

    return list(dict.fromkeys(emotions))[:5]  


def analyze_lines(img_bytes: bytes) -> Dict:
    """تحليل اتجاه الخطوط"""
    try:
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)

        if img is None:
            return {"avg_angle": None, "horizontal": 0, "vertical": 0, "diagonal": 0, "total_lines": 0, "pattern": "فشل التحميل"}

        img = cv2.GaussianBlur(img, (5, 5), 0)
        edges = cv2.Canny(img, 50, 150)

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

                angle = np.degrees(np.arctan2(dy, dx))
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
                    h_pct = horizontal / total
                    v_pct = vertical / total
                    d_pct = diagonal / total

                    if h_pct > 0.4 and h_pct > v_pct and h_pct > d_pct:
                        line_pattern = "أفقي"
                    elif v_pct > 0.4 and v_pct > h_pct and v_pct > d_pct:
                        line_pattern = "عمودي"
                    elif d_pct > 0.4 and d_pct > h_pct and d_pct > v_pct:
                        line_pattern = "مائل/ديناميكي"
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

        gray = cv2.medianBlur(gray, 5)

        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        shape_counts = Counter()
        min_area = img.size * 0.0001  

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
            sides = len(approx)

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

        grid_h = height // 3
        grid_w = width // 3

        brightness_grid = []
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        for i in range(3):
            for j in range(3):
                y_start = i * grid_h
                y_end = (i + 1) * grid_h
                x_start = j * grid_w
                x_end = (j + 1) * grid_w

                cell = gray_img[y_start:y_end, x_start:x_end]

                if cell.size > 0:
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
        # معادلة تقريبية للدرجة
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
            return "وصف غير متاح - يرجى التحقق من إعدادات النظام"

        headers = {"Authorization": f"Bearer {HF_TOKEN}"}

        HF_API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"

        response = requests.post(
            HF_API_URL,
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

        # محاولة أخرى للنموذج الأقوى إذا فشل الأول
        HF_API_URL_2 = "https://api-inference.huggingface.co/models/nlpconnect/vit-gpt2-image-captioning"
        response = requests.post(
            HF_API_URL_2,
            headers=headers,
            data=image_bytes,
            timeout=30
        )
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                caption = result[0].get(
                    "generated_text") or result[0].get("caption")
                if caption:
                    return caption

        logger.warning(
            f"⚠️ فشل كشف الوصف. Status: {response.status_code}. Details: {response.text}")
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
    elif line_pattern == "مائل/ديناميكي":
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
    """توليد تقرير شامل وحفظه في MongoDB"""

    logger.info(f"📊 بدء تحليل وحفظ رسم الطفل {child_id}")

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

    # بناء التقرير النصي
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

    # ---------------------------------------------
    #   منطق حفظ التقرير في MongoDB 
    # ---------------------------------------------
    if REPORT_COLLECTION:
        try:
            # يجب التأكد من قراءة parent_email من request.form لأنه غير موجود كـ argument للدالة
            parent_email = request.form.get('parent_email', 'غير محدد')

            report_document = {
                "child_id": child_id,
                "child_name": child_name,
                "child_age": child_age,
                "parent_email": parent_email,
                "full_report_text": full_report_text,
                "raw_data": analysis_results,
                "created_at": datetime.utcnow()
            }
            # تخزين التقرير 
            REPORT_COLLECTION.replace_one(
                {'child_id': child_id},
                report_document,
                upsert=True
            )
            logger.info(f"تم حفظ التقرير بنجاح في MongoDB لـ {child_id}")
        except Exception as e:
            logger.error(f"فشل حفظ التقرير في MongoDB: {e}")

    return full_analysis


# -------------------------------------------------------------
#  دالة إرسال الإيميل باستخدام SendGrid
# -------------------------------------------------------------
def send_email_sendgrid(parent_email: str, subject: str, analysis_data: Dict, child_name: str, child_id: str) -> Tuple[bool, str]:
    """إرسال إيميل التقرير باستخدام SendGrid API"""
    try:
        logger.info(f"📧 بدء إرسال إيميل إلى: {parent_email} عبر SendGrid")

        if not SENDGRID_API_KEY or not GMAIL_SENDER or not parent_email:
            return False, "بيانات البريد ناقصة أو إيميل غير صالح أو مفتاح SendGrid مفقود"

        # 1. تجهيز المحتوى الديناميكي لـ HTML (ملخص النقاط)
        psychological_notes = analysis_data.get('psychological_notes', [])
        educational_advice = analysis_data.get('educational_advice', [])
        summary_points = psychological_notes + educational_advice

        summary_html = ""
        for item in summary_points[:5]:
            summary_html += f'<li style="margin-bottom: 8px; font-size: 15px;"><span style="color: #28a745;">✅</span> {item}</li>'

        # رابط لصفحة التقرير الكامل
        report_link = f"https://afaq-analysis-api.onrender.com/report/{child_id}"

        # 2. بناء محتوى الإيميل HTML
        html_content = f"""
        <!DOCTYPE html>
        <html lang="ar" dir="rtl">
        <head>
            <meta charset="UTF-8">
            <title>{subject}</title>
        </head>
        <body style="font-family: Arial, Tahoma, sans-serif; line-height: 1.6; color: #333333; background-color: #f4f4f4; padding: 20px;">
            <div style="max-width: 600px; margin: 20px auto; padding: 20px; border: 1px solid #e0e0e0; border-radius: 10px; background-color: #ffffff;">

                <div style="text-align: center; padding-bottom: 20px; border-bottom: 1px solid #eeeeee;">
                    <h1 style="color: #007bff; margin: 0; font-size: 24px;">نتائج تحليل رسم طفلك ✨</h1>
                    <p style="color: #666; margin: 5px 0 0;">منصة أفق للتحليل النفسي الفني</p>
                </div>

                <div style="padding-top: 20px;">
                    <p style="font-size: 16px;">
                        تحية طيبة، ولي أمر **{child_name}**،
                    </p>
                    <p style="font-size: 16px;">
                        يسر فريق "أفق" أن يشاركك تقرير تحليل الرسم الخاص بطفلك. نقدم لك ملخصًا لأهم الاستنتاجات:
                    </p>

                    <div style="background-color: #f7f7f7; padding: 15px; border-radius: 8px; margin-top: 20px; margin-bottom: 20px; border-right: 5px solid #28a745;">
                        <h3 style="color: #28a745; margin-top: 0; font-size: 18px; text-align: right;">📋 أهم النقاط:</h3>
                        <ul style="list-style-type: none; padding-right: 0; margin-right: 15px;">
                            {summary_html}
                        </ul>
                    </div>

                    <p style="font-size: 16px;">
                        للاطلاع على التقرير التفصيلي الكامل، يرجى النقر على الزر أدناه.
                    </p>

                    <div style="text-align: center; margin-top: 30px;">
                        <a href="{report_link}" 
                            style="display: inline-block; padding: 12px 25px; background-color: #007bff; color: #ffffff; text-decoration: none; border-radius: 5px; font-weight: bold; font-size: 16px;">
                            عرض التقرير الكامل الآن
                        </a>
                    </div>
                </div>

                <div style="text-align: center; padding-top: 20px; border-top: 1px solid #eeeeee; margin-top: 20px;">
                    <p style="margin: 0; font-size: 12px; color: #999999;">
                        هذه الرسالة تم إرسالها تلقائيًا. شكرًا لثقتكم في "أفق".
                    </p>
                </div>
            </div>
        </body>
        </html>
        """

        # 3. إرسال الإيميل باستخدام SendGrid
        message = Mail(
            from_email=GMAIL_SENDER,
            to_emails=parent_email,
            subject=subject,
            html_content=html_content
        )

        sg = SendGridAPIClient(SENDGRID_API_KEY)
        response = sg.send(message)

        if response.status_code == 202:
            logger.info("تم الإرسال بنجاح عبر SendGrid (Status 202)")
            return True, "تم الإرسال بنجاح عبر SendGrid"
        else:
            error_details = response.body.decode(
                'utf-8') if response.body else "لا يوجد تفاصيل."
            logger.error(
                f" فشل إرسال SendGrid. Status: {response.status_code}. Details: {error_details}")
            return False, f"فشل SendGrid. Status: {response.status_code}"

    except Exception as e:
        logger.error(
            f" خطأ غير متوقع في إرسال SendGrid: {traceback.format_exc()}")
        return False, f"خطأ غير متوقع: {str(e)}"


# ----------------------------------
# خادم Flask
# ----------------------------------
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB
app.config['JSON_AS_ASCII'] = False

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
                logger.warning(f"  تجاوز الحد للـ IP: {ip}")
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
        "version": "4.0 (MongoDB Integrated)",
        "endpoints": {
            "/health": "فحص حالة الخادم",
            "/analyze": "تحليل الرسم وحفظه وإرسال التقرير (POST)",
            "/analyze-only": "تحليل وحفظ الرسم فقط (POST)",
            "/report/<child_id>": "عرض التقرير الكامل من MongoDB"
        },
        "documentation": "https://github.com/afaq-project/docs"
    })


@app.route('/health')
def health_check():
    """فحص حالة الخادم واتصال MongoDB"""
    mongo_status = "unconfigured"
    try:
        if MONGO_CLIENT:
            # محاولة عمل أمر بسيط للتحقق
            MONGO_CLIENT.admin.command('ping')
            mongo_status = "connected"
        elif MONGO_URI:
            mongo_status = "reconnect_needed"

    except Exception:
        mongo_status = "failed_to_connect"

    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "hf_token_configured": bool(HF_TOKEN),
        "email_configured": bool(GMAIL_SENDER and SENDGRID_API_KEY),
        "mongo_status": mongo_status
    }
    return jsonify(health_status)


def common_analysis_logic(send_email: bool = True):
    """منطق مشترك لنقاط النهاية /analyze و /analyze-only"""
    start_time = time.time()

    # 1. التحقق من الإدخالات الأساسية
    if 'image' not in request.files:
        return jsonify({"status": "error", "message": "ملف الصورة مفقود."}), 400

    required_form_fields = ['child_id']
    if send_email:
        required_form_fields.append('parent_email')

    for field in required_form_fields:
        if field not in request.form:
            return jsonify({"status": "error", "message": f"الحقل المطلوب '{field}' مفقود."}), 400

    image_file = request.files['image']
    child_id = request.form['child_id'].strip()
    parent_email = request.form.get('parent_email', '').strip()
    child_name = request.form.get('child_name', 'طفل غير مسمى').strip()
    child_age = request.form.get('child_age', 'غير محدد').strip()

    # 2. تحميل ومراجعة ملف الصورة
    try:
        image_bytes = image_file.read()
        is_valid, error_msg = validate_image_file_content(
            image_bytes, image_file.filename)
        if not is_valid:
            return jsonify({"status": "error", "message": error_msg}), 400
    except Exception as e:
        logger.error(f"خطأ في قراءة ملف الصورة: {e}")
        return jsonify({"status": "error", "message": "خطأ في قراءة ملف الصورة."}), 500

    # 3. توليد التقرير والتحليل الفعلي 
    try:
        analysis_data = generate_report(
            image_bytes, child_id, child_name, child_age)

        report_link = f"https://afaq-analysis-api.onrender.com/report/{child_id}"

        analysis_time_sec = time.time() - start_time

        email_success, email_msg = "لم يُرسل", "لم يُرسل"

        # 4. إرسال الإيميل 
        if send_email and parent_email:
            email_subject = f"تقرير تحليل رسم الطفل: {child_name}"
            email_success, email_msg = send_email_sendgrid(
                parent_email=parent_email,
                subject=email_subject,
                analysis_data=analysis_data,
                child_name=child_name,
                child_id=child_id
            )

        # 5. إعداد الاستجابة
        response_data = {
            "status": "success",
            "message": "تم تحليل الرسم وحفظه وإرسال التقرير بنجاح." if send_email else "تم تحليل الرسم وحفظه بنجاح.",
            "analysis_time": round(analysis_time_sec, 2),
            "child_id": child_id,
            "child_name": child_name,
            "report_link": report_link,
            "email_status": "success" if email_success else "failed",
            "email_message": email_msg,
            "raw_analysis": analysis_data['raw_analysis']
        }

        if not send_email:
            response_data["full_report_text"] = analysis_data['report_text']

        return jsonify(response_data)

    except Exception as e:
        logger.error(
            f"خطأ غير متوقع أثناء التحليل: {traceback.format_exc()}")
        return jsonify({"status": "error", "message": f"حدث خطأ غير متوقع في الخادم: {str(e)}"}), 500


@app.route('/analyze', methods=['POST'])
@rate_limit(max_per_minute=5)
def analyze_and_send():
    """تحليل الرسم وحفظه وإرسال التقرير عبر الإيميل"""
    return common_analysis_logic(send_email=True)


@app.route('/analyze-only', methods=['POST'])
@rate_limit(max_per_minute=10)
def analyze_without_sending():
    """تحليل الرسم وحفظه فقط بدون إرسال الإيميل (للاختبار)"""
    return common_analysis_logic(send_email=False)


# -------------------------------------------------------------
#  نقطة نهاية عرض التقرير (تسحب من MongoDB)
# -------------------------------------------------------------

@app.route('/report/<string:child_id>')
def view_report(child_id):
    """
    نقطة نهاية لسحب وعرض التقرير الكامل من MongoDB.
    """
    report_content = None

    if REPORT_COLLECTION:
        try:
            # البحث عن التقرير في MongoDB
            report_doc = REPORT_COLLECTION.find_one({"child_id": child_id})

            if report_doc:
                report_content = report_doc.get("full_report_text")
            else:
                logger.warning(
                    f"⚠️ لم يتم العثور على التقرير ID: {child_id} في MongoDB.")

        except Exception as e:
            logger.error(f"خطأ في سحب التقرير من MongoDB: {e}")
            # إذا فشل الاتصال بقاعدة البيانات
            return """
            <!DOCTYPE html><html dir="rtl" lang="ar"><head><meta charset="UTF-8"><title>خطأ في الخادم</title></head>
            <body style="font-family: Arial, sans-serif; text-align: center; padding: 50px;">
            <h1>500 - خطأ في الخادم</h1>
            <p>حدث خطأ أثناء الاتصال بقاعدة البيانات. الرجاء المحاولة لاحقاً أو الاتصال بالدعم الفني.</p>
            </body></html>
            """, 500

    if report_content is None:
        # إذا لم يتم العثور على التقرير في قاعدة البيانات
        return """
        <!DOCTYPE html>
        <html dir="rtl" lang="ar"><head><meta charset="UTF-8"><title>التقرير غير موجود</title></head>
        <body style="font-family: Arial, sans-serif; text-align: center; padding: 50px;">
        <h1>404 - التقرير غير موجود</h1>
        <p>قد يكون الرابط خاطئاً، أو أن التقرير لم يتم حفظه بعد.</p>
        <p>الرجاء التواصل مع الدعم الفني إذا كنت متأكداً من صحة الرابط.</p>
        </body></html>
        """, 404

    # قالب HTML بسيط لعرض التقرير النصي
    html_template = """
    <!DOCTYPE html>
    <html dir="rtl" lang="ar">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>تقرير تحليل رسم فني - {{ child_id }}</title>
        <style>
            body { font-family: Tahoma, Arial, sans-serif; line-height: 1.6; padding: 20px; background-color: #f4f4f9; color: #333; }
            .container { max-width: 800px; margin: auto; background: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
            h1 { color: #007bff; text-align: center; border-bottom: 2px solid #007bff; padding-bottom: 10px; }
            pre { white-space: pre-wrap; font-size: 16px; border: 1px solid #eee; padding: 15px; background-color: #fff; border-radius: 5px; text-align: right; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>تقرير تحليل الرسم الفني الكامل</h1>
            <p style="text-align: center; color: #666;">كود الطفل: {{ child_id }}</p>
            <pre>{{ report_content }}</pre>
            <p style="text-align: center; margin-top: 30px;">مع تحيات فريق عمل أفق.</p>
        </div>
    </body>
    </html>
    """

    return render_template_string(html_template, report_content=report_content, child_id=child_id)

# -------------------------------------------------------------
# تشغيل التطبيق
# -------------------------------------------------------------


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
