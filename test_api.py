import requests
import os
import json
import time

# ----------------------------------------------------
# الإعدادات
# ----------------------------------------------------

# **** ضعي مسار ملف صورتكِ هنا ****
# يجب أن يكون المسار صحيحًا، واستخدام r"" يسهل التعامل مع مسارات ويندوز.
IMAGE_FILE_NAME = r"C:\Users\ACER\Desktop\ai_analysis_api\test_drawing.jpg"

# نستخدم analyze لاختبار وظيفة الإيميل
API_URL = "http://127.0.0.1:5000/analyze"

# 1. إعداد بيانات الحقول النصية (form-data)
data_fields = {
    "child_id": "Email_Test_003",
    "child_name": "ريم",
    "child_age": "7",
    "parent_email": "ahmedalgarably49@gmail.com"
}

# ----------------------------------------------------
# منطق التشغيل
# ----------------------------------------------------

# 2. إعداد ملف الصورة (File)
try:
    with open(IMAGE_FILE_NAME, 'rb') as f:
        # تأكدي من نوع MIME الصحيح (.jpg تكون image/jpeg)
        files = {'image': (os.path.basename(
            IMAGE_FILE_NAME), f.read(), 'image/jpeg')}
except FileNotFoundError:
    print(f"❌ خطأ: لم يتم العثور على ملف الصورة: {IMAGE_FILE_NAME}")
    print("الرجاء التأكد من اسم الملف وموقعه.")
    exit()

# 3. إرسال الطلب POST
print(f"🚀 إرسال طلب POST إلى {API_URL}...")
start_time = time.time()
try:
    response = requests.post(
        API_URL,
        data=data_fields,
        files=files,
        timeout=30
    )
    end_time = time.time()

    # 4. معالجة الاستجابة (الجزء المصحح)
    print("-" * 40)
    print(f"✅ حالة الاستجابة (Status Code): {response.status_code}")
    print(f"⏱️ زمن التحليل: {end_time - start_time:.2f} ثانية")
    print("-" * 40)

    if response.status_code == 200:
        response_json = response.json()

        # عند استخدام /analyze، تكون الاستجابة مختصرة (تحتوي على email_status و report_summary)
        if 'email_status' in response_json:
            print("🎉 تم التحليل بنجاح.")
            print(f"📧 حالة الإيميل: **{response_json['email_status']}**")
            print("يرجى التحقق من صندوق الوارد (و مجلد الـ Spam) في بريدك الإلكتروني.")

            # طباعة ملخص التقرير المرفق في الرد
            if 'report_summary' in response_json:
                print("\n🌟 ملخص التقرير:")
                for item in response_json['report_summary']:
                    print(f"- {item}")

        # في حال تم تغيير الرابط لـ /analyze-only (للتأكد)
        elif 'analysis_data' in response_json:
            print("⚠️ تم استقبال رد /analyze-only. الكود يعمل لقراءة الرد الكامل.")
            # يمكنك إضافة عرض مفصل للبيانات هنا

    else:
        print(f"❌ حدث خطأ في الخادم (Status: {response.status_code})")
        try:
            # محاولة قراءة رسالة الخطأ من الخادم
            print("رسالة الخطأ:", response.json().get("message", response.text))
        except:
            print("الرد غير JSON:", response.text)

except requests.exceptions.ConnectionError:
    print(f"❌ خطأ: فشل الاتصال بالخادم. هل الخادم يعمل على {API_URL}؟")
except requests.exceptions.Timeout:
    print("❌ خطأ: انتهت مهلة الطلب (Timeout). قد يكون التحليل استغرق وقتاً طويلاً.")
except Exception as e:
    print(f"❌ خطأ غير متوقع: {e}")
