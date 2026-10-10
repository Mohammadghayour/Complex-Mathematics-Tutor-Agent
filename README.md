# Complex Mathematics Tutor Agent

یک Agent آموزشی ریاضی مبتنی بر **Graph** که سؤال یا مفهوم ریاضی را از طریق **CLI** دریافت می‌کند،
نوع درخواست را تشخیص می‌دهد، مسیر مناسب را در Graph انتخاب می‌کند، پاسخ را **قبل از نمایش بررسی
می‌کند** و در نهایت پاسخی ساده، دقیق، مرحله‌به‌مرحله و متناسب با سطح کاربر ارائه می‌دهد.

هیچ رابط گرافیکی وجود ندارد؛ تمام تعامل از طریق ترمینال انجام می‌شود.

---

## ۱. فهرست فایل‌ها

```
complex_math_tutor/
├── agent.py              منطق اصلی Agent: State، Nodeها، مسیرها (Edges) و اجرای Graph
├── cli.py                رابط خط فرمان: دریافت ورودی، نمایش خروجی، دستورها، argparse
├── config.py             تنظیمات از محیط / فایل .env (+ اعتبارسنجی تصویر)
├── llm_provider.py       انتزاع provider: مدل واقعی (OpenAI-compatible) + fallback آفلاین
├── heuristics.py         موتور ریاضی آفلاین (تجزیه، حل، و بازبینی مستقل پاسخ)
├── prompts.py            تمام promptهای داخلی (جدا از منطق Graph)
├── requirements.txt      وابستگی‌ها
├── .env.example          نمونه فایل تنظیمات (هیچ API key واقعی در کد نیست)
├── pytest.ini            تنظیمات تست
├── README.md             همین فایل
├── images/
│   └── .gitkeep          پوشه پیشنهادی برای قرار دادن تصویر سؤال
└── tests/
    ├── conftest.py           fixtureهای مشترک (کاملاً آفلاین)
    ├── fakes.py              providerهای تقلبی (mock) برای تست بدون API key
    ├── test_agent.py         تست State، Nodeها، مسیرها، طبقه‌بندی و بازبینی پاسخ
    ├── test_cli.py           تست CLI در حالت mock
    └── test_image_input.py   تست مسیر تصویر
```

**نقش فایل‌ها در یک نگاه**

| فایل | مسئولیت |
|---|---|
| `agent.py` | `TutorState`، `AgentDeps`، ۱۱ Node، تابع مسیریابی `next_node`، کلاس `TutorAgent` و دو backend اجرا (LangGraph / builtin) |
| `cli.py` | فقط ورودی/خروجی: حلقه تعاملی، دستورها، `argparse`، نمایش خطا — هیچ منطق Agent اینجا نیست |
| `heuristics.py` | ریاضی واقعی: نرمال‌سازی متن، استخراج عبارت، چندجمله‌ای، مشتق، حل معادله، و `verify_answer` |
| `llm_provider.py` | `DeterministicProvider` (آفلاین) و `OpenAICompatibleProvider` (شبکه) + `build_provider` |
| `prompts.py` | همه promptها؛ منطق Graph هیچ متن طبیعی ندارد |
| `config.py` | خواندن `.env`/محیط، محدودیت‌های Graph، مجوز فرمت تصویر |

---

## ۲. نصب

نیازمندی: **Python 3.10+** (با 3.13 تست شده است).

```bash
cd complex_math_tutor

# (اختیاری ولی پیشنهادی) محیط مجازی
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows (Git Bash)
source .venv/Scripts/activate

pip install -r requirements.txt
```

> **مهم:** پروژه با **کتابخانه استاندارد پایتون هم کار می‌کند**.
> اگر `pip install` انجام نشود، از runner داخلی استفاده می‌شود و همه تست‌ها باز هم پاس می‌شوند.
> نصب `langgraph` باعث می‌شود همان Nodeها در یک `StateGraph` واقعی ثبت و اجرا شوند.

---

## ۳. اجرا

```bash
# حالت تعاملی
python cli.py

# با لاگ تشخیصی و نمایش مسیر اجرای Graph
python cli.py --debug

# یک سؤال و خروج
python cli.py --once "مشتق تابع x^2 + 3x را توضیح بده"

# تحلیل تصویر سؤال
python cli.py --image images/math_question.png

# انتخاب provider / مدل
python cli.py --provider openai --model gpt-4o-mini
```

### نمونه تعامل

```
$ python cli.py

=========================================================
  Complex Mathematics Tutor Agent  (v1.0.0)
=========================================================
  Type your mathematics question and press Enter.
  Commands: exit | quit | help | reset | image | level
  Backend : langgraph
  Provider: deterministic / offline engine
  Vision  : disabled (no vision model)
  Images  : .../complex_math_tutor/images
=========================================================

You: مشتق تابع x^2 + 3x را توضیح بده

نوع درخواست: مفهومی + محاسباتی
موضوع: مشتق
سطح تقریبی: مبتدی

تعریف ساده:
از پایه و بدون فرض دانش پیشرفته شروع می‌کنیم. مشتق، آهنگ تغییر یک تابع نسبت به متغیر آن است؛ ...

توضیح شهودی:
تصور کن در حال رانندگی هستی: مکان تو تابع زمان است و مشتق آن، سرعت لحظه‌ای تو. ...

مثال:
برای f(x) = x^2 + 3x، مشتق برابر است با f'(x) = 2x + 3.

نکات مهم:
قواعد پایه: مشتق x^n برابر n·x^(n-1)، مشتق عدد ثابت صفر، و مشتق مجموع برابر مجموع مشتق‌هاست.

تحلیل مسئله:
- صورت مسئله / problem: x^2+3*x
- خواسته / asked: مقدار نهایی
- روش حل / method: مشتق‌گیری با قاعده توان

مراحل حل:
1. عبارت داده‌شده: f(x) = x^2 + 3x
2. اعمال قاعده توان روی هر جمله: d/dx a·x^n = a·n·x^(n-1)
3. نتیجه: f'(x) = 2x + 3

پاسخ نهایی:
2x + 3

بررسی پاسخ:
- بررسی ساختاری: پاسخ یک عبارت جبری است؛ گام‌ها منسجم‌اند ...
- بررسی مستقل مشتق با تفاضل عددی در x = 1.3: 5.6 ≈ 5.6 / derivative check passed.
- نتیجه بررسی / verification: درست / correct

You: اگر x=2 باشد، مقدار 3x^2 + 1 چقدر است؟

نوع درخواست: مسئله قابل حل
موضوع: محاسبات
سطح تقریبی: مبتدی

تحلیل مسئله:
- صورت مسئله / problem: 3*x^2+1
- داده‌ها / given: x = 2
- خواسته / asked: مقدار نهایی
- روش حل / method: جایگذاری مستقیم

مراحل حل:
1. جایگذاری مقادیر داده‌شده: 3*(2)^2+1
2. ساده‌سازی عبارت: 13
3. محاسبه مقدار نهایی: 13

پاسخ نهایی:
13

بررسی پاسخ:
- محاسبه مجدد مستقل: 3*x^2+1 با {'x': 2.0} = 13 (مطابق با پاسخ 13)
- تمام گام‌های عددی با محاسبه مستقل سازگارند / all numeric steps match.
- نتیجه بررسی / verification: درست / correct
```

### نمونه حالت تصویر

```
You: image
Image path: images/math_question.png
تصویر دریافت شد و در حال تحلیل است... (/abs/path/images/math_question.png)
سؤال متنی (اختیاری، Enter برای رد کردن):

Agent:
...
```

اگر مدل بینایی در دسترس نباشد:

```
[هشدار / warning] No vision model is configured, so the image cannot be analysed.
Set OPENAI_API_KEY (and optionally MATH_TUTOR_VISION_MODEL) to enable image support,
or type the problem as text.
```

### دستورهای داخل CLI

| دستور | کار |
|---|---|
| `exit` / `quit` / `:q` | خروج (Ctrl+C و Ctrl+D هم کار می‌کنند) |
| `help` | راهنما |
| `reset` | پاک کردن زمینه گفتگو |
| `image` | دادن مسیر تصویر سؤال |
| `level beginner\|intermediate\|advanced\|auto` | تعیین دستی سطح کاربر |
| `<سؤال>` | پرسیدن سؤال ریاضی |

### آرگومان‌های CLI

| آرگومان | توضیح |
|---|---|
| `--model` | نام مدل زبانی (روی `MATH_TUTOR_MODEL` اولویت دارد) |
| `--provider` | `auto` \| `openai` \| `openai_compatible` \| `ollama` \| `deterministic` |
| `--image` | مسیر تصویر سؤال (PNG/JPG/JPEG/WEBP/GIF/BMP) |
| `--debug` | نمایش مسیر اجرای Graph + لاگ کامل |
| `--once "..."` | پاسخ به یک سؤال و خروج |
| `--env` | مسیر فایل `.env` |
| `--no-model-review` | حذف بازبینی اضافی مدل (بازبینی قطعی همیشه اجرا می‌شود) |

---

## ۴. معماری

### ۴.۱ State مرکزی

`TutorState` یک `TypedDict` است و در طول اجرا **واقعاً** به‌روزرسانی می‌شود
(هر Node فقط کلیدهایی که تغییر داده را برمی‌گرداند).

| گروه | کلیدها |
|---|---|
| ورودی | `original_query`, `normalized_query`, `pending_user_input`, `image_path`, `image_analysis`, `image_processed` |
| طبقه‌بندی | `request_type`, `problem_statement`, `subject_or_topic`, `user_level`, `language`, `is_problem_clear`, `missing_information`, `clarification_question`, `route_decision`, `classification_meta` |
| کار در جریان | `collected_context`, `solution_steps`, `candidate_answer`, `checked_answer`, `is_answer_correct`, `verification_notes`, `verification_method`, `solution_meta`, `concept_sections` |
| خروجی | `explanation`, `final_answer`, `error_message` |
| دفترداری | `iteration_count`, `solve_attempts`, `awaiting_input`, `path_log`, `last_node`, `conversation_history` |

`request_type` یکی از: `conceptual` / `solvable_problem` / `mixed` / `unclear`
`user_level` یکی از: `beginner` / `intermediate` / `advanced` / `unknown` (پیش‌فرض: `beginner`)

### ۴.۲ Nodeها

| Node | کار |
|---|---|
| `read_problem` | دریافت ورودی، نرمال‌سازی (ارقام فارسی/عربی، فاصله‌ها)، ثبت در history، استخراج اولیه topic/level |
| `inspect_image` | اعتبارسنجی فایل تصویر و ارسال به مدل بینایی (یا خطای قابل‌فهم) |
| `classify_request` | تشخیص نوع درخواست، موضوع، سطح کاربر، صورت مسئله، نیاز به توضیح |
| `check_problem_clarity` | **تصمیم واقعی Graph**: کافی بودن اطلاعات و نوشتن `route_decision` |
| `ask_question` | تولید سؤال تکمیلی کوتاه، دریافت پاسخ کاربر، بازگشت به `read_problem` با سقف تکرار |
| `explain_concept` | توضیح مفهوم متناسب با سطح (تعریف/شهود/مثال/نکات) |
| `solve_problem` | تحلیل مسئله، انتخاب روش، تولید مراحل و پاسخ نامزد |
| `check_answer` | **بازبینی مستقل**: محاسبه مجدد با Python، بررسی سازگاری گام‌ها، جایگذاری ریشه‌ها، تفاضل عددی مشتق، و در صورت وجود مدل یک بازبینی جداگانه |
| `repair_solution` | در صورت غلط بودن پاسخ، حل مجدد با بازخورد بازبین |
| `format_final_answer` | قالب‌بندی نهایی بر اساس نوع درخواست و سطح کاربر |
| `final_answer` | خاتمه Graph و تضمین وجود `final_answer` |

### ۴.۳ مسیرهای Graph

```
                         ┌─ (unclear / ناقص) ─→ ask_question ─┐
                         │            ↑                        │
Start → read_problem → classify_request → check_problem_clarity │
           ↑  ↑                              │ (clear)          │
           │  └── inspect_image ←────────────┘                  │
           └───────────────────────────────────────────────────-┘

مسیر مفهومی:
  read_problem → classify_request → check_problem_clarity → explain_concept
               → format_final_answer → final_answer

مسیر مسئله قابل حل:
  read_problem → classify_request → check_problem_clarity → solve_problem
               → check_answer → format_final_answer → final_answer

مسیر ترکیبی (مفهومی + محاسباتی):
  ... → explain_concept → solve_problem → check_answer → format_final_answer → final_answer

مسیر خطای پاسخ:
  solve_problem → check_answer → (غلط) → repair_solution → check_answer
               → (درست یا پایان سهم تلاش) → format_final_answer → final_answer

مسیر تصویر:
  read_problem → inspect_image → classify_request → ...
```

`path_log` در State همیشه مسیر واقعی طی‌شده را نگه می‌دارد؛ با `--debug` نمایش داده می‌شود.

### ۴.۴ دو Backend اجرا

| Backend | چه زمانی | توضیح |
|---|---|---|
| `langgraph` | وقتی `langgraph` نصب باشد | همان Nodeها در `StateGraph` ثبت می‌شوند؛ هر Node یک conditional edge با تابع `next_node` دارد |
| `builtin` | وقتی نصب نباشد | یک walker روی همان `NODES` و همان `next_node` — رفتار یکسان، بدون وابستگی |

هر دو مسیر از **یک منبع حقیقت** برای مسیریابی استفاده می‌کنند: `agent.next_node(current, state, deps)`.

---

## ۵. مدل زبانی و Fallback آفلاین

```
MATH_TUTOR_PROVIDER=auto  →  اگر OPENAI_API_KEY وجود داشت: مدل واقعی
                          →  در غیر این صورت: موتور قطعی آفلاین
```

- **`OpenAICompatibleProvider`**: با `urllib` استاندارد به `/chat/completions` وصل می‌شود
  (OpenAI، Groq، Together، OpenRouter، vLLM، Ollama Gateway…). تنها providerی که تصویر را می‌پذیرد.
- **`DeterministicProvider`**: بدون شبکه و بدون API key. کلاسیفای و حل و بازبینی **واقعی** انجام می‌دهد
  (استخراج عبارت، جایگذاری، حل معادله درجه ۱/۲، مشتق با قاعده توان) — به همین دلیل Graph و CLI و تست‌ها
  بدون هیچ کلیدی کار می‌کنند.

هیچ API key داخل کد نیست؛ همه‌چیز از `.env` یا متغیرهای محیطی خوانده می‌شود:

```bash
cp .env.example .env
# سپس OPENAI_API_KEY را در .env پر کنید
```

متغیرهای مهم: `MATH_TUTOR_PROVIDER`، `MATH_TUTOR_MODEL`، `MATH_TUTOR_VISION_MODEL`،
`OPENAI_API_KEY`، `OPENAI_BASE_URL`، `MATH_TUTOR_MAX_CLARIFICATIONS`،
`MATH_TUTOR_MAX_SOLVE_ATTEMPTS`، `MATH_TUTOR_MODEL_REVIEW`، `MATH_TUTOR_IMAGES_DIR`.

---

## ۶. پشتیبانی از تصویر

1. فایل تصویر را در پوشه `images/` پروژه (یا هر مسیر دیگر) قرار دهید.
2. در CLI بنویسید `image` و سپس مسیر را وارد کنید — یا مستقیم `--image images/q.png`.
3. **مسیر نسبی و مطلق هر دو پذیرفته می‌شود** (نسبت به cwd، ریشه پروژه و `images/` جست‌وجو می‌شود).
4. اعتبارسنجی انجام می‌شود: وجود فایل، پسوند مجاز، حجم، خالی نبودن، و تطابق محتوای فایل با پسوند
   (magic number بررسی می‌شود).
5. اگر مدل بینایی پیکربندی شده باشد، Node `inspect_image` تصویر را به مدل می‌فرستد و متن مسئله را
   استخراج می‌کند؛ در غیر این صورت پیام خطای قابل‌فهم نمایش داده می‌شود و کاربر می‌تواند مسئله را تایپ کند.

> کاربر **هرگز** نیازی به تبدیل تصویر به Base64 ندارد؛ این کار درون `llm_provider.inspect_image` انجام می‌شود.

فرمت‌های پشتیبانی‌شده: `.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`, `.bmp` — حداکثر ۲۰ مگابایت.

---

## ۷. تست

همه تست‌ها **آفلاین** هستند (provider قطعی یا fake)، پس بدون API key و بدون شبکه اجرا می‌شوند:

```bash
cd complex_math_tutor
python -m pytest -v
```

پوشش تست‌ها:

| فایل | موارد |
|---|---|
| `tests/test_agent.py` | وجود همه Nodeها، مسیریابی واقعی، محدود بودن حلقه clarification، تشخیص مفهومی/قابل‌حل/ترکیبی/نامشخص، ask_question، به‌روزرسانی State، تصمیم `check_problem_clarity`، مسیر `check_answer` و `repair_solution`، بازبینی مستقل عددی/معادله/مشتق، خطای provider بدون crash |
| `tests/test_cli.py` | argparse، ادغام config، `exit/quit/:q`، EOF، help، reset، level، دستور ناشناخته، حالت `--once`، اجرای CLI در حالت mock |
| `tests/test_image_input.py` | مسیر مطلق/نسبی، پسوند نامعتبر، فایل خالی، حجم بیش از حد، عدم تطابق محتوا، خطای نبود مدل بینایی، مسیر `inspect_image` |

نتیجه واقعی اجرا:

```
77 passed
```

---

## ۸. توسعه‌پذیری

- **افزودن provider جدید**: یک کلاس با متدهای `classify/clarify/explain/solve/review/inspect_image`
  بنویسید و در `build_provider` ثبت کنید (نمونه: `tests/fakes.py`).
- **افزودن Node جدید**: تابعی با امضای `(state, deps) -> dict` بنویسید، به `NODES` اضافه کنید و
  مسیر آن را در `next_node` مشخص کنید. برای LangGraph خودکار ثبت می‌شود.
- **تغییر promptها**: فقط `prompts.py`.
- **تنظیم محدودیت‌ها**: `MATH_TUTOR_MAX_CLARIFICATIONS` و `MATH_TUTOR_MAX_SOLVE_ATTEMPTS`
  (محافظت از حلقه بی‌نهایت) — `MAX_TOTAL_STEPS` نیز سقف کل Nodeهای هر اجرا را می‌بندد.

---

## ۹. عیب‌یابی

| مشکل | راه‌حل |
|---|---|
| «LangGraph نصب نیست» | `pip install -r requirements.txt` — یا نادیده بگیرید؛ runner داخلی همان Graph را اجرا می‌کند |
| «مدل بینایی در دسترس نیست» | `OPENAI_API_KEY` و در صورت نیاز `MATH_TUTOR_VISION_MODEL` را تنظیم کنید |
| «تصویر پیدا نشد» | فایل را در `images/` بگذارید و مسیر نسبی یا مطلق آن را بدهید |
| پاسخ‌ها همیشه فارسی است | زبان به‌طور خودکار تشخیص داده می‌شود؛ متن ریاضی خالص (مثل `2x + 3 = 7`) فارسی پاسخ داده می‌شود |
| می‌خواهم ببینم Graph چه مسیری رفت | `python cli.py --debug` و به خط «مسیر اجرای گراف» نگاه کنید |
| لاگ کامل | `complex_math_tutor/logs/agent.log` (چرخشی) |

---

## ۱۰. معیارهای پذیرش — وضعیت

| معیار | وضعیت |
|---|---|
| Agent فقط یک prompt ساده نیست | ✅ Graph و Nodeهای واقعی در `agent.py` |
| Graph و Nodeهای قابل مشاهده و واقعی | ✅ ۱۱ Node + LangGraph/`builtin` + `path_log` |
| State در طول اجرا تغییر می‌کند | ✅ هر Node فقط کلیدهای تغییریافته را برمی‌گرداند؛ تست `test_state_is_updated_throughout_the_run` |
| مسیر اجرا بر اساس نوع سؤال تغییر می‌کند | ✅ `next_node` + تست‌های مسیر |
| سؤال مفهومی و محاسباتی رفتار متفاوت دارند | ✅ مسیر `explain_concept` در برابر `solve_problem` |
| سؤال ناقص باعث `ask_question` می‌شود | ✅ با سقف تکرار و خاتمه تضمین‌شده |
| پاسخ قبل از نمایش بررسی می‌شود | ✅ `check_answer` + `verify_answer` (محاسبه مجدد، ریشه‌ها، تفاضل عددی) |
| CLI قابل اجرا | ✅ `python cli.py` |
| تست‌های mock بدون API key اجرا می‌شوند | ✅ `77 passed` |
| مسیر تصویر پشتیبانی می‌شود | ✅ `inspect_image` + اعتبارسنجی کامل |
| پاسخ نهایی متناسب با سطح کاربر | ✅ `user_level` در همه Nodeهای تولید متن |
