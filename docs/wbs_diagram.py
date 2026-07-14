"""Generate a clean WBS diagram as PNG and PDF for Overleaf."""
from PIL import Image, ImageDraw, ImageFont
import os, math

# ── Canvas ────────────────────────────────────────────────────────────────────
W, H = 3200, 2000
SCALE = 2          # retina
WC, HC = W * SCALE, H * SCALE
img = Image.new("RGB", (WC, HC), "#F8FAFC")
d = ImageDraw.Draw(img)

def s(x): return int(x * SCALE)

# ── Fonts ──────────────────────────────────────────────────────────────────────
FONT_DIR = r"C:\Users\Aniket Shinde\AppData\Roaming\Claude\local-agent-mode-sessions\skills-plugin\e7a86ad0-3799-4ca6-b497-f591e4ba83f0\35755334-731c-4a9d-8577-05926aa71ba1\skills\canvas-design\canvas-fonts"

def font(name, size):
    try:
        return ImageFont.truetype(os.path.join(FONT_DIR, name), s(size))
    except:
        return ImageFont.load_default()

F_ROOT    = font("BricolageGrotesque-Bold.ttf", 22)
F_WP      = font("BricolageGrotesque-Bold.ttf", 14)
F_TASK    = font("InstrumentSans-Regular.ttf", 11)
F_NAME    = font("InstrumentSans-Italic.ttf", 9)
F_LEGEND  = font("InstrumentSans-Regular.ttf", 10)
F_LEGEND_B= font("InstrumentSans-Bold.ttf", 10)
F_TITLE   = font("BricolageGrotesque-Bold.ttf", 11)

# ── Colors ────────────────────────────────────────────────────────────────────
COLORS = {
    "Rodrigo": {"border": "#22C55E", "fill": "#F0FDF4", "text": "#14532D"},
    "Aniket":  {"border": "#0EA5E9", "fill": "#F0F9FF", "text": "#0F172A"},
    "Clara":   {"border": "#A855F7", "fill": "#FAF5FF", "text": "#581C87"},
    "Asma":    {"border": "#F59E0B", "fill": "#FFFBEB", "text": "#78350F"},
}
ROOT_FILL   = "#0F172A"
ROOT_TEXT   = "#F8FAFC"
WP_FILL     = "#1E293B"
WP_TEXT     = "#F8FAFC"
LINE_COLOR  = "#94A3B8"

# ── Data ──────────────────────────────────────────────────────────────────────
WPS = [
    {
        "label": "WP1",
        "title": "Dataset Selection\n& Schema",
        "tasks": [
            ("1.1 Review & Select\nCall Data",           "Rodrigo"),
            ("1.2 Design Grading\n& Schema Rules",       "Rodrigo"),
            ("1.3 Ingest Text\nCall Scripts",            "Aniket"),
            ("1.4 Prepare Acoustic\nEmotion Corpora",    "Clara"),
        ]
    },
    {
        "label": "WP2",
        "title": "Audio & Text\nAnalytics",
        "tasks": [
            ("2.1 Dual-Channel\nData Ingestion",         "Aniket"),
            ("2.2 Local Transcription\n& Merging",       "Aniket"),
            ("2.3 Context Eng.\n& Workflow Agent",       "Rodrigo"),
            ("2.4 Agentic Root-Cause\nSubtasks",         "Aniket"),
        ]
    },
    {
        "label": "WP3",
        "title": "Acoustic Speech\n& Sentiment",
        "tasks": [
            ("3.1 Fine-Tune\nWav2Vec2 Classifier",       "Clara"),
            ("3.2 Cross-Dataset\nEval & MLflow",         "Clara"),
            ("3.3 Implement\nText Sentiment",            "Clara"),
            ("3.4 Fuse Audio\n& Text Outputs",           "Rodrigo"),
        ]
    },
    {
        "label": "WP4",
        "title": "Backend, Rules\n& UI Dashboard",
        "tasks": [
            ("4.1 Construct\nSupabase Schemas",          "Asma"),
            ("4.2 Build Async\nFastAPI Pipelines",       "Asma"),
            ("4.3 Program Action\nRouting Logic",        "Asma"),
            ("4.4 Deploy Frontend\nUI via Vercel",       "Asma"),
        ]
    },
]

# ── Layout ────────────────────────────────────────────────────────────────────
MARGIN      = 80
ROOT_W      = 520
ROOT_H      = 72
ROOT_X      = (W - ROOT_W) // 2
ROOT_Y      = 80
WP_W        = 620
WP_H        = 68
TASK_W      = 580
TASK_H      = 88
GAP_X       = 40          # horizontal gap between WP columns
GAP_Y       = 22          # vertical gap between tasks
WP_TOP      = ROOT_Y + ROOT_H + 80
TASK_TOP    = WP_TOP + WP_H + 36
N_WPS       = len(WPS)
TOTAL_W     = N_WPS * WP_W + (N_WPS - 1) * GAP_X
START_X     = (W - TOTAL_W) // 2
RADIUS      = 10

def rounded_rect(draw, x1, y1, x2, y2, r, fill, outline, width=2):
    x1, y1, x2, y2, r = s(x1), s(y1), s(x2), s(y2), s(r)
    draw.rounded_rectangle([x1, y1, x2, y2], radius=r, fill=fill, outline=outline, width=width*SCALE)

def centered_text(draw, cx, cy, text, font, fill, line_spacing=1.25):
    lines = text.split("\n")
    total_h = sum(font.getbbox(l)[3] - font.getbbox(l)[1] for l in lines)
    total_h += int((len(lines) - 1) * font.getbbox("A")[3] * (line_spacing - 1))
    y = s(cy) - total_h // 2
    for line in lines:
        bb = font.getbbox(line)
        lw = bb[2] - bb[0]
        draw.text((s(cx) - lw // 2, y), line, font=font, fill=fill)
        y += int((bb[3] - bb[1]) * line_spacing)

# ── Draw root ─────────────────────────────────────────────────────────────────
rounded_rect(d, ROOT_X, ROOT_Y, ROOT_X + ROOT_W, ROOT_Y + ROOT_H, RADIUS,
             ROOT_FILL, "#38BDF8", width=2)
centered_text(d, ROOT_X + ROOT_W // 2, ROOT_Y + ROOT_H // 2,
              "Call Center Agent QA System", F_ROOT, ROOT_TEXT)

root_cx = ROOT_X + ROOT_W // 2
root_bot = ROOT_Y + ROOT_H

# ── Draw WPs and tasks ────────────────────────────────────────────────────────
for wi, wp in enumerate(WPS):
    wp_x = START_X + wi * (WP_W + GAP_X)
    wp_cx = wp_x + WP_W // 2

    # connector: root → WP header
    wp_top_mid_y = WP_TOP
    mid_y = (root_bot + WP_TOP) // 2
    d.line([s(root_cx), s(root_bot), s(root_cx), s(mid_y)],
           fill=LINE_COLOR, width=s(2))
    d.line([s(root_cx), s(mid_y), s(wp_cx), s(mid_y)],
           fill=LINE_COLOR, width=s(2))
    d.line([s(wp_cx), s(mid_y), s(wp_cx), s(WP_TOP)],
           fill=LINE_COLOR, width=s(2))

    # WP header box
    rounded_rect(d, wp_x, WP_TOP, wp_x + WP_W, WP_TOP + WP_H, RADIUS,
                 WP_FILL, "#475569", width=2)
    label_text = f"{wp['label']}: {wp['title']}"
    centered_text(d, wp_cx, WP_TOP + WP_H // 2, label_text, F_WP, WP_TEXT)

    # tasks
    prev_bot = WP_TOP + WP_H
    task_x = wp_x + (WP_W - TASK_W) // 2
    for ti, (task_text, owner) in enumerate(wp["tasks"]):
        c = COLORS[owner]
        ty = TASK_TOP + ti * (TASK_H + GAP_Y)
        task_cx = task_x + TASK_W // 2

        # connector line WP/prev task → this task
        connector_from = prev_bot
        d.line([s(task_cx), s(connector_from), s(task_cx), s(ty)],
               fill=LINE_COLOR, width=s(1))
        prev_bot = ty + TASK_H

        # task box
        rounded_rect(d, task_x, ty, task_x + TASK_W, ty + TASK_H, 8,
                     c["fill"], c["border"], width=2)

        # task text + owner name
        lines = task_text.split("\n")
        total_lines_h = len(lines) * s(13) + s(14)
        text_start_y = s(ty) + (s(TASK_H) - total_lines_h) // 2
        for li, line in enumerate(lines):
            bb = F_TASK.getbbox(line)
            lw = bb[2] - bb[0]
            d.text((s(task_cx) - lw // 2, text_start_y + li * s(13)),
                   line, font=F_TASK, fill=c["text"])
        # owner name
        owner_names = {"Rodrigo": "Rodrigo Mayorga", "Aniket": "Aniket Shinde",
                       "Clara": "Clara Yousif", "Asma": "Asma Haneef"}
        name_str = owner_names[owner]
        bb = F_NAME.getbbox(name_str)
        nw = bb[2] - bb[0]
        d.text((s(task_cx) - nw // 2, text_start_y + len(lines) * s(13) + s(2)),
               name_str, font=F_NAME, fill=c["border"])

# ── Legend ────────────────────────────────────────────────────────────────────
legend_items = [
    ("Rodrigo Mayorga", "Rodrigo"),
    ("Aniket Shinde",   "Aniket"),
    ("Clara Yousif",    "Clara"),
    ("Asma Haneef",     "Asma"),
]
lx = MARGIN
ly = H - 110
lw_box = 16; lh_box = 16; lgap = 10; litem_w = 200

d.text((s(lx), s(ly - 22)), "TEAM LEGEND", font=F_TITLE, fill="#475569")
for i, (name, key) in enumerate(legend_items):
    c = COLORS[key]
    bx = lx + i * litem_w
    rounded_rect(d, bx, ly, bx + lw_box, ly + lh_box, 3,
                 c["fill"], c["border"], width=2)
    d.text((s(bx + lw_box + lgap), s(ly + 1)), name, font=F_LEGEND, fill="#1E293B")

# ── Save ──────────────────────────────────────────────────────────────────────
out_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(out_dir, exist_ok=True)
out_png = os.path.join(out_dir, "wbs_diagram.png")
img_final = img.resize((W, H), Image.LANCZOS)
img_final.save(out_png, dpi=(300, 300))
print(f"Saved PNG: {out_png}")

# PDF via pillow
out_pdf = os.path.join(out_dir, "wbs_diagram.pdf")
img_final.save(out_pdf, "PDF", resolution=300)
print(f"Saved PDF: {out_pdf}")
