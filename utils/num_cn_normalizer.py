# -*- coding: utf-8 -*-
"""
num_cn_normalizer.py (2025-08-16 fix)

变更要点：
1) 数字区间先处理，再处理“负号+单位”，修复 3500-4000 mL 被误解读为“负四千毫升”的问题。
2) 仅保护“字母在前或字母夹数字”的混合 token；对“数字开头+英文单位”(如 15kg, 512GB) 不保护，走“数字+单位”正则。
3) 一律不使用“两”，读“二”；年份逐位读（2024年 -> 二零二四年）。
4) 继续保护 USB3.0 / H2O / v2 / A1-2B / C-130 / A2024 等。
"""

import re
from typing import Dict, Tuple, List

DIGITS = "零一二三四五六七八九"
UNITS_SMALL = ["", "十", "百", "千"]
UNITS_BIG = ["", "万", "亿", "兆"]

UNIT_MAP: Dict[str, str] = {
    # 温度/角度
    "℃": "摄氏度", "°c": "摄氏度", "°C": "摄氏度", "°": "度",
    # 速度
    "m/s": "米每秒", "km/h": "千米每小时", "km/s": "千米每秒",
    # 速度（英文与中文写法都支持）
    
    "米/秒": "米每秒",      # ← 新增：中文写法
    "千米/小时": "千米每小时", # （可选）中文写法
    "千米/秒": "千米每秒",    # （可选）中文写法
    "公里/小时": "千米每小时", # （可选）同义：公里视为千米
    "公里/秒": "千米每秒",    # （可选）
    # 体积
    "ml": "毫升", "mL": "毫升", "l": "升", "L": "升",
    # 长度
    "km": "千米", "m": "米", "cm": "厘米", "mm": "毫米", "um": "微米", "μm": "微米",
    # 质量
    "kg": "千克", "g": "克", "mg": "毫克", "t": "吨",
    # 频率
    "hz": "赫兹", "khz": "千赫兹", "mhz": "兆赫兹", "ghz": "吉赫兹",
    # 电学/存储
    "v": "伏", "a": "安", "w": "瓦",
    "kb": "千比", "mb": "兆比", "gb": "吉比",
}

MEASURE_WORDS = {
    # 时间/日期
    "年","月","日","号","周","天","小时","时","分钟","分","秒","岁",
    # 数量/等级
    "级","层","人","台","件","个","位","次","条","只","张","本","辆","处","口","倍","页",
    # 距离/体积/质量
    "米","千米","公里","厘米","毫米","升","毫升","千克","克","吨","寸","英寸",
    # 货币/杂项
    "元","块","美元","欧元","日元","韩元","卢比","分辨率"
}

RE_NUM = r"\d+(?:\.\d+)?"
RE_PERCENT = re.compile(rf"(?P<num>{RE_NUM})\s*%")

def _make_unit_regex(keys: List[str]) -> re.Pattern:
    keys = sorted(keys, key=len, reverse=True)
    alt = "|".join(re.escape(k) for k in keys)
    return re.compile(rf"(?P<num>{RE_NUM})\s*(?P<unit>(?:{alt}))", re.IGNORECASE)

RE_NUM_UNIT = _make_unit_regex(list(UNIT_MAP.keys()))

# “纯数字”（避免紧挨字母/百分号）
RE_PURE_NUM = re.compile(rf"(?<![A-Za-z0-9_])(?P<num>{RE_NUM})(?![A-Za-z%])")
# 纯负数
RE_NEG_PURE = re.compile(rf"(?<![A-Za-z0-9_])-(?P<num>{RE_NUM})(?![A-Za-z%])")

def _make_signed_unit_regex(keys: List[str]) -> re.Pattern:
    keys = sorted(keys, key=len, reverse=True)
    alt = "|".join(re.escape(k) for k in keys)
    return re.compile(rf"(?P<sign>[-−])\s*(?P<num>{RE_NUM})\s*(?P<unit>(?:{alt}))",
                      re.IGNORECASE)
RE_SIGNED_NUM_UNIT = _make_signed_unit_regex(list(UNIT_MAP.keys()))

# 数字区间（两侧不能紧邻英文字母）
RE_RANGE = re.compile(
    rf"(?<![A-Za-z])(?P<a>{RE_NUM})\s*[-~—–]\s*(?P<b>{RE_NUM})(?![A-Za-z])"
)

# 日期/时间
RE_YMD_DASH = re.compile(r"\b(?P<y>\d{4})-(?P<m>\d{1,2})-(?P<d>\d{1,2})\b")
RE_YMD_CN   = re.compile(r"(?<![A-Za-z])(?P<y>\d{4})年(?P<m>\d{1,2})月(?P<d>\d{1,2})日")
RE_MD_CN    = re.compile(r"(?<![A-Za-z])(?P<m>\d{1,2})月(?P<d>\d{1,2})日")
RE_HM_CN    = re.compile(r"(?<![A-Za-z])(?P<h>\d{1,2})时(?P<m>\d{1,2})分")

# “字母+数字混合”token（允许 . _ -），用于保护
RE_ALNUM_TOKEN = re.compile(r"(?i)\b[0-9A-Za-z]+(?:[._-][0-9A-Za-z]+)*\b")

# ---------------- 数字转中文 ----------------
def digits_to_chinese(s: str) -> str:
    out = []
    if s.startswith("-"):
        out.append("负"); s = s[1:]
    for ch in s:
        if ch.isdigit(): out.append(DIGITS[ord(ch)-48])
        elif ch == ".":  out.append("点")
        else:            out.append(ch)
    return "".join(out)

def _four_digit_to_chinese(n: int) -> str:
    assert 0 <= n <= 9999
    if n == 0: return "零"
    s, zero_flag, started = [], False, False
    digits = [int(d) for d in f"{n:04d}"]  # 千百十个
    for i, d in enumerate(digits):
        pos = 3 - i
        if d == 0:
            if started: zero_flag = True
            continue
        if zero_flag and started:
            s.append("零"); zero_flag = False
        started = True
        s.append(DIGITS[d])          # 2 -> “二”
        s.append(UNITS_SMALL[pos])
    return "".join(s).rstrip("零")

def int_to_chinese(n: int) -> str:
    if n == 0: return DIGITS[0]
    if n < 0:  return "负" + int_to_chinese(-n)
    parts, idx = [], 0
    while n > 0:
        seg = n % 10000
        if seg != 0:
            seg_str = _four_digit_to_chinese(seg)
            if idx > 0: seg_str += UNITS_BIG[idx]
            parts.append(seg_str)
        else:
            parts.append("")
        n //= 10000; idx += 1
    raw = "".join(reversed(parts))
    while "零零" in raw: raw = raw.replace("零零","零")
    raw = raw.rstrip("零")
    raw = re.sub(r"^一十", "十", raw)
    return raw

def number_to_chinese(num_str: str) -> str:
    if num_str.startswith("-"):
        return "负" + number_to_chinese(num_str[1:])
    if "." in num_str:
        a, b = num_str.split(".", 1)
        left  = int_to_chinese(int(a))
        right = "".join(DIGITS[ord(c)-48] for c in b if c.isdigit())
        return left + "点" + right
    return int_to_chinese(int(num_str))

# -------------- 保护/还原混合 token --------------
def _should_protect_tok(tok: str) -> bool:
    """只保护 '字母在前/夹数字' 的混合 token；'数字开头+字母'(如 15kg, 512GB) 不保护。"""
    if re.match(r'^\d+(?:\.\d+)?[A-Za-z]+$', tok):
        return False  # 让 15kg / 512GB 去走“数字+单位”规则
    has_alpha = re.search(r'[A-Za-z]', tok) is not None
    has_digit = re.search(r'\d', tok) is not None
    return has_alpha and has_digit  # 如 USB3.0 / H2O / A1-2B / C-130 / A2024

def _protect_alnum_tokens(text: str) -> Tuple[str, Dict[str, str]]:
    mapping, idx = {}, 0
    def repl(m: re.Match) -> str:
        nonlocal idx
        tok = m.group(0)
        if _should_protect_tok(tok):
            key = f"<<<TOK{idx}>>>"
            mapping[key] = tok; idx += 1
            return key
        return tok
    return RE_ALNUM_TOKEN.sub(repl, text), mapping

def _restore_tokens(text: str, mapping: Dict[str,str]) -> str:
    for k, v in mapping.items():
        text = text.replace(k, v)
    return text

# -------------- 规则应用顺序（很重要）--------------
def _normalize_dates(text: str) -> str:
    def rep_dash(m: re.Match) -> str:
        y = m.group("y"); mm = int(m.group("m")); dd = int(m.group("d"))
        return f"{digits_to_chinese(y)}年{int_to_chinese(mm)}月{int_to_chinese(dd)}日"
    text = RE_YMD_DASH.sub(rep_dash, text)

    def rep_ymd(m: re.Match) -> str:
        y = m.group("y"); mm = int(m.group("m")); dd = int(m.group("d"))
        return f"{digits_to_chinese(y)}年{int_to_chinese(mm)}月{int_to_chinese(dd)}日"
    text = RE_YMD_CN.sub(rep_ymd, text)

    def rep_md(m: re.Match) -> str:
        mm = int(m.group("m")); dd = int(m.group("d"))
        return f"{int_to_chinese(mm)}月{int_to_chinese(dd)}日"
    text = RE_MD_CN.sub(rep_md, text)

    def rep_hm(m: re.Match) -> str:
        hh = int(m.group("h")); mm = int(m.group("m"))
        return f"{int_to_chinese(hh)}时{int_to_chinese(mm)}分"
    text = RE_HM_CN.sub(rep_hm, text)
    return text

def _normalize_percent(text: str) -> str:
    return RE_PERCENT.sub(lambda m: "百分之" + number_to_chinese(m.group("num")), text)

def _normalize_ranges(text: str) -> str:
    # 先把 3500-4000 / 27-29 等替换成 “到”
    return RE_RANGE.sub(lambda m: f"{m.group('a')}到{m.group('b')}", text)

def _normalize_signed_units(text: str) -> str:
    # 再处理 -3.5°C / -12℃ 等（放在区间之后，避免把右端当负数）
    def rep(m: re.Match) -> str:
        num = m.group("num"); unit = m.group("unit")
        unit_std = UNIT_MAP.get(unit, UNIT_MAP.get(unit.lower(), unit))
        return "负" + number_to_chinese(num) + unit_std
    return RE_SIGNED_NUM_UNIT.sub(rep, text)

def _normalize_num_units(text: str) -> str:
    # 3.14m/s、15kg、512GB 等（保护后的 USB3.0/H2O 不会落到这里）
    def rep(m: re.Match) -> str:
        num  = m.group("num"); unit = m.group("unit")
        unit_std = UNIT_MAP.get(unit, UNIT_MAP.get(unit.lower(), unit))
        return number_to_chinese(num) + unit_std
    return RE_NUM_UNIT.sub(rep, text)

def _normalize_neg_pure(text: str) -> str:
    return RE_NEG_PURE.sub(lambda m: "负" + number_to_chinese(m.group("num")), text)

def _looks_like_id_left(left: str, num: str) -> bool:
    if "." not in num and len(num) >= 5:  # 长串默认编号逐位读
        return True
    if left.endswith(("：", ":", "、", "「", "【")):
        return True
    return False

def _looks_like_measure_right(right: str) -> bool:
    for w in sorted(MEASURE_WORDS, key=len, reverse=True):
        if right.startswith(w):
            return True
    return False

def _normalize_pure_numbers(text: str) -> str:
    def rep(m: re.Match) -> str:
        n = m.group("num")
        s, e = m.span()
        left  = text[max(0, s-12):s]
        right = text[e:e+6]
        if _looks_like_measure_right(right):
            return number_to_chinese(n)
        if _looks_like_id_left(left, n):
            return digits_to_chinese(n)
        if "." not in n and len(n) <= 4:
            return number_to_chinese(n)
        return digits_to_chinese(n)
    return RE_PURE_NUM.sub(rep, text)

def normalize_text(text: str) -> str:
    # 先保护“字母在前/夹数字”的混合 token
    protected, tok_map = _protect_alnum_tokens(text)

    t = _normalize_dates(protected)
    t = _normalize_percent(t)
    t = _normalize_ranges(t)        # 提前
    t = _normalize_signed_units(t)  # 之后再处理 -3.5°C
    t = _normalize_num_units(t)
    t = _normalize_neg_pure(t)
    t = _normalize_pure_numbers(t)

    return _restore_tokens(t, tok_map)

# --- quick test ---
if __name__ == "__main__":
    samples = [
        "编号：1789310",
        "15kg",
        "300元",
        "2024年3月15日",
        "2024-03-15",
        "3-5天",
        "2万, 2000万, 200万, 20万",
        "USB3.0 接口, H2O, v2, A1-2B, C-130",
        "A2024年 不应改, V2kg 不应改",
        "03月05日",
        "违停罚单，编号20240305",
        "人民币0.75元（约合0.11美元",
        "订单号是：08195761",
        "配上16GB内存和512GB硬盘",
        "订3月15日MU5115上海飞北京",
        "修武县气象台2023年6月11日20时53分发布",
        "65英寸，4k分辨率，",
        "大概是100元",
        "下个月的15号到17号",
        "支付137.9元",
        "指南：1.政府及相关部",
        "缴费300元",
        "，200元",
        "超过5000公里了",
        "，总共不超过2000元",
        "男性平均为 3500-4000 mL，女性为 2500-3000 mL",
        "住宿日期从11月20日到11月23日",
        "充值300返1000’的",
        "11月20号入",
        "素受体（PTH1R）广泛分",
        "充值300元餐费",
        "总共2000元",
        "台2022年11月27日11时40分发布",
        "伊犁州气象局2023年3月31日17时联",
        "筛查，价格1280元，",
        "预计27-29日",
        "29日全县最高气温降至-12到-6℃",
        "全县有5级左右（5.5到13.8米/秒",
        "天气（能见度1到9千米",
        "18000元",
        "温度-3.5°C，折扣12.5%"
    ]
    for s in samples:
        print(s, "=>", normalize_text(s))
