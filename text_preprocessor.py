"""
text_preprocessor.py
Converts raw input text into clean, pronounceable text for F5-TTS.
Handles: abbreviations, brand names, numbers, phone numbers, currencies,
         ordinals, mixed Hindi-English, special symbols.
"""

import re

# ── Abbreviation / Brand dictionary ──────────────────────────
# Add any domain-specific terms here
ABBREV_MAP = {
    # Brand names
    r'\bJioCX\b':       'Jio C X',
    r'\bJio\b':         'Jio',
    r'\bHDFC\b':        'H D F C',
    r'\bICICI\b':       'I C I C I',
    r'\bSBI\b':         'S B I',
    r'\bUPI\b':         'U P I',
    r'\bOTP\b':         'O T P',
    r'\bKYC\b':         'K Y C',
    r'\bEMI\b':         'E M I',
    r'\bAPI\b':         'A P I',
    r'\bAI\b':          'A I',
    r'\bML\b':          'M L',
    r'\bUI\b':          'U I',
    r'\bUX\b':          'U X',
    r'\bCEO\b':         'C E O',
    r'\bCTO\b':         'C T O',
    r'\bHR\b':          'H R',
    r'\bIT\b':          'I T',
    r'\bAC\b':          'A C',
    r'\bTV\b':          'T V',
    r'\bDOB\b':         'date of birth',
    r'\bDOJ\b':         'date of joining',
    r'\bPAN\b':         'P A N',
    r'\bGST\b':         'G S T',
    r'\bETA\b':         'E T A',
    r'\bFAQ\b':         'frequently asked questions',
    r'\bAsap\b':        'as soon as possible',
    r'\bASAP\b':        'as soon as possible',

    # Units
    r'\bkm\b':          'kilometers',
    r'\bkg\b':          'kilograms',
    r'\bmg\b':          'milligrams',
    r'\bkm/h\b':        'kilometers per hour',
    r'\bkWh\b':         'kilowatt hours',
}


# ── Number helpers ────────────────────────────────────────────

ONES = ['', 'one', 'two', 'three', 'four', 'five', 'six',
        'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve',
        'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen',
        'eighteen', 'nineteen']

TENS = ['', '', 'twenty', 'thirty', 'forty', 'fifty',
        'sixty', 'seventy', 'eighty', 'ninety']


def number_to_words(n: int) -> str:
    """Convert integer to English words."""
    if n < 0:
        return 'minus ' + number_to_words(-n)
    if n == 0:
        return 'zero'
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ((' ' + ONES[n % 10]) if n % 10 else '')
    if n < 1000:
        rest = number_to_words(n % 100)
        return ONES[n // 100] + ' hundred' + ((' and ' + rest) if rest else '')
    if n < 100000:
        rest = number_to_words(n % 1000)
        return number_to_words(n // 1000) + ' thousand' + ((' ' + rest) if rest else '')
    if n < 10000000:
        rest = number_to_words(n % 100000)
        return number_to_words(n // 100000) + ' lakh' + ((' ' + rest) if rest else '')
    rest = number_to_words(n % 10000000)
    return number_to_words(n // 10000000) + ' crore' + ((' ' + rest) if rest else '')


def ordinal_to_words(n: int) -> str:
    """Convert integer to ordinal words: 1 → first."""
    ordinals = {
        1: 'first', 2: 'second', 3: 'third', 4: 'fourth',
        5: 'fifth', 6: 'sixth', 7: 'seventh', 8: 'eighth',
        9: 'ninth', 10: 'tenth', 11: 'eleventh', 12: 'twelfth',
    }
    if n in ordinals:
        return ordinals[n]
    word = number_to_words(n)
    if word.endswith('y'):
        return word[:-1] + 'ieth'
    return word + 'th'


# ── Phone number normaliser ───────────────────────────────────

def expand_phone(match) -> str:
    """
    91-9876543210 or 9876543210 or 022-12345678
    → nine eight seven six five four three two one zero
    """
    digits = re.sub(r'[\s\-\(\)]', '', match.group())
    # Remove country code if starts with 91 and is 12 digits
    if len(digits) == 12 and digits.startswith('91'):
        digits = digits[2:]
    return ' '.join(
        ['zero','one','two','three','four',
         'five','six','seven','eight','nine'][int(d)]
        for d in digits
    )


# ── Currency normaliser ───────────────────────────────────────

def expand_currency(match) -> str:
    symbol  = match.group(1)
    amount  = match.group(2).replace(',', '')
    currency_name = {
        '₹': 'rupees', '$': 'dollars',
        '€': 'euros',  '£': 'pounds',
    }.get(symbol, '')
    try:
        val = float(amount)
        if val == int(val):
            return number_to_words(int(val)) + ' ' + currency_name
        rupees  = int(val)
        paise   = round((val - rupees) * 100)
        result  = number_to_words(rupees) + ' ' + currency_name
        if paise:
            result += ' and ' + number_to_words(paise) + ' paise'
        return result
    except ValueError:
        return match.group()


# ── Decimal / float normaliser ────────────────────────────────

def expand_decimal(match) -> str:
    integer_part, decimal_part = match.group(1), match.group(2)
    result = number_to_words(int(integer_part.replace(',', '')))
    result += ' point '
    result += ' '.join(
        ['zero','one','two','three','four',
         'five','six','seven','eight','nine'][int(d)]
        for d in decimal_part
    )
    return result


# ── Time normaliser ───────────────────────────────────────────

def expand_time(match) -> str:
    hour, minute = int(match.group(1)), int(match.group(2))
    suffix = match.group(3) or ''
    result = number_to_words(hour)
    if minute == 0:
        result += " o'clock"
    elif minute < 10:
        result += ' oh ' + number_to_words(minute)
    else:
        result += ' ' + number_to_words(minute)
    if suffix.lower() == 'am':
        result += ' in the morning'
    elif suffix.lower() == 'pm':
        result += ' in the evening' if hour >= 5 else ' in the afternoon'
    return result


# ── Main preprocessor ─────────────────────────────────────────

def preprocess_text(text: str) -> str:
    """
    Full pipeline — apply all normalisations in correct order.
    """

    # 1. Expand abbreviations and brand names
    for pattern, replacement in ABBREV_MAP.items():
        text = re.sub(pattern, replacement, text)

    # 2. Phone numbers (before general number expansion)
    #    Matches: 9876543210, 022-12345678, +91-9876543210
    text = re.sub(
        r'\+?(?:91[-\s]?)?\d{10}|\d{3}[-\s]\d{7,8}',
        expand_phone,
        text
    )

    # 3. Currency
    text = re.sub(
        r'([₹$€£])([\d,]+(?:\.\d{1,2})?)',
        expand_currency,
        text
    )

    # 4. Percentages
    text = re.sub(
        r'(\d+(?:\.\d+)?)\s*%',
        lambda m: number_to_words(int(float(m.group(1)))) + ' percent'
        if float(m.group(1)) == int(float(m.group(1)))
        else expand_decimal(re.match(r'(\d+)\.(\d+)', m.group(1))) + ' percent',
        text
    )

    # 5. Time (before ordinals/generals)
    text = re.sub(
        r'\b(\d{1,2}):(\d{2})\s*(am|pm|AM|PM)?\b',
        expand_time,
        text
    )

    # 6. Ordinals: 1st, 2nd, 3rd, 4th ...
    text = re.sub(
        r'\b(\d+)(st|nd|rd|th)\b',
        lambda m: ordinal_to_words(int(m.group(1))),
        text
    )

    # 7. Decimal numbers
    text = re.sub(
        r'\b(\d{1,3}(?:,\d{3})*|\d+)\.(\d+)\b',
        expand_decimal,
        text
    )

    # 8. Large numbers with commas: 1,00,000 or 1,000,000
    text = re.sub(
        r'\b\d{1,3}(?:,\d{2,3})+\b',
        lambda m: number_to_words(int(m.group().replace(',', ''))),
        text
    )

    # 9. Plain integers
    text = re.sub(
        r'\b(\d+)\b',
        lambda m: number_to_words(int(m.group())),
        text
    )

    # 10. Clean up symbols
    text = re.sub(r'&', ' and ', text)
    text = re.sub(r'@', ' at ', text)
    text = re.sub(r'#', ' number ', text)
    text = re.sub(r'/', ' or ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# ── Test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        "Your JioCX account has been activated.",
        "Please pay ₹1,499 as your EMI.",
        "Call us at 9876543210 for support.",
        "Your OTP is 482910.",
        "HDFC Bank offers 8.5% interest rate.",
        "Meeting at 3:30 PM on the 1st of July.",
        "Your order #12345 will arrive in 3 days.",
        "The distance is 42.5 km from here.",
        "We serve 1,00,000 customers daily.",
        "Your account balance is ₹23,450.75.",
    ]
    for t in tests:
        print(f"IN  : {t}")
        print(f"OUT : {preprocess_text(t)}")
        print()