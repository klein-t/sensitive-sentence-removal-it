"""Deterministic regex mask for personal identifiers in appraisal text.

Applied to every text before any other step (labelling, training, evaluation,
baselines). Two principles:
  - mask the TRIPLE that identifies a person (name, fiscal code, birth date), not
    everything that looks like personal data: property addresses, contract dates,
    amounts and cadastral data stay intact;
  - an age is a number and identifies nobody, a birth date does. So a birth date is
    replaced by the age computed from it before masking.
"""
import datetime as dt
import re

EVEN = {c: i for i, c in enumerate("0123456789")}
EVEN.update({c: i for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")})
ODD = dict(zip("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
               [1, 0, 5, 7, 9, 13, 15, 17, 19, 21, 1, 0, 5, 7, 9, 13, 15, 17, 19, 21, 2, 4, 18, 20, 11, 3, 6, 8, 12, 14, 16, 10, 22, 25, 24, 23],
               strict=True))
MONTHS = dict(zip("ABCDEHLMPRST", range(1, 13), strict=True))


def valid_fiscal_code(cf):
    """The fiscal code has a check character: verifying it removes false positives."""
    cf = cf.upper()
    if len(cf) != 16:
        return False
    s = sum((ODD if i % 2 == 0 else EVEN)[c] for i, c in enumerate(cf[:15]))
    return "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[s % 26] == cf[15]


def birth_date_from_fiscal_code(cf):
    """The fiscal code encodes the birth date: decode it so it can be masked by value."""
    cf = cf.upper()
    try:
        year = int(cf[6:8])
        month = MONTHS[cf[8]]
        day = int(cf[9:11])
        if day > 40:
            day -= 40  # women have the day increased by 40
        today = dt.date.today()
        full = 1900 + year if 1900 + year <= today.year - 10 else 2000 + year
        return dt.date(full, month, day)
    except Exception:
        return None


RE_FISCAL_CODE = re.compile(r"\b[A-Za-z]{6}\d{2}[A-Za-z]\d{2}[A-Za-z]\d{3}[A-Za-z]\b")
RE_BORN = re.compile(r"(?i)\b(nat[oa]\s+(?:a|in)\s+[A-Z][\w'’\-]+(?:\s+[A-Z][\w'’\-]+)?\s*(?:\([A-Z]{2}\))?\s*(?:,\s*)?"
                     r"(?:il|in data)\s*)(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4})")
RE_TITLE = re.compile(r"\b(Sig(?:\.|ra\.?|nor[ae]?)?|Dott(?:\.|oressa|\.ssa)?|Avv\.?|Geom\.?|Arch\.?|"
                      r"Ing\.?|Rag\.?|Prof\.?|Perito|C\.T\.U\.?|Notaio)[ \t]+"
                      r"([A-Z][a-zà-ú'’]{2,}(?:\s+(?:[A-Z]\.|[A-Z][a-zà-ú'’]{2,})){0,2})")
RE_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b")
RE_IBAN = re.compile(r"\bIT\d{2}[A-Z]\d{10}[0-9A-Z]{12}\b")
# no \s in the class: it would include the newline, and the number would swallow the start
# of the next line, which in an appraisal is almost always a cadastral table row.
RE_PHONE = re.compile(r"(?i)\b(?:tel|fax|cell)\.?[ \t]*:?[ \t]*\+?[\d][\d ().\-]{5,16}\d")


def age(birth, at=None):
    at = at or dt.date.today()
    return at.year - birth.year - ((at.month, at.day) < (birth.month, birth.day))


def mask(t, keep_age=True):
    """Return (masked text, counts). The text stays readable: identifiers are replaced
    by descriptive placeholders, so a reader still sees that a person was mentioned there."""
    c = {"fiscal_codes": 0, "birth_dates": 0, "names": 0, "emails": 0, "ibans": 0, "phones": 0}
    births = []

    def _fiscal_code(m):
        # The check character is used to DECODE the birth date, not to decide whether to mask:
        # a code corrupted by OCR still identifies a person. A sequence with that shape is
        # almost never anything else, so it is always masked.
        v = m.group(0)
        c["fiscal_codes"] += 1
        if valid_fiscal_code(v):
            d = birth_date_from_fiscal_code(v)
            if d:
                births.append(d)
        return "[CF]"
    t = RE_FISCAL_CODE.sub(_fiscal_code, t)

    def _born(m):
        c["birth_dates"] += 1
        head, date = m.group(1), m.group(2)
        if keep_age:
            for f in ("%d/%m/%Y", "%d.%m.%Y", "%d-%m-%Y", "%d/%m/%y", "%d.%m.%y", "%d-%m-%y"):
                try:
                    d = dt.datetime.strptime(date, f).date()
                    return f"{head}[DATA DI NASCITA, eta' {age(d)} anni]"
                except ValueError:
                    continue
        return f"{head}[DATA DI NASCITA]"
    t = RE_BORN.sub(_born, t)

    # birth dates decoded from fiscal codes, wherever they appear
    for d in births:
        for f in ("%d/%m/%Y", "%d.%m.%Y", "%d-%m-%Y", "%d/%m/%y"):
            s = d.strftime(f)
            if s in t:
                t = t.replace(s, f"[DATA DI NASCITA, eta' {age(d)} anni]" if keep_age else "[DATA DI NASCITA]")
                c["birth_dates"] += 1

    def _title(m):
        c["names"] += 1
        return f"{m.group(1)} [nome]"
    t = RE_TITLE.sub(_title, t)

    def count(regex, key, repl):
        nonlocal t
        n = len(regex.findall(t))
        if n:
            c[key] += n
            t = regex.sub(repl, t)
    count(RE_EMAIL, "emails", "[email]")
    count(RE_IBAN, "ibans", "[IBAN]")
    count(RE_PHONE, "phones", "[tel]")
    return t, c
