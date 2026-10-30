"""Localized final-reply safety checks and plain-text formatting, without API calls."""
import re
import unicodedata


def normalized(text):
    text = unicodedata.normalize("NFKD", text.casefold())
    return "".join(c for c in text if not unicodedata.combining(c) and c != "ـ").replace("’", "'")


def is_allergy(text):
    return bool(re.search(r"\b(?:allerg\w*|7asasi\w*|7assasi\w*|hasasi\w*|hassasi\w*)\b|حساسي|تحسس", normalized(text)))


def is_dietary(text):
    return bool(re.search(r"\b(?:gluten|glout(?:en|ine)|vegan\w*|vegetari\w*|vegetali\w*|nabati\w*|nabete\w*|vegetal\w*)\b|غلوتين|جلوتين|غلوتن|نباتي", normalized(text)))


SAFETY = {
    "en": "Our kitchen handles nuts, gluten, dairy and sesame. Cross-contact cannot be ruled out, so we cannot guarantee any dish is safe. The team can call you back before you order.",
    "fr": "Notre cuisine utilise des noix, du gluten, des produits laitiers et du sésame. Un contact croisé ne peut pas être exclu, donc nous ne pouvons garantir qu'un plat est sans danger. L'équipe peut vous rappeler avant votre commande.",
    "ar": "مطبخنا بيتعامل مع المكسرات والغلوتين والحليب ومشتقاته والسمسم. ما فينا نستبعد التلامس بين المكونات، وما فينا نضمن إنو أي طبق آمن. الفريق فيه يتصل فيك قبل ما تطلب.",
    "az": "Bel matbakh fi nuts, gluten, 7alib w mecht2ato w semsom. Ma fina nesteb3ed cross-contamination, w ma fina nodman ennu ay taba2 safe. El team fiyo ye7kik abel ma totlob.",
}
CROSS_CONTACT = {
    "en": "Cross-contact in our kitchen cannot be ruled out.",
    "fr": "Un contact croisé dans notre cuisine ne peut pas être exclu.",
    "ar": "ما فينا نستبعد التلامس بين المكونات بمطبخنا.",
    "az": "Bas ma fina nesteb3ed cross-contamination bel matbakh.",
}


def plain_text(reply):
    reply = re.sub(r"^\s*#{1,6}\s+", "", reply, flags=re.M)
    reply = re.sub(r"^\s*```[^\n]*\n?|^\s*```\s*$", "", reply, flags=re.M)
    reply = re.sub(r"(\*\*|__)(.*?)\1", r"\2", reply, flags=re.S)
    reply = re.sub(r"(?<!\w)[*_]([^\n*_]+)[*_](?!\w)", r"\1", reply)
    reply = re.sub(r"^\s*[*•]\s+", "- ", reply, flags=re.M)
    reply = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", reply)
    return reply.replace("`", "").strip()


def asks_prices(user):
    return bool(re.search(r"\b(?:prices?|costs?|how much|combien|prix|addesh|addech|adeesh|2addesh)\b|قديش|بكم|اسعار|سعر", normalized(user)))


def is_menu_category(user):
    return is_dietary(user) or bool(re.search(
        r"\b(?:menu|mezze|mezza|desserts?|drinks|grills|entrees)\b|منيو|مازة|مقبلات|حلويات|مشروبات|مشاوي", normalized(user)))


def turn_instructions(user):
    if is_allergy(user):
        return ("ALLERGY TURN: The application will automatically prepend the complete localized safety statement. "
                "Do NOT write kitchen-allergen warnings, cross-contact statements, safety guarantees or callback offers yourself. "
                "Only briefly answer the specific dish question using reference ingredient facts, then ask only for the name "
                "(or the next missing contact field if the name was already supplied). Ingredient facts about the asked dish "
                "are allowed, e.g. hummus has no nuts but contains tahini (sesame paste). Never guarantee safety.")
    instructions = []
    if is_dietary(user):
        instructions.append("DIETARY TURN: The application will automatically append ONE localized cross-contact line. "
                            "Do NOT write cross-contact or general kitchen-allergen/safety warnings yourself.")
    if is_menu_category(user) and not asks_prices(user):
        instructions.append("For a dietary/menu-category LIST, give dish NAMES ONLY, comma-separated on one or two lines, "
                            "at most 10 dishes. No prices, VG/GF/V/N labels, portions, headers or bullet lists. "
                            "Then offer more details in one short sentence. For a question about one specific dish, "
                            "answer that question briefly instead of listing a category. Keep 'without bread' on qualifying "
                            "gluten-free grills; arayes always contains gluten. Prices only when the customer asks for prices.")
    return "\n".join(instructions)


# Remove generic warning sentences, not dish-specific facts about nuts/sesame/gluten.
# Unicode hyphens and diacritics are normalized before matching.
WARNING_PATTERNS = (
    r"cross[\s-]*(?:contact|contamination)|(?:contact|contamination)\s+croise\w*|التلامس|تلوث|تلامس|اختلاط",
    r"\b(?:guarantee\w*|garanti\w*|nodman|nadman|nkoun akidin)\b|نضمن|ضمان",
    r"\b(?:allergens?|allergenes?)\b|مسببات الحساسية",
    r"\b(?:kitchen|cuisine|matbakh)\b.{0,200}(?:nuts|gluten|dairy|milk|sesame|noix|lait|7alib|semsom|mkassarat)|مطبخ.{0,200}(?:مكسرات|غلوتين|جلوتين|حليب|سمسم)",
    r"\b(?:call\s+(?:you\s+)?back|callback|rappeler|rappelera|ye7kik|bya7kik|yettesel)\b|(?:الفريق|فريقنا).{0,60}(?:يتصل|يحكيك|يتواصل)",
    r"(?:100\s*%|completely|totally|entirely)\s+safe|sans aucun risque|امن تماما|مضمون.{0,20}امن",
)


def remove_safety_sentences(reply):
    kept = []
    for line in reply.splitlines():
        # Keep questions following a warning on the same line as separate sentences.
        sentences = re.split(r"(?<=[.!?؟؛;])\s+", line.strip())
        clean = []
        for sentence in sentences:
            text = normalized(sentence).translate(str.maketrans("‐‑‒–—", "-----"))
            if sentence and not any(re.search(pattern, text) for pattern in WARNING_PATTERNS):
                clean.append(sentence)
        if clean:
            kept.append(" ".join(clean))
    return "\n".join(kept).strip()


MORE_DETAILS = {
    "en": "Would you like more details?",
    "fr": "Vous voulez plus de détails ?",
    "ar": "بدك تفاصيل أكتر؟",
    "az": "Baddak tafasil aktar?",
}
NAME_QUESTION = {"en": "May I have your name?", "fr": "Quel est votre nom ?", "ar": "شو اسمك؟", "az": "Shu esmak?"}


def short_category_list(reply, language):
    # Defensive formatting for model lists, including the previously observed bullet output.
    # Avoid rewriting a single-dish prose answer just because it mentions a dietary term.
    # Strip label groups and prices before splitting commas (VG, GF is one label group).
    reply = re.sub(r"\((?:\s*(?:VG|GF|V|N)\s*[,/]?)+\)", "", reply, flags=re.I)
    reply = re.sub(r"(?:[$€]\s*\d+(?:[.,]\d+)?|\d+(?:[.,]\d+)?\s*(?:[$€]|USD|dollars?|دولار))", "", reply, flags=re.I)
    bullets = re.findall(r"^\s*(?:[-*•]|\d+[.)])\s+(.+)$", reply, re.M)
    if bullets:
        items = bullets
    else:
        lines = [line.strip() for line in reply.splitlines() if line.strip()]
        list_lines = [line for line in lines if len(re.split(r"[,،]", line)) >= 3 and not re.search(r"[?؟]", line)]
        if not list_lines:
            return reply
        items = [item for line in list_lines[:2] for item in re.split(r"[,،]", line.split(":", 1)[-1])]
    dishes = []
    for item in items:
        item = re.sub(r"\([^)]*(?:without bread|bala khebez|sans pain|بلا خبز)[^)]*\)",
                      {"en":" without bread", "fr":" sans pain", "ar":" بلا خبز", "az":" bala khebez"}[language], item, flags=re.I)
        item = re.sub(r"\((?:\s*(?:VG|GF|V|N)\s*[,/]?)+\)", "", item, flags=re.I)
        item = re.sub(r"\b(?:VG|GF)\b", "", item)
        item = re.sub(r"\(\s*\d+\s*(?:pcs|pieces|personnes|people|قطع)\s*\)", "", item, flags=re.I)
        item = re.sub(r"(?:[$€]\s*\d+(?:[.,]\d+)?|\d+(?:[.,]\d+)?\s*(?:[$€]|USD|dollars?|دولار))", "", item, flags=re.I)
        item = re.split(r"\s+[–—-]\s+|:", item, maxsplit=1)[0].strip(" -–—:،,.")
        item = re.sub(r"\s+", " ", item)
        if item and item.casefold() not in {d.casefold() for d in dishes}:
            dishes.append(item)
        if len(dishes) == 10:
            break
    if not dishes:
        return reply
    return ", ".join(dishes) + ".\n" + MORE_DETAILS[language]


def finalize_reply(user, reply, language):
    reply = plain_text(reply)
    if is_allergy(user):
        body = remove_safety_sentences(reply) or NAME_QUESTION[language]
        return SAFETY[language] + "\n\n" + body
    dietary = is_dietary(user)
    if dietary:
        reply = remove_safety_sentences(reply)
    if is_menu_category(user) and not asks_prices(user):
        reply = short_category_list(reply, language)
    if dietary:
        return (reply + "\n\n" if reply else "") + CROSS_CONTACT[language]
    return reply
