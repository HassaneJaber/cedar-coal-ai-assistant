"""Deterministic large-group intent detection; no provider calls or customer storage."""
import re
import unicodedata


def normalize(text):
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(c for c in text if not unicodedata.combining(c) and c != "ـ")
    return " ".join(re.findall(r"\w+", text))


# Values are used only to distinguish <=10 from >10, not to create a booking.
# French compound tens (quatre-vingt, etc.) need not yield an exact party size.
NUMBER_WORDS = {}
for value, words in {
    1: "one un une واحد واحدة احد احدى حدا hda 7da wahad wa7ad",
    2: "two deux اتنين اثنين اثنتين اثنا اثني اثنتا اثنتي تنتين tnen tnayn",
    3: "three trois ثلاثة ثلاث تلاتة تلات tlata tlete tlet",
    4: "four quatre اربعة اربع arba3a arba3",
    5: "five cinq خمسة خمس khamse khamsa khames",
    6: "six ستة ست sette sitte sett",
    7: "seven sept سبعة سبع sab3a sab3",
    8: "eight huit ثمانية ثماني ثمان تمانية تمن tmene tmanye tmen",
    9: "nine neuf تسعة تسع tes3a tes3",
    10: "ten dix عشرة عشر 3ashra 3ashara 3ashar 3ashr",
    11: "eleven onze حدعش حداش احدعش حداشر hda3ash hda3esh 7da3ash 7da3esh hdaash hde3ash",
    12: "twelve douze اتنعش اتناش اتناشر اثنعش tna3ash tna3esh tnash tnayn3ash",
    13: "thirteen treize تلتعش تلاتعش تلتاشر tlet3ash tlet3esh",
    14: "fourteen quatorze اربعتعش اربعطعش اربعتاشر arba3ta3ash arba3tash arba3ta3esh",
    15: "fifteen quinze خمستعش خمستاشر khamsta3ash khamsta3esh khamstash",
    16: "sixteen seize ستعش ستاشر ستطعش setta3ash setta3esh settash",
    17: "seventeen سبعتعش سبعتاشر saba3ta3ash sab3ta3ash sab3tash",
    18: "eighteen تمنتعش تمنتاشر tmanta3ash tmanta3esh tmantash",
    19: "nineteen تسعتعش تسعتاشر tesa3ta3ash tes3ta3ash tes3tash",
    20: "twenty vingt عشرين عشرون 3eshreen 3ishreen 3echrin 3ishrin 3eshreene eshrin",
    30: "thirty trente ثلاثين ثلاثون تلاتين tletin tleteen talatin",
    40: "forty quarante اربعين اربعون arba3in arba3een",
    50: "fifty cinquante خمسين خمسون khamsin khamseen",
    60: "sixty soixante ستين ستون settin setteen",
    70: "seventy سبعين سبعون sab3in sab3een",
    80: "eighty ثمانين ثمانون تمانين tmenin tmeneen",
    90: "ninety تسعين تسعون tes3in tes3een",
    100: "hundred cent cents مية مئة مائة miye miyye",
    200: "ميتين مئتين مائتين mitayn miteen",
    1000: "thousand mille الف الفين alef alf",
}.items():
    for word in words.split():
        NUMBER_WORDS[normalize(word)] = value
# Arabic attaches the conjunction to the next number: واحد وعشرين.
NUMBER_WORDS.update({"و" + w: v for w, v in list(NUMBER_WORDS.items())
                     if re.search(r"[\u0600-\u06ff]", w)})
ATOM = r"(?:\d{1,5}|" + "|".join(sorted(map(re.escape, NUMBER_WORDS), key=len, reverse=True)) + r")"
COUNT = ATOM + r"(?:\s+(?:(?:and|et|w|wa|و)\s+)?" + ATOM + r"){0,7}"
PEOPLE = r"(?:people|persons?|guests?|seats?|adults?|personnes?|convives?|اشخاص|شخص|افراد|ضيوف|عالم|ناس|ashkhas|achkhas|shakhes|shakhs|nes|nafar)"
PARTY_COUNT = re.compile(r"\b(?P<count>" + COUNT + r")\s+" + PEOPLE + r"\b")
FOR_COUNT = re.compile(r"\b(?:for|pour|la|ل|لعدد)\s*(?P<count>" + COUNT + r")\b")
GROUP_COUNT = re.compile(r"\b(?:party of|group of|groupe de|we are|nous sommes|نحنا|نحن|nehna|ne7na)\s+(?P<count>" + COUNT + r")\b")
BOOKING = re.compile(r"\b(?:book|booking|reserve|reservation|table|reserver|reservons|reservez|hajez|7ajez|ehjoz|e7joz|nehjoz|ne7joz|a7joz|ahjoz|tawle)\b|حجز|احجز|نحجز|طاولة|طاوله")


def is_large_group_booking(text):
    # Keep clocks/dates from turning into apparent party sizes when punctuation is removed.
    text = re.sub(r"\b\d{1,2}:\d{2}(?:\s*[ap]m)?\b|\b\d{1,2}\s*[ap]m\b|\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}[/.-]\d{1,2}(?:[/.-]\d{2,4})?\b", " clock_or_date ", text, flags=re.I)
    text = normalize(text)
    if not BOOKING.search(text):
        return False
    for pattern in (PARTY_COUNT, FOR_COUNT, GROUP_COUNT):
        for match in pattern.finditer(text):
            count = match.group("count")
            # Negated alternatives are not a request for that party size.
            if re.search(r"\b(?:not|pas|مش|مو|mish|mesh)\s+(?:(?:for|pour|la|ل)\s+)?$", text[:match.start()]):
                continue
            if pattern is FOR_COUNT and re.match(r"\s+(?:january|february|march|april|may|june|july|august|september|october|november|december|janvier|fevrier|mars|avril|mai|juin|juillet|aout|septembre|octobre|novembre|decembre|كانون|شباط|اذار|نيسان|ايار|حزيران|تموز|اب|ايلول|تشرين)\b", text[match.end():]):
                continue
            parts = count.split()
            values = [int(w) if w.isdecimal() else NUMBER_WORDS.get(w, 0) for w in parts]
            if sum(values) >= 11:
                return True
    return False


LARGE_GROUP_REPLIES = {
    "en": "Online bookings are limited to 10 people. The team can call you back to arrange larger groups. May I have your name?",
    "ar": "الحجز أونلاين محدود لـ١٠ أشخاص. الفريق فيه يتصل فيك ليرتّب حجز للمجموعات الأكبر. شو اسمك؟",
    "az": "El hajez online mahdoud la 10 ashkhas. El team fiyo yettesel fik la yrattib hajez la majmou3at akbar. Shu esmak?",
    "fr": "Les réservations en ligne sont limitées à 10 personnes. L'équipe peut vous rappeler pour organiser l'accueil de groupes plus nombreux. Quel est votre nom ?",
}
