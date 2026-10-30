"""Session booking validation, conservative extraction and deterministic reply checks."""
import re
import unicodedata
from datetime import datetime, timedelta
from booking_guard import NUMBER_WORDS, normalize

FIELDS = ('name','phone','date','time','party_size','seating','notes')
REQUIRED = FIELDS[:-1]
FORBIDDEN = {'كام':'قديش','عشان':'تا','تقدر':'فيك','ترسلو':'تبعتو'}
LANGUAGE = {
    'en':'English only.', 'fr':'French only.',
    'ar':'Lebanese Arabic only: بندورة، توم، شو، فيك، هيدا. Never use كام، عشان، الحين، تقدر، ترسلو، طماطم. ليش is fine.',
    'az':'Beirut Arabizi ONLY: Latin letters and numbers, NEVER Arabic script. Natural short Lebanese texting; familiar English words are allowed.',
}
QUESTIONS = {
    'en': ["What's your name?", "What's your phone number?", "Which day would you like to come?", "What time?", "How many people?", "Inside or on the terrace?"],
    'ar': ["شو اسمك؟", "شو رقم التلفون؟", "أي يوم بدك تجي؟", "أي ساعة؟", "قديش شخص؟", "جوّا ولا عالتراس؟"],
    'az': ["Shu esmak?", "Shu ra2m el telefone?", "Ayya yom baddak teje?", "Ayya se3a?", "Addesh shakhes?", "Jowwa wala 3al terrace?"],
    'fr': ["Quel est votre nom ?", "Quel est votre numéro de téléphone ?", "Pour quel jour ?", "À quelle heure ?", "Pour combien de personnes ?", "À l'intérieur ou en terrasse ?"],
}


def ascii_digits(text):
    return ''.join(str(unicodedata.decimal(c)) if c.isdecimal() else c for c in str(text))


def arabic_script(text):
    return bool(re.search(r'[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\ufb50-\ufdff\ufe70-\ufeff]', text))


def forbidden_words(text):
    text = normalize(text)
    return [word for word in FORBIDDEN if re.search(r'(?<!\w)' + word + r'(?!\w)',text)]


def detect_language(text):
    normalized = normalize(text)
    # Contact details, dates and neutral acknowledgements never loosen the stored language.
    if normalized in {'ok','okay','yes','yes please','confirm','sure','no','cancel','oui','non','اكيد','تمام','نعم','ايه','اي'}:
        return None
    if re.fullmatch(r'[+\d\s():/.-]+',text.strip()):
        return None
    if normalized in {'السبت','يوم السبت','الاحد','يوم الاحد','الاثنين','الثلاثاء','الاربعاء','الخميس','الجمعة'}:
        return None
    if re.search(r'\b(بدي|بدنا|شو|عندي|عندكن|احجز|حجز|اريد|اكل|بدون|جلوتين|غلوتين|سعر|خليهن|بكرا|الساعة|السبت|حساسية|في|قديش|فيك|ليش|لا)\b|عالتراس|توصيل',normalized):
        return 'ar'
    tokens = re.findall(r'[a-z0-9]+',text.lower())
    if any(not re.fullmatch(r'\d+(?:am|pm)',token) and re.search(r'(?:[a-z]+[23578]|[23578][a-z]+)',token) for token in tokens):
        return 'az'
    if re.search(r'\b(badde|baddi|bokra|bukra|tawle|ashkhas|khames|shu|fi|la2|eh|tamam|ya3ne|addesh|fiyye|hajez|nehna|ehjoz|nabati|hasasiye)\b',normalized):
        return 'az'
    if re.search(r'\b(je|vous|bonjour|salut|reserver|reservation|personnes|pour|combien|livrez|livraison|allergique|allergie|vegetarien|vegetalien|sans|merci)\b',normalized):
        return 'fr'
    if re.search(r'\b(i|what|where|when|how|can|do|is|book|booking|table|actually|change|make|want|need|my|the|please)\b',normalized):
        return 'en'
    return None


def booking_intent(text):
    return bool(re.search(r'\b(book|booking|reserve|reserver|reservation|table|tawle|e7joz|ehjoz|hajez|7ajez)\b|حجز|احجز|طاولة',normalize(text)))


def bounds(day, knowledge):
    label = 'Monday to Thursday' if day.weekday() < 4 else 'Friday to Sunday'
    match = re.search(r'\| '+label+r' \| (\d\d):(\d\d) to \d\d:\d\d \| (\d\d):(\d\d) \|',knowledge)
    if not match:
        raise ValueError('Opening hours need team review.')
    oh,om,kh,km = map(int,match.groups())
    return oh*60+om,kh*60+km


def resolve_date(value,current):
    raw = ascii_digits(value).strip()
    text = normalize(raw)
    if text in {'tomorrow','bokra','bukra','بكرا','غدا','demain'}:
        day = current.date()+timedelta(days=1)
    elif text in {'today','اليوم','lyom','aujourdhui'}:
        day = current.date()
    else:
        weekdays = {'monday':0,'الاثنين':0,'lundi':0,'tuesday':1,'الثلاثاء':1,'mardi':1,'wednesday':2,'الاربعاء':2,'mercredi':2,'thursday':3,'الخميس':3,'jeudi':3,'friday':4,'الجمعة':4,'vendredi':4,'saturday':5,'السبت':5,'samedi':5,'sunday':6,'الاحد':6,'dimanche':6}
        key = next((key for key in weekdays if re.search(r'\b'+key+r'\b',text)),None)
        if key:
            day = current.date()+timedelta(days=(weekdays[key]-current.weekday())%7)
        else:
            if not re.fullmatch(r'\d{4}-\d{2}-\d{2}',raw):
                raise ValueError('Use a date within the next 30 days.')
            day = datetime.strptime(raw,'%Y-%m-%d').date()
    if not current.date() <= day <= current.date()+timedelta(days=30):
        raise ValueError('Choose today or a date within the next 30 days.')
    return day.isoformat()


def resolve_time(value,day,current,knowledge):
    raw = ascii_digits(value).strip().lower()
    match = re.fullmatch(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)?',raw)
    if not match:
        raise ValueError('Use a valid time, e.g. 21:00.')
    hour,minute = int(match[1]),int(match[2] or 0)
    period = match[3]
    if minute > 59 or hour > 23 or (period and not 1 <= hour <= 12):
        raise ValueError('Use a valid time.')
    visit_day = datetime.fromisoformat(day).date() if day else None
    opening, closing = bounds(visit_day,knowledge) if visit_day else (12*60,23*60+30)
    if period:
        hour = hour%12 + (12 if period=='pm' else 0)
    elif 1 <= hour <= 11 and hour*60+minute < opening:
        hour += 12
    if not opening <= hour*60+minute <= closing:
        raise ValueError('Choose a time between opening and kitchen last order.')
    if visit_day == current.date() and (hour,minute) <= (current.hour,current.minute):
        raise ValueError('Choose a future booking time.')
    return f'{hour:02d}:{minute:02d}'


def update_draft(draft,values,current,knowledge,old_errors=None):
    errors = {k:v for k,v in (old_errors or {}).items() if k in FIELDS or k == "change"}
    accepted = {}
    for field,value in values.items():
        if field not in FIELDS:
            errors[field] = 'Unknown booking field.'
            continue
        try:
            if field == 'name':
                if not isinstance(value,str) or not value.strip() or len(value)>100:
                    raise ValueError('Please supply a valid name.')
                value = value.strip()
            elif field == 'phone':
                if not isinstance(value,str):
                    raise ValueError('Please supply a valid phone number.')
                value = ascii_digits(value).strip()
                if not re.fullmatch(r'[+\d ()-]{7,30}',value) or not 7 <= len(re.sub(r'\D','',value)) <= 15:
                    raise ValueError('Please supply a valid phone number.')
            elif field == 'date':
                value = resolve_date(value,current)
            elif field == 'time':
                # Validate times after any date update, independent of tool-argument order.
                continue
            elif field == 'party_size':
                if type(value) is not int or not 1 <= value <= 10:
                    raise ValueError('Online bookings allow 1-10 people; larger groups need a callback.')
            elif field == 'seating':
                if value not in {'indoor','terrace'}:
                    raise ValueError('Choose indoor or terrace.')
            elif field == 'notes':
                if not isinstance(value,str) or len(value)>1000:
                    raise ValueError('Keep notes below 1000 characters.')
            draft[field]=value
            accepted[field]=value
            errors.pop(field,None)
        except (ValueError,TypeError) as exc:
            errors[field]=str(exc)
    if 'time' in values or ('date' in accepted and 'time' in draft):
        try:
            time = resolve_time(values.get('time',draft.get('time')),draft.get('date'),current,knowledge)
            draft['time']=time
            if 'time' in values:
                accepted['time']=time
            errors.pop('time',None)
        except (ValueError,TypeError) as exc:
            errors['time']=str(exc)
    if accepted:
        errors.pop('change',None)
    draft.setdefault('notes','')
    return accepted,errors


def missing_fields(draft):
    return [field for field in REQUIRED if draft.get(field) in (None,'')]


def next_field(draft,errors):
    return next((field for field in REQUIRED if field in errors),None) or next(iter(missing_fields(draft)),None)


def extract_details(user,draft,current):
    """Capture only unambiguous details; the tool handles other natural-language forms."""
    raw = ascii_digits(user).strip()
    text = normalize(raw)
    result = {}
    if re.fullmatch(r'\+?[\d ()-]{7,30}',raw) and 7 <= len(re.sub(r'\D','',raw)) <= 15 and not re.fullmatch(r'\d{4}-\d\d-\d\d',raw):
        result['phone']=raw
        return result
    date_match = re.search(r'\b\d{4}-\d{2}-\d{2}\b',raw)
    if date_match:
        result['date']=date_match[0]
    else:
        date_match = re.search(r'\b(tomorrow|today|bokra|bukra|demain|بكرا|اليوم|غدا|monday|tuesday|wednesday|thursday|friday|saturday|sunday|الاثنين|الثلاثاء|الاربعاء|الخميس|الجمعة|السبت|الاحد|lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche)\b',text)
        if date_match:
            result['date']=date_match[0]
    time_match = re.search(r'(?:\b(?:at|3al|3a)|الساعة|الساعه)\s*(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)(?!\d)',raw,re.I)
    if not time_match:
        time_match = re.search(r'\b(\d{1,2}:\d{2}\s*(?:am|pm)?|\d{1,2}\s*(?:am|pm))\b',raw,re.I)
    if time_match:
        result['time']=time_match[1].strip()
    elif next_field(draft,{})=='time' and re.fullmatch(r'\d{1,2}',raw):
        result['time']=raw
    number = r'(?:\d{1,2}(?!\d)|'+'|'.join(sorted((re.escape(k) for k,v in NUMBER_WORDS.items() if v<=10),key=len,reverse=True))+r')'
    patterns = [r'\b(?:for|la|pour|make it|make that)\s+('+number+r')\b',r'(?:احجز|حجز|خليهن)\s+('+number+r')\b',r'\b('+number+r')\s+(?:people|guests|persons|personnes|ashkhas|shakhes|اشخاص|شخص)\b']
    for pattern in patterns:
        match = re.search(pattern,text)
        if match:
            result['party_size']=int(match[1]) if match[1].isdecimal() else NUMBER_WORDS[match[1]]
            break
    if not result and next_field(draft,{}) == 'party_size':
        if raw.isdecimal() and len(raw) <= 2:
            result['party_size'] = int(raw)
        elif text in NUMBER_WORDS:
            result['party_size'] = NUMBER_WORDS[text]
    if re.search(r'\b(terrace|terrasse)\b|تراس',text):
        result['seating']='terrace'
    elif re.search(r'\b(inside|indoor|indoors|jowwa|interieur)\b|بالداخل|جوا',text):
        result['seating']='indoor'
    if not result and next_field(draft,{})=='name' and detect_language(raw) is None and text not in {'ok','okay','yes','sure','confirm','tamam','eh tamam','eh','لا','نعم','اكيد'} and re.fullmatch(r"[^\d?!؟:]{2,100}",raw) and len(raw.split())<=4 and not booking_intent(raw) and not re.search(r'\b(change|actually|what|how|instead|cancel|yes|no|hello|hi)\b|غير|بدل|شو|لا',text):
        result['name']=raw
    return result


FIELD_WORDS = {
    'name':r'\b(name|nom|esmak|esmek|esem|ismak)\b|اسمك|الاسم',
    'phone':r'\b(phone|telephone|telefone|telephonak|telephon|ra2m|ra2mak|numero)\b|تلفون|هاتف|رقم',
    'date':r'\b(date|day|nhar|tarikh|yom|jour)\b|نهار|تاريخ|يوم',
    'time':r'\b(time|heure|se3a)\b|ساعة|وقت',
    'party_size':r'\b(guests|people|persons|personnes|shakhes|ashkhas)\b|شخص|اشخاص|عدد',
    'seating':r'\b(seating|inside|indoor|terrace|terrasse)\b|تراس|بالداخل|جلسة',
}


def asked_fields(reply):
    found = set()
    for sentence in re.split(r'[\n.!]',reply):
        text = normalize(sentence)
        if not ('?' in sentence or '؟' in sentence or re.search(r'\b(provide|share|tell me|give me|supply)\b|عطيني|خبرني',text)):
            continue
        found.update(field for field,pattern in FIELD_WORDS.items() if re.search(pattern,text))
    return found


def reply_issues(reply,language,draft,errors):
    issues=[]
    if language=='az' and arabic_script(reply):
        issues.append('Arabizi must contain NO Arabic script; use Latin letters and numbers only.')
    if language=='ar' and forbidden_words(reply):
        issues.append('Use Lebanese words: '+', '.join(k+' -> '+v for k,v in FORBIDDEN.items()))
    repeated=asked_fields(reply)&{f for f in REQUIRED if f in draft and f not in errors}
    if repeated:
        issues.append('Already collected '+', '.join(sorted(repeated))+'. Never ask for these again. Ask ONLY for the next missing field.')
    if any(re.search(pattern,normalize(reply)) for f,pattern in FIELD_WORDS.items() if f in draft) and re.search(r"not recorded|no .{0,20}(?:recorded|on file)|don't have|do not have|ما عندي|مش مسجل",reply,re.I):
        issues.append('Collected fields are in Booking so far; do not claim they are missing.')
    return issues


def safe_reply(language,draft,errors):
    field=next_field(draft,errors)
    if field:
        return QUESTIONS[language][REQUIRED.index(field)]
    return {'en':'Which detail would you like to change?', 'ar':'أي تفصيل بدك تغيّره؟','az':'Ay tafsil baddak tghayyer?','fr':'Quel détail souhaitez-vous modifier ?'}[language]


# Only recognized booking-only inputs take the no-model path. Unknown words,
# questions and additional restaurant requests remain available to the model.
_BOOKING_WORDS = set(normalize("""i id would like want need to book booking reserve a table for at on
inside indoor indoors terrace tomorrow today please actually make it that change instead and
am pm people guests persons yes ok okay sure confirm
badde baddi e7joz ehjoz tawle la bokra bukra 3al 3a ashkhas shakhes jowwa eh tamam
je voudrais veux reserver une un pour a en le la les demain aujourd hui personnes terrasse interieur
بدي بدنا احجز حجز طاولة ل يوم بكرا اليوم غدا الساعة الساعه عالتراس تراس التراس بالداخل جوا
لا خليهن غير بدل شخص اشخاص اكيد تمام نعم
monday tuesday wednesday thursday friday saturday sunday
lundi mardi mercredi jeudi vendredi samedi dimanche
الاثنين الثلاثاء الاربعاء الخميس الجمعة السبت الاحد""").split()) | set(NUMBER_WORDS)


def routine_booking_turn(user, details):
    """True only when a turn is fully understood as booking details/acknowledgment."""
    raw = ascii_digits(user).strip()
    if '?' in raw or '؟' in raw:
        return False
    if details and set(details) in ({'name'}, {'phone'}):
        return True
    text = normalize(raw)
    if not details and not booking_intent(raw) and text not in {'ok','okay','yes','sure','eh','tamam','eh tamam','اكيد','تمام','نعم','oui'}:
        return False
    return all(word in _BOOKING_WORDS or word.isdecimal() or re.fullmatch(r'\d{1,2}(?:am|pm)',word)
               for word in text.split())


def fixed_field_question(reply, language, draft, errors):
    """Canonicalize a standalone next-field question; preserve substantive answers."""
    field = next_field(draft, errors)
    if not field or errors or asked_fields(reply) != {field}:
        return reply
    text = reply.strip()
    text = re.sub(r"^(?:sure|great|thanks|thank you|okay|ok|akid|اكيد|bien sur)[!, .]*", "", text, flags=re.I)
    if len(re.split(r'[.!\n]',text)) == 1 and (text.endswith('?') or text.endswith('؟')):
        question = normalize(text)
        if not re.search(r"\b(menu|parking|delivery|allerg|hummus|price|cost|gluten)\w*|حساسية|منيو|توصيل",question):
            return QUESTIONS[language][REQUIRED.index(field)]
    return reply
