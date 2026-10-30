"""Shared phone parsing and display; no network calls or logging."""
import unicodedata
import phonenumbers

PHONE_ERRORS = {
    "en": "That number looks incomplete. Could you check it? For example: 70 123 456",
    "ar": "الرقم ناقص شي رقم، فيك تتأكد منو؟ مثلاً: 70 123 456",
    "az": "El ra2m na2es, fik tet2akkad menno? Masalan: 70 123 456",
    "fr": "Le numéro semble incomplet. Pouvez-vous vérifier ? Par exemple : 70 123 456",
}


class PhoneValidationError(ValueError):
    pass


def compact_phone(value):
    if not isinstance(value, str):
        raise PhoneValidationError("Please supply a valid phone number.")
    value = ''.join(str(unicodedata.decimal(c)) if c.isdecimal() else c for c in value)
    value = ''.join(c for c in value if not c.isspace() and c not in '-.()')
    return '+' + value[2:] if value.startswith('00') else value


def is_phone_input(value):
    """Recognize numeric contact input, including incomplete numbers, not validity."""
    if not isinstance(value, str):
        return False
    compact = compact_phone(value)
    digits = compact[1:] if compact.startswith('+') else compact
    return bool(digits) and digits.isascii() and digits.isdigit()


def normalize_phone(value):
    compact = compact_phone(value)
    if not is_phone_input(compact):
        raise PhoneValidationError("Please supply a valid phone number.")
    try:
        number = phonenumbers.parse(compact, "LB")
    except phonenumbers.NumberParseException:
        raise PhoneValidationError("Please supply a valid phone number.") from None
    if not phonenumbers.is_valid_number(number):
        raise PhoneValidationError("Please supply a valid phone number.")
    return phonenumbers.format_number(number, phonenumbers.PhoneNumberFormat.E164)


def national_phone(value):
    number = phonenumbers.parse(normalize_phone(value), "LB")
    return phonenumbers.format_number(number, phonenumbers.PhoneNumberFormat.NATIONAL)
