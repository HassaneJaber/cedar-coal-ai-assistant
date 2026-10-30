"""All phone literals below are synthetic fixtures or user-supplied format examples."""
import csv
import json
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
import phonenumbers
from openai.types.chat import ChatCompletionMessage
from assistant import Assistant, now, validate_contact, validate_booking
from booking_state import QUESTIONS
from phone_utils import normalize_phone, national_phone, PhoneValidationError, PHONE_ERRORS
from reply_policy import SAFETY, CALLBACK_OFFER
from storage import Storage, safe_cell

# Explicit user-requested format examples; never dialed.
VALID = {
    '70123456': '+96170123456',
    '+961 70 123 456': '+96170123456',
    '03 123456': '+9613123456',
    '+961 3 123456': '+9613123456',
    '01 234567': '+9611234567',
    '+961 1 234567': '+9611234567',
    '٠٧٠١٢٣٤٥٦': '+96170123456',
    '٧٠١٢٣٤٥٦': '+96170123456',
    '٠٣ ١٢٣٤٥٦': '+9613123456',
    '٠١ ٢٣٤٥٦٧': '+9611234567',
    '+٩٦١ ٧٠ ١٢٣ ٤٥٦': '+96170123456',
    '٠٠٩٦١.٧٠.(١٢٣)-٤٥٦': '+96170123456',
    '00961 (70) 123-456': '+96170123456',
    '70.123.456': '+96170123456',
    '+1 (202) 555-0123': '+12025550123',
    '001 202 555 0123': '+12025550123',
}
INVALID = ['7012345','+961 7012345','0312345','٧٠١٢٣٤٥','+٩٦١ ٧٠١٢٣٤٥','٠٣١٢٣٤٥',
           '123','bad','',None,'++96170123456','+1 200 123 0101','2025550123']


def booking():
    return dict(name='Demo Guest',phone='+96170000001',date=(now().date()+timedelta(days=1)).isoformat(),
                time='20:00',party_size=2,seating='terrace',notes='')


class PhoneValidationTests(unittest.IsolatedAsyncioTestCase):
    def test_requested_valid_formats_and_national_display(self):
        for value,expected in VALID.items():
            with self.subTest(value=value):
                self.assertEqual(normalize_phone(value),expected)
                self.assertTrue(phonenumbers.is_valid_number(phonenumbers.parse(expected,'LB')))
        self.assertEqual(national_phone('+96170123456'),'70 123 456')
        self.assertEqual(national_phone('70000001'),'70 000 001')

    def test_rejects_incomplete_and_invalid_numbers(self):
        for value in INVALID:
            with self.subTest(value=value),self.assertRaises(PhoneValidationError):
                normalize_phone(value)
        # This is possible-length but invalid according to numbering-plan metadata.
        number=phonenumbers.parse('+12001230101','LB')
        self.assertTrue(phonenumbers.is_possible_number(number))
        self.assertFalse(phonenumbers.is_valid_number(number))

    def test_current_synthetic_fixtures_are_valid(self):
        for value in ['70000000','70000001','70000002','+96170000001','+96170000002','+96171111111','+12025550123']:
            self.assertTrue(phonenumbers.is_valid_number(phonenumbers.parse(normalize_phone(value),'LB')))
        for value,expected in VALID.items():
            self.assertEqual(validate_contact({'name':'Demo Guest','phone':value})['phone'],expected)
            self.assertEqual(validate_booking({**booking(),'phone':value})['phone'],expected)

    async def test_incomplete_booking_phone_blocks_progress_without_model_four_languages(self):
        for language in PHONE_ERRORS:
            for value in INVALID[:6]:
                with self.subTest(language=language,value=value):
                    provider=SimpleNamespace(complete=AsyncMock(side_effect=AssertionError('No model call expected')))
                    bot=Assistant(provider,dry_run=True)
                    bot.language=language
                    bot.execute('update_booking_draft',{k:v for k,v in booking().items() if k!='phone'})
                    self.assertEqual(await bot.reply(value),PHONE_ERRORS[language])
                    self.assertIn('phone',bot.draft_errors)
                    self.assertIsNone(bot.pending)
                    self.assertFalse(bot.saved)
                    self.assertNotIn('phone',bot.booking_draft)
                    review=await bot.reply('٠٠٩٦١ ٧٠ ٠٠٠ ٠٠١')
                    self.assertIn('70 000 001',review)
                    self.assertEqual(bot.pending['phone'],'+96170000001')
                    provider.complete.assert_not_awaited()

    async def test_invalid_correction_retains_draft_but_blocks_confirmation(self):
        bot=Assistant(None,dry_run=True)
        bot.execute('save_booking',booking())
        self.assertEqual(await bot.reply('7012345'),PHONE_ERRORS['en'])
        self.assertEqual(bot.booking_draft['phone'],'+96170000001')
        self.assertIsNone(bot.pending)
        self.assertEqual(await bot.reply('yes'),PHONE_ERRORS['en'])
        self.assertFalse(bot.saved)
        await bot.reply('+961 (70) 000-002')
        self.assertEqual(bot.pending['phone'],'+96170000002')

    async def test_allergy_callbacks_use_same_retry_and_e164_storage(self):
        for language in PHONE_ERRORS:
            with self.subTest(language=language),tempfile.TemporaryDirectory() as directory:
                bot=Assistant(None,Storage(directory,local_only=True))
                bot.language=language
                bot.allergy_offer_pending=True
                bot.allergy_reason='Allergy question'
                await bot.reply('yes')
                self.assertEqual(await bot.reply('Demo Guest'),QUESTIONS[language][1])
                self.assertEqual(await bot.reply('+961 7012345'),PHONE_ERRORS[language])
                self.assertFalse((Path(directory)/'handoffs.csv').exists())
                await bot.reply('70.000.001')
                with (Path(directory)/'handoffs.csv').open(encoding='utf-8-sig',newline='') as stream:
                    rows=list(csv.DictReader(stream))
                self.assertEqual(rows[0]['phone'],safe_cell('+96170000001'))

    async def test_handoff_tool_rejection_has_fixed_reply_and_no_followup_save(self):
        for language in PHONE_ERRORS:
            bad={'name':'Demo Guest','phone':'0312345','reason':'Group of 15'}
            good={**bad,'phone':'70000001'}
            calls=[{'id':str(index),'type':'function','function':{'name':'request_handoff','arguments':json.dumps(values)}}
                   for index,values in enumerate([bad,good])]
            provider=SimpleNamespace(complete=AsyncMock(return_value=ChatCompletionMessage(role='assistant',tool_calls=calls)))
            bot=Assistant(provider,dry_run=True)
            bot.language=language
            self.assertEqual(await bot.reply('ok'),PHONE_ERRORS[language])
            self.assertFalse(bot.saved)
            self.assertEqual(len(bot.captured_tools),1)
            self.assertEqual(bot.captured_tools[0]['result']['field'],'phone')

    async def test_repeated_incomplete_generic_handoff_numbers_do_not_call_model(self):
        provider=SimpleNamespace(complete=AsyncMock(return_value=ChatCompletionMessage(
            role='assistant',content="What's your phone number?")))
        bot=Assistant(provider,dry_run=True)
        await bot.reply('Demo Guest')
        for invalid in ['7012345','0312345','123']:
            self.assertEqual(await bot.reply(invalid),PHONE_ERRORS['en'])
        provider.complete.assert_awaited_once()
        self.assertFalse(bot.saved)

    async def test_invalid_save_tool_preserves_other_details_and_blocks_old_review(self):
        bot=Assistant(None,dry_run=True)
        bot.execute('save_booking',booking())
        result=bot.execute('save_booking',{**booking(),'phone':'0312345'})
        self.assertEqual(result['field'],'phone')
        self.assertIsNone(bot.pending)
        self.assertIn('phone',bot.draft_errors)
        self.assertEqual(bot.booking_draft['name'],'Demo Guest')
        self.assertEqual(await bot.reply('yes'),PHONE_ERRORS['en'])
        await bot.reply('70000002')
        self.assertEqual(bot.pending['phone'],'+96170000002')

    async def test_booking_tool_rejection_has_fixed_reply(self):
        call=ChatCompletionMessage(role='assistant',tool_calls=[{'id':'phone','type':'function','function':{
            'name':'update_booking_draft','arguments':'{"phone":"7012345"}'}}])
        bot=Assistant(SimpleNamespace(complete=AsyncMock(return_value=call)),dry_run=True)
        bot.execute('update_booking_draft',{'name':'Demo Guest'})
        self.assertEqual(await bot.reply('My number needs checking'),PHONE_ERRORS['en'])
        self.assertIsNone(bot.pending)

    async def test_saved_booking_has_e164_and_review_has_national_format(self):
        with tempfile.TemporaryDirectory() as directory:
            bot=Assistant(None,Storage(directory,local_only=True))
            bot.execute('save_booking',{**booking(),'phone':'(70) 000.001'})
            self.assertIn('Phone: 70 000 001',bot.notice())
            await bot.confirm()
            self.assertEqual(bot.last_booking['phone'],'+96170000001')
            with (Path(directory)/'bookings.csv').open(encoding='utf-8-sig',newline='') as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(rows[0]['phone'],safe_cell('+96170000001'))

    def test_native_arabizi_lines_exactly_and_rejected_words_absent(self):
        self.assertEqual(SAFETY['az'],'Bas lal ma3loumet, bel matbakh fi mkasarat, gluten, 7alib w sesame, fa ma fina nadmanlak 100% ennu ma ela 3awared.')
        self.assertEqual(CALLBACK_OFFER['az'],'Baddak el team yotoslo fik abel ma totlob?')
        root=Path(__file__).resolve().parents[1]
        for name in ['reply_policy.py','assistant.py','knowledge/arabizi_examples.md']:
            text=(root/name).read_text('utf-8')
            self.assertNotIn('yet'+'salo',text)
            self.assertNotIn('ma fi '+'atar',text)
