import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from openai.types.chat import ChatCompletionMessage
from assistant import Assistant, system_prompt
from booking_state import QUESTIONS, LANGUAGE
from reply_policy import SAFETY, CALLBACK_OFFER, finalize_reply, local_wording
from storage import Storage

CASES = {
    'en': ("I'm allergic to nuts. Is hummus safe?", "Hummus has no nuts but contains sesame.", "yes please"),
    'fr': ("Je suis allergique aux noix. Le hummus est sans danger ?", "Le hummus contient du sésame, pas de noix.", "oui"),
    'ar': ("عندي حساسية من المكسرات، الحمص آمن؟", "الحمص ما فيه مكسرات، بس فيه سمسم.", "أكيد"),
    'az': ("3ande 7asasiye men nuts, el hummus safe?", "El hummus ma fi nuts, bas fi sesame.", "eh"),
}


class AllergyConsentTests(unittest.IsolatedAsyncioTestCase):
    async def test_offer_accept_name_phone_handoff_in_four_languages(self):
        for lang,(question,dish,accept) in CASES.items():
            with self.subTest(language=lang),tempfile.TemporaryDirectory() as directory:
                provider=SimpleNamespace(complete=AsyncMock(return_value=ChatCompletionMessage(role='assistant',content=dish+'\n'+QUESTIONS[lang][0])))
                bot=Assistant(provider,Storage(directory,local_only=True))
                first=await bot.reply(question)
                self.assertEqual(first,dish+'\n\n'+SAFETY[lang]+'\n\n'+CALLBACK_OFFER[lang])
                self.assertEqual(first.count(SAFETY[lang]),1)
                self.assertNotIn(QUESTIONS[lang][0],first)
                self.assertFalse((Path(directory)/'handoffs.csv').exists())
                self.assertEqual(await bot.reply(accept),QUESTIONS[lang][0])
                self.assertEqual(await bot.reply('سامي' if lang=='ar' else 'Demo Guest'),QUESTIONS[lang][1])
                self.assertFalse((Path(directory)/'handoffs.csv').exists())
                await bot.reply('70000001')
                with (Path(directory)/'handoffs.csv').open(encoding='utf-8-sig',newline='') as stream:
                    rows=list(csv.DictReader(stream))
                self.assertEqual(len(rows),1)
                self.assertEqual(rows[0]['phone'],'70000001')
                self.assertTrue(any(t['name']=='request_handoff' and t['result'].get('ok') for t in bot.captured_tools))
                provider.complete.assert_awaited_once()
                self.assertIsNone(bot.allergy_callback)

    async def test_acceptance_variants_keep_language(self):
        for accept in ['yes','eh','oui','akid','ايه','أكيد','yes! 👍','eh tamam']:
            bot=Assistant(None,dry_run=True)
            bot.language='az'
            bot.allergy_offer_pending=True
            self.assertEqual(await bot.reply(accept),QUESTIONS['az'][0],accept)

    async def test_decline_and_topic_change_do_not_collect_or_save(self):
        for lang,(question,dish,_) in CASES.items():
            with self.subTest(language=lang):
                decline={'en':'no','fr':'non','ar':'لا','az':'la2'}[lang]
                provider=SimpleNamespace(complete=AsyncMock(side_effect=[
                    ChatCompletionMessage(role='assistant',content=dish),
                    ChatCompletionMessage(role='assistant',content='Parking information.')]))
                bot=Assistant(provider,dry_run=True)
                await bot.reply(question)
                answer=await bot.reply(decline)
                self.assertNotIn(QUESTIONS[lang][0],answer)
                self.assertFalse(bot.allergy_offer_pending)
                self.assertIsNone(bot.allergy_callback)
                self.assertEqual(await bot.reply('Is there parking?'),'Parking information.')
                self.assertFalse(bot.saved)
        for accepted in [False,True]:
            provider=SimpleNamespace(complete=AsyncMock(side_effect=[
                ChatCompletionMessage(role='assistant',content='Hummus contains sesame.'),
                ChatCompletionMessage(role='assistant',content='Parking information.')]))
            bot=Assistant(provider,dry_run=True)
            await bot.reply(CASES['en'][0])
            if accepted:
                await bot.reply('yes')
            self.assertEqual(await bot.reply('Is there parking?'),'Parking information.')
            self.assertFalse(bot.allergy_offer_pending)
            self.assertIsNone(bot.allergy_callback)
            self.assertFalse(bot.allergy_callback_consent)
            self.assertFalse(bot.saved)

    async def test_unconsented_model_handoff_and_contact_tools_are_blocked(self):
        call=ChatCompletionMessage(role='assistant',tool_calls=[{
            'id':'handoff','type':'function','function':{'name':'request_handoff','arguments':json.dumps(
                {'name':'Demo Guest','phone':'70000001','reason':'Allergy concern'})}}])
        provider=SimpleNamespace(complete=AsyncMock(side_effect=[call,
            ChatCompletionMessage(role='assistant',content='Hummus contains sesame. What is your name?')]))
        bot=Assistant(provider,dry_run=True)
        answer=await bot.reply(CASES['en'][0])
        self.assertFalse(bot.saved)
        self.assertIn('error',bot.captured_tools[0]['result'])
        self.assertNotIn('your name',answer)
        self.assertTrue(answer.endswith(CALLBACK_OFFER['en']))
        bot._allergy_turn=False
        self.assertIn('error',bot.execute('request_handoff',{'name':'Demo Guest','phone':'70000001','reason':'Allergy concern'}))
        bot._allergy_turn=True
        self.assertIn('error',bot.execute('update_booking_draft',{'name':'Demo Guest'}))
        self.assertEqual(bot.booking_draft,{})

    async def test_invalid_callback_phone_does_not_save_and_cancellation_stops_collection(self):
        bot=Assistant(None,dry_run=True)
        bot.allergy_offer_pending=True
        await bot.reply('yes')
        await bot.reply('Demo Guest')
        self.assertEqual(await bot.reply('123'),QUESTIONS['en'][1])
        self.assertFalse(bot.saved)
        await bot.reply('cancel')
        self.assertIsNone(bot.allergy_callback)
        self.assertFalse(bot.allergy_callback_consent)

    def test_safety_order_deduplication_and_no_name_for_each_language(self):
        for lang,(_,dish,_) in CASES.items():
            raw=SAFETY[lang]+'\n'+dish+'\n'+SAFETY[lang]+'\n'+CALLBACK_OFFER[lang]+'\n'+QUESTIONS[lang][0]
            reply=finalize_reply('allergy',raw,lang)
            self.assertEqual(reply,dish+'\n\n'+SAFETY[lang]+'\n\n'+CALLBACK_OFFER[lang])
            self.assertEqual(finalize_reply('allergy',reply,lang),reply)

    async def test_word_replacements_are_whole_word_and_do_not_regenerate(self):
        raw='طماطم، ثوم، مخلل، الحين. طماطمية وثوم.'
        provider=SimpleNamespace(complete=AsyncMock(return_value=ChatCompletionMessage(role='assistant',content=raw)))
        bot=Assistant(provider,dry_run=True)
        self.assertEqual(await bot.reply('شو المكونات؟'),'بندورة، توم، كبيس، هلّق. طماطمية وثوم.')
        provider.complete.assert_awaited_once()
        self.assertEqual(local_wording('ghir GHIR kharej men Beirut ma fi wala shi aghir ghira','az'),
                         'gher gher bara Beirut ma fi shi aghir ghira')
        self.assertEqual(local_wording('ghir','en'),'ghir')

    async def test_other_dialect_words_still_regenerate_once(self):
        provider=SimpleNamespace(complete=AsyncMock(side_effect=[
            ChatCompletionMessage(role='assistant',content='تقدر تطلب طماطم.'),
            ChatCompletionMessage(role='assistant',content='فيك تطلب بندورة.')]))
        bot=Assistant(provider,dry_run=True)
        self.assertEqual(await bot.reply('شو المكونات؟'),'فيك تطلب بندورة.')
        self.assertEqual(provider.complete.await_count,2)

    def test_local_dates_in_cards_replies_and_prompt(self):
        bot=Assistant(None,dry_run=True)
        bot.pending=dict(name='Demo Guest',phone='70000001',date='2026-10-31',time='21:00',party_size=2,seating='terrace')
        bot.language='az'
        self.assertIn('Sabet 31 October',bot.notice())
        self.assertNotIn('Teshreen',bot.notice())
        bot.language='ar'
        self.assertIn('السبت 31/10',bot.notice())
        self.assertNotIn('تشرين',bot.notice())
        self.assertEqual(local_wording('Sabet 31 Teshreen awwal','az'),'Sabet 31 October')
        self.assertEqual(local_wording('السبت ٣١ تشرين الأول','ar'),'السبت 31/10')
        self.assertEqual(local_wording('السبت October 31','ar'),'السبت 31/10')
        self.assertEqual(local_wording('Esem: Ab', 'az'), 'Esem: Ab')
        self.assertEqual(local_wording('الاسم: أيار', 'ar'), 'الاسم: أيار')
        prompt=system_prompt(LANGUAGE['az'])
        self.assertIn('English month names in Arabizi',prompt)
        for opener in ['Be3tezer,','بعتذر،','Sorry,','Désolé,']:
            self.assertIn(opener,prompt)
