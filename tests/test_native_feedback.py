"""Offline regression checks for the native-speaker phone-test feedback."""
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from openai.types.chat import ChatCompletionMessage

from assistant import Assistant, BEIRUT, KNOWLEDGE, ARABIZI_EXAMPLES, system_prompt
from booking_state import asked_fields, extract_details, resolve_date, update_draft
from reply_policy import CALLBACK_OFFER, SAFETY, finalize_reply, strip_unneeded_apology, local_wording


CURRENT = datetime(2026, 10, 30, 10, 0, tzinfo=BEIRUT)


class NativeFeedbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_booking_openers_differ_from_callback_names(self):
        cases = {
            'en': ('book a table', 'Sure! Under what name?', "What's your name?", 'yes'),
            'ar': ('بدي احجز طاولة', 'أكيد، بإسم مين؟', 'شو اسمك؟', 'أكيد'),
            'az': ('badde e7joz tawle', 'Akeed, bi esm min?', 'Shu esmak?', 'eh'),
            'fr': ('je veux reserver une table', 'Bien sûr ! À quel nom ?', 'Quel est votre nom ?', 'oui'),
        }
        for lang, (booking, opener, callback_name, accept) in cases.items():
            with self.subTest(language=lang):
                provider=SimpleNamespace(complete=AsyncMock(side_effect=AssertionError('Unexpected model call')))
                bot=Assistant(provider,dry_run=True)
                self.assertEqual(await bot.reply(booking),opener)
                self.assertEqual(asked_fields(opener),{'name'})
                callback=Assistant(provider,dry_run=True)
                callback.language=lang
                callback.allergy_offer_pending=True
                self.assertEqual(await callback.reply(accept),callback_name)
                provider.complete.assert_not_awaited()

    async def test_exact_reported_booking_captures_tomorrow_and_nine_pm(self):
        provider=SimpleNamespace(complete=AsyncMock(side_effect=AssertionError('Unexpected model call')))
        bot=Assistant(provider,dry_run=True)
        with patch('assistant.now',return_value=CURRENT):
            reply=await bot.reply('Badde e7joz tawle la 2 bkra 3al 9')
        self.assertEqual(reply,'Akeed, bi esm min?')
        self.assertEqual(bot.language,'az')
        self.assertEqual(bot.booking_draft['date'],'2026-10-31')
        self.assertEqual(bot.booking_draft['time'],'21:00')
        self.assertEqual(bot.booking_draft['party_size'],2)
        self.assertEqual(bot.draft_errors,{})
        provider.complete.assert_not_awaited()

    def test_relative_spellings_in_full_booking_messages(self):
        cases={
            1: ['bokra','bkra','bukra','boukra','bokrah','بكرا','بكرة'],
            2: ['ba3d bokra','ba3ed bokra','ba3d bkra','بعد بكرا','بعد بكرة'],
            0: ['lyom','el yom','elyom','l yom','el leile','lleile','اليوم','الليلة'],
        }
        for offset,phrases in cases.items():
            for phrase in phrases:
                with self.subTest(phrase=phrase):
                    details=extract_details(f'badde e7joz tawle la 2 {phrase} 3al 9',{},CURRENT)
                    draft={}
                    _,errors=update_draft(draft,details,CURRENT,KNOWLEDGE)
                    self.assertEqual(errors,{})
                    self.assertEqual(draft['date'],(CURRENT.date()+timedelta(days=offset)).isoformat())
                    self.assertEqual(draft['party_size'],2)
                    self.assertEqual(draft['time'],'21:00')

    def test_weekday_spellings_choose_next_occurrence(self):
        cases=[
            ['tanen','itnen','el tanen','الاثنين','التنين'],
            ['talata','tlata','الثلاثاء','التلاتا'],
            ['arb3a','arbe3a','arba3a','الأربعاء','الأربعا'],
            ['khamis','5amis','الخميس'],
            ['jem3a','jom3a','jam3a','الجمعة'],
            ['sabet','sabt','السبت'],
            ['a7ad','ahad','el a7ad','الأحد'],
        ]
        for weekday,phrases in enumerate(cases):
            expected=(CURRENT.date()+timedelta(days=(weekday-CURRENT.weekday())%7)).isoformat()
            for phrase in phrases:
                with self.subTest(phrase=phrase):
                    details=extract_details(f'badde e7joz tawle la 2 {phrase} 3al 9',{},CURRENT)
                    self.assertEqual(resolve_date(details['date'],CURRENT,KNOWLEDGE),expected)

    def test_weekday_today_only_when_still_bookable(self):
        for day,weekday,last_hour,last_minute in [(26,'tanen',22,30),(30,'jem3a',23,30)]:
            current=CURRENT.replace(day=day,hour=last_hour,minute=last_minute)
            next_week=(current.date()+timedelta(days=7)).isoformat()
            self.assertEqual(resolve_date(weekday,current-timedelta(minutes=1),KNOWLEDGE),current.date().isoformat())
            self.assertEqual(resolve_date(weekday,current,KNOWLEDGE),next_week)
            self.assertEqual(resolve_date(weekday,current+timedelta(minutes=1),KNOWLEDGE),next_week)
        evening=CURRENT.replace(hour=22)
        self.assertEqual(resolve_date('jom3a',evening,KNOWLEDGE,time_value='9'),'2026-11-06')
        self.assertEqual(resolve_date('jom3a',evening,KNOWLEDGE,time_value='23:00'),'2026-10-30')

    def test_relative_dates_use_beirut_at_midnight_and_match_whole_words(self):
        utc=datetime(2026,10,30,23,30,tzinfo=timezone.utc)
        self.assertEqual(resolve_date('bkra',utc,KNOWLEDGE),'2026-11-01')
        self.assertEqual(resolve_date('lyom',utc,KNOWLEDGE),'2026-10-31')
        for phrase in ['bokram','sabten','aljom3at']:
            self.assertNotIn('date',extract_details(phrase,{'name':'Demo Guest'},CURRENT))
        details=extract_details('badde tawle la arba3a bokra 3al 9',{},CURRENT)
        self.assertEqual(details['party_size'],4)
        self.assertEqual(resolve_date(details['date'],CURRENT,KNOWLEDGE),'2026-10-31')

    def test_apology_removed_for_ingredient_facts_and_positive_answers(self):
        cases=[
            ('Be3tezer, el hummus ma fiyyo mokasarat, bas fiyyo sesame.','el hummus ma fiyyo mokasarat, bas fiyyo sesame.'),
            ('Be3tezer, el hummus ma fi mkasarat.','el hummus ma fi mkasarat.'),
            ('بعتذر، الحمص ما فيه مكسرات، بس فيه سمسم.','الحمص ما فيه مكسرات، بس فيه سمسم.'),
            ("Sorry, hummus has no nuts but contains sesame.","hummus has no nuts but contains sesame."),
            ("Sorry, we don't have nuts in the hummus.","we don't have nuts in the hummus."),
            ('Désolé, le hummus ne contient pas de noix, mais contient du sésame.','le hummus ne contient pas de noix, mais contient du sésame.'),
            ('Sorry, yes, delivery is available.','yes, delivery is available.'),
            ('Sorry. Hummus contains sesame.','Hummus contains sesame.'),
        ]
        for original,expected in cases:
            with self.subTest(original=original):
                self.assertEqual(strip_unneeded_apology(original),expected)

    def test_real_refusals_keep_polite_opener(self):
        for reply in [
            'Be3tezer, ma fi arguileh 3anna.',
            'Be3tezer, ma mnwasel bara Beirut.',
            'بعتذر، ما في أرغيلة عنّا.',
            'بعتذر، ما منوصل برا بيروت.',
            "Sorry, we don't deliver outside Beirut.",
            'Sorry, delivery is unavailable.',
            'Sorry, we are closed now.',
            'Désolé, nous ne livrons pas hors de Beyrouth.',
            "Désolé, nous n'avons pas de chicha.",
        ]:
            self.assertEqual(strip_unneeded_apology(reply),reply)
        # A dish fact followed by a genuine refusal must not hide that refusal.
        reply='Sorry, hummus has no nuts, but we cannot accept this booking.'
        self.assertEqual(strip_unneeded_apology(reply),reply)

    async def test_allergy_conjunction_variants_keep_safe_body_and_one_offer(self):
        answer=('El hummus ma fiyyo mkasarat, bas fiyyo 7ommos w tahini (sesame). '
                'Iza 3andak 7asasiye 3al hummus kamen, a7san ma totlobo.')
        self.assertIn(answer,ARABIZI_EXAMPLES)
        self.assertEqual(CALLBACK_OFFER['az'],'Baddak el team yetwasalo ma3ak abel ma totlob?')
        for conjunction in ['wl','wel','w el','w l','el','l']:
            question=f'3ande 7asasiye 3al mkasarat {conjunction} hummus ok?'
            with self.subTest(conjunction=conjunction):
                provider=SimpleNamespace(complete=AsyncMock(return_value=ChatCompletionMessage(
                    role='assistant',content='Be3tezer, '+answer+'\n'+SAFETY['az']+'\n'+CALLBACK_OFFER['az'])))
                bot=Assistant(provider,dry_run=True)
                reply=await bot.reply(question)
                self.assertTrue(" ".join(reply.split()).startswith(answer))
                self.assertEqual(reply.count(SAFETY['az']),1)
                self.assertEqual(reply.count(CALLBACK_OFFER['az']),1)
                self.assertEqual(reply.count('?'),1)
                self.assertNotIn('Shu esmak?',reply)
                self.assertEqual(bot.captured_tools,[])
                self.assertTrue(bot.allergy_offer_pending)
                self.assertEqual(finalize_reply(question,reply,'az'),reply)
                prompt=provider.complete.await_args.args[0][0]['content']
                self.assertIn('BOTH readings',prompt)
                self.assertIn('never say or imply the dish is safe or recommended',prompt)
                self.assertIn('ONE short clarifying question instead of guessing',prompt)

    def test_generated_callback_verb_is_corrected_without_a_model_retry(self):
        old_verb='yot'+'oslo'
        for source in [f'Baddak el team {old_verb} fik abel ma totlob?',
                       f'Baddak el team {old_verb} ma3ak abel ma totlob?']:
            self.assertEqual(local_wording(source,'az'),CALLBACK_OFFER['az'])
            self.assertNotIn(old_verb,finalize_reply('3ande 7asasiye',source,'az'))

    def test_unclear_allergy_can_ask_one_clarification_before_fixed_lines(self):
        question='3ande 7asasiye, wl hayda ok?'
        clarification='Ayya taba2 w ayya 7asasiye 3andak?'
        reply=finalize_reply(question,clarification,'az')
        self.assertEqual(reply,clarification+'\n\n'+SAFETY['az']+'\n\n'+CALLBACK_OFFER['az'])
        self.assertIn('genuinely unclear',system_prompt('az',question))


if __name__ == '__main__':
    unittest.main()
