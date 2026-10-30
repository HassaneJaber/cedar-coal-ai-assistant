import asyncio
import copy
import json
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from openai.types.chat import ChatCompletionMessage
from assistant import Assistant, KNOWLEDGE, BEIRUT, now
from booking_state import LANGUAGE, QUESTIONS, REQUIRED, detect_language, next_field, forbidden_words, arabic_script, resolve_time, asked_fields
from conversation_eval import SCENARIOS, run_case


class StateAwareFake:
    def __init__(self):
        self.calls=[]
    async def complete(self,messages,tools=None):
        self.calls.append(copy.deepcopy(messages))
        state=next(m['content'] for m in messages if m['role']=='system' and m['content'].startswith('Booking so far:'))
        draft=json.loads(state.split('Booking so far: ',1)[1].split('. Still missing:',1)[0])
        field=next_field(draft,{})
        system=messages[0]['content']
        language=next(lang for lang,instruction in LANGUAGE.items() if instruction in system)
        return ChatCompletionMessage(role='assistant',content=QUESTIONS[language][REQUIRED.index(field)])


class BookingStateTests(unittest.IsolatedAsyncioTestCase):
    async def test_three_scripted_conversations_use_real_csv_and_corrections(self):
        for scenario in SCENARIOS:
            with self.subTest(conversation=scenario['id']):
                provider=StateAwareFake()
                result=await run_case(provider,scenario)
                self.assertEqual(result['verdict'],'PASS',result['errors'])
                self.assertEqual(result['saved_csv_rows'],1)
                self.assertEqual(provider.calls, [])
                self.assertTrue(any(t['name']=='update_booking_draft' for t in result['tool_calls']))

    async def test_conversation_reservation_is_once_and_separate_from_full_run(self):
        from conversation_eval import reserve_conversation_run
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'tests').mkdir()
            (root/'data').mkdir()
            day=now().date().isoformat()
            (root/'tests/full_eval_runs.json').write_text(json.dumps([day]),encoding='utf-8')
            with patch('conversation_eval.ROOT',root):
                reserve_conversation_run()
                with self.assertRaises(ValueError):
                    reserve_conversation_run()
            self.assertEqual(json.loads((root/'tests/full_eval_runs.json').read_text()),[day])

    async def test_partial_validation_errors_preserve_other_fields(self):
        bot=Assistant(None,dry_run=True)
        result=bot.execute('update_booking_draft',{'name':'Karim','phone':'bad','party_size':11,'date':'1900-01-01','time':'9am','seating':'roof'})
        self.assertEqual(result['accepted'],{'name':'Karim'})
        self.assertEqual(set(result['errors']),{'phone','party_size','date','time','seating'})
        tomorrow=(now().date()+timedelta(days=1)).isoformat()
        result=bot.execute('update_booking_draft',{'phone':'70000001','party_size':3,'date':tomorrow,'time':'9','seating':'terrace'})
        self.assertEqual(result['errors'],{})
        self.assertEqual(result['draft']['time'],'21:00')
        self.assertEqual(result['draft']['name'],'Karim')
        bot.execute('update_booking_draft',{'phone':'invalid'})
        self.assertEqual(bot.booking_draft['phone'],'70000001')
        self.assertIn('error',bot.execute('save_booking',dict(bot.booking_draft)))

    async def test_date_change_revalidates_kitchen_cutoff(self):
        day=now().date()+timedelta(days=1)
        while day.weekday()!=0:
            day+=timedelta(days=1)
        bot=Assistant(None,dry_run=True)
        bot.execute('update_booking_draft',{'time':'23:00'})
        result=bot.execute('update_booking_draft',{'date':day.isoformat()})
        self.assertIn('time',result['errors'])
        self.assertEqual(bot.execute('update_booking_draft',{'time':'22:30'})['errors'],{})

    async def test_pm_inference_does_not_override_explicit_am(self):
        current=datetime(2026,10,30,10,0,tzinfo=BEIRUT)
        for hour in range(1,12):
            self.assertEqual(resolve_time(str(hour),'2026-10-31',current,KNOWLEDGE),f'{hour+12:02d}:00')
        self.assertEqual(resolve_time('9:15','2026-10-31',current,KNOWLEDGE),'21:15')
        with self.assertRaises(ValueError):
            resolve_time('9am','2026-10-31',current,KNOWLEDGE)

    async def test_language_stays_for_names_phones_dates_and_acknowledgements(self):
        for text in ['Karim','Maya','سامي','70000001','+96171111111','2026-10-31','السبت','ok','yes please']:
            self.assertIsNone(detect_language(text),text)
        for text in ['badde e7joz tawle','3al 9','ra2m','khames ashkhas','eh tamam','la2','bokra']:
            self.assertEqual(detect_language(text),'az',text)
        self.assertEqual(detect_language('table for 2 tomorrow at 8pm inside'),'en')
        provider=SimpleNamespace(complete=AsyncMock(return_value=ChatCompletionMessage(role='assistant',content='OK')))
        bot=Assistant(provider,dry_run=True)
        await bot.reply('fi parking?')
        for text in ['Karim','70000001','ok','2026-10-31']:
            await bot.reply(text)
            self.assertEqual(bot.language,'az')
            self.assertIn(LANGUAGE['az'],provider.complete.await_args.args[0][0]['content'])
        await bot.reply('What time do you close?')
        self.assertEqual(bot.language,'en')

    async def test_reask_and_wrong_script_regenerate_once_then_use_safe_step(self):
        create=AsyncMock(return_value=ChatCompletionMessage(role='assistant',content='شو اسمك؟'))
        bot=Assistant(SimpleNamespace(complete=create),dry_run=True)
        bot.execute('update_booking_draft',{'name':'Karim'})
        bot.language='az'
        reply=await bot.reply('fi parking?')
        self.assertEqual(create.await_count,2)
        self.assertEqual(reply,'Shu ra2m el telefone?')
        self.assertFalse(arabic_script(reply))
        self.assertIn('Already collected name',create.await_args.args[0][-1]['content'])

    async def test_arabic_forbidden_words_regenerate_and_laysh_is_allowed(self):
        for word in ['كام','عشان','الحين','تقدر','ترسلو','طماطم']:
            self.assertTrue(forbidden_words(word))
        self.assertFalse(forbidden_words('ليش'))
        create=AsyncMock(side_effect=[ChatCompletionMessage(role='assistant',content='تقدر تعطيني اسمك؟'),ChatCompletionMessage(role='assistant',content='شو اسمك؟')])
        bot=Assistant(SimpleNamespace(complete=create),dry_run=True)
        reply=await bot.reply('فيك تخبرني عن المنيو؟')
        self.assertEqual(reply,'شو اسمك؟')
        self.assertEqual(create.await_count,2)

    async def test_draft_survives_history_truncation_and_sessions_are_isolated(self):
        fake=SimpleNamespace(complete=AsyncMock(return_value=ChatCompletionMessage(role='assistant',content='What phone number?')))
        bot=Assistant(fake,dry_run=True)
        bot.execute('update_booking_draft',{'name':'Karim','party_size':3})
        for _ in range(14):
            await bot.reply('ok')
        self.assertEqual(len(bot.history),20)
        self.assertEqual(bot.booking_draft['name'],'Karim')
        self.assertEqual(Assistant(None,dry_run=True).booking_draft,{})
        state=bot.booking_context()
        self.assertIn('Karim',state)

    async def test_model_tool_can_update_subset_and_save_uses_authoritative_draft(self):
        bot=Assistant(None,dry_run=True)
        bot.execute('update_booking_draft',{'name':'Maya','phone':'+96171111111','date':'tomorrow','time':'8pm','party_size':2})
        call=ChatCompletionMessage(role='assistant',tool_calls=[{'id':'draft','type':'function','function':{'name':'update_booking_draft','arguments':'{"seating":"terrace"}'}}])
        create=AsyncMock(return_value=call)
        bot.provider=SimpleNamespace(complete=create)
        review=await bot.reply('the outside seating please')
        self.assertIsNotNone(bot.pending)
        self.assertEqual(bot.pending['seating'],'terrace')
        self.assertIn('Maya',review)
        self.assertIn('error',bot.execute('save_booking',{**bot.booking_draft,'phone':'70000000'}))


if __name__=='__main__':
    unittest.main()
