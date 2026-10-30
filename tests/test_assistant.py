import csv
import json
import tempfile
import unittest
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
import httpx
from openai import RateLimitError
from assistant import Provider, one_contact_question, CONFIRMATIONS, CANCELLATIONS
from datetime import timedelta
from pathlib import Path
from openai.types.chat import ChatCompletionMessage
from assistant import Assistant, now, validate_booking
from storage import Storage

def booking():
    return dict(name="Demo Guest", phone="+96170000002",
                date=(now().date() + timedelta(days=1)).isoformat(),
                time="20:30", party_size=4, seating="terrace", notes="")

class ScriptedProvider:
    """A fake provider exercises the same orchestration without a network or API key."""
    def __init__(self):
        self.calls = 0
    async def complete(self, messages, tools=None):
        self.calls += 1
        if self.calls == 1:
            return ChatCompletionMessage(role="assistant", content="What name should I use?")
        if self.calls == 2:
            return ChatCompletionMessage(role="assistant", content="What phone number can we reach you on?")
        if self.calls == 3:
            return ChatCompletionMessage(role="assistant", tool_calls=[{
                "id":"demo_call", "type":"function",
                "function":{"name":"save_booking","arguments":json.dumps(booking())}}])
        return ChatCompletionMessage(role="assistant", content="Demo Guest, 4 people tomorrow at 20:30 on the terrace, +96170000002. Confirm?")

class Tests(unittest.IsolatedAsyncioTestCase):
    async def test_full_booking_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            bot = Assistant(ScriptedProvider(), Storage(directory, local_only=True))
            await bot.reply("Terrace for 4 tomorrow at 20:30")
            await bot.reply("Demo Guest")
            summary = await bot.reply("+96170000002")
            self.assertIn("Confirm to save", summary)
            self.assertNotIn("has been saved", summary)
            path = Path(directory) / "bookings.csv"
            self.assertFalse(path.exists(), "Must not save before confirmation")
            self.assertIsNotNone(bot.pending)
            self.assertEqual(await bot.reply("yes"), "Your booking is confirmed! Thank you.")
            with path.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["party_size"], "4")
            self.assertIn("No booking", await bot.confirm())

    async def test_dry_run_never_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            bot = Assistant(ScriptedProvider(), Storage(directory, local_only=True), dry_run=True)
            bot.execute("save_booking", booking())
            await bot.confirm()
            bot.execute("request_handoff", dict(name="Demo", phone="70000002", reason="Complaint"))
            self.assertEqual(list(Path(directory).iterdir()), [])

    async def test_correction_invalidates_confirmation(self):
        bot = Assistant(ScriptedProvider(), dry_run=True)
        bot.execute("save_booking", booking())
        await bot.reply("Actually change it")
        self.assertIsNone(bot.pending)

    async def test_handoff_and_deduplication(self):
        with tempfile.TemporaryDirectory() as directory:
            bot = Assistant(None, Storage(directory, local_only=True))
            bot.allergy_callback_consent = True
            values = dict(name="Demo", phone="70000002", reason="Severe allergy")
            self.assertTrue(bot.execute("request_handoff", values)["ok"])
            self.assertTrue(bot.execute("request_handoff", values)["already_saved"])
            with (Path(directory)/"handoffs.csv").open(encoding="utf-8-sig") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 1)

    async def test_validation_boundaries(self):
        for update in [dict(party_size=11), dict(party_size=0), dict(party_size=True),
                       dict(seating="roof"), dict(time="00:00"), dict(time="11:59"),
                       dict(date=(now().date()+timedelta(days=31)).isoformat()),
                       dict(date=(now().date()-timedelta(days=1)).isoformat()),
                       dict(phone="x")]:
            with self.subTest(update=update), self.assertRaises(ValueError):
                validate_booking({**booking(), **update})
        self.assertEqual(validate_booking(booking()), booking())
        edge = {**booking(), "date": (now().date()+timedelta(days=30)).isoformat()}
        validate_booking(edge)

    async def test_weekday_and_weekend_kitchen_cutoffs(self):
        day = now().date()+timedelta(days=1)
        while day.weekday() != 0:
            day += timedelta(days=1)
        for offset, good, bad in [(0,"22:30","22:31"), (5,"23:30","23:31")]:
            values={**booking(), "date":(day+timedelta(days=offset)).isoformat()}
            validate_booking({**values,"time":good})
            with self.assertRaises(ValueError):
                validate_booking({**values,"time":bad})

    async def test_rate_limit_retries_then_friendly_error(self):
        provider = Provider.__new__(Provider)
        request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
        error = RateLimitError("Rate limited", response=httpx.Response(429, request=request), body=None)
        create = AsyncMock(side_effect=error)
        provider.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        provider.lock = asyncio.Lock()
        provider.last_call = 0
        provider.interval_seconds = 0
        provider.model = "openai/gpt-oss-120b"
        provider.reasoning_effort = "low"
        provider.usage = {}
        provider.fallback_model = ""
        provider.daily_exhausted = set()
        with patch("assistant.asyncio.sleep", new_callable=AsyncMock):
            with self.assertRaisesRegex(RuntimeError, "temporarily unavailable"):
                await provider.complete([])
        self.assertEqual(create.await_count, 4)

    async def test_single_contact_question(self):
        self.assertEqual(one_contact_question("May I have your name and phone number?"), "May I have your name?")
        self.assertEqual(one_contact_question("Votre nom et votre numéro de téléphone ?"), "Votre nom ?")
        self.assertEqual(one_contact_question("شو اسمك ورقم تلفونك؟"), "شو اسمك؟")
        self.assertEqual(one_contact_question("Phone: 70000002. Confirm?"), "Phone: 70000002. Confirm?")

    async def test_malformed_output_is_retried(self):
        provider = Provider.__new__(Provider)
        bad = SimpleNamespace(choices=[SimpleNamespace(finish_reason="length", message=SimpleNamespace(content="ـ"*100))])
        good = SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="Delivery costs $3."))])
        create = AsyncMock(side_effect=[bad, good])
        provider.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        provider.lock = asyncio.Lock()
        provider.last_call = 0
        provider.interval_seconds = 0
        provider.model = "openai/gpt-oss-120b"
        provider.reasoning_effort = "low"
        provider.usage = {}
        provider.fallback_model = ""
        provider.daily_exhausted = set()
        with patch("assistant.asyncio.sleep", new_callable=AsyncMock):
            result = await provider.complete([])
        self.assertEqual(result.content, "Delivery costs $3.")
        self.assertEqual(create.await_count, 2)

    async def test_complaint_explains_callback_before_contact(self):
        bot = Assistant(ScriptedProvider(), dry_run=True)
        reply = await bot.reply("I want to complain about my order")
        self.assertIn("sorry", reply)
        self.assertIn("team can call you back", reply)
        self.assertIn("name", reply)
        self.assertNotIn("phone", reply)

    async def test_live_provider_requests_overlap(self):
        with patch.dict("os.environ", {"GROQ_API_KEY": "offline-test"}):
            provider = Provider()
        await provider.client.close()
        both_started = asyncio.Event()
        count = 0
        async def create(**kwargs):
            nonlocal count
            count += 1
            if count == 2:
                both_started.set()
            await asyncio.wait_for(both_started.wait(), 1)
            return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="OK"))])
        provider.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        await asyncio.gather(provider.complete([]), provider.complete([]))
        self.assertEqual(count, 2)

    async def test_daily_quota_switches_only_this_request(self):
        for phrase in ("tokens per day", "requests per day"):
            with patch.dict("os.environ", {"GROQ_API_KEY":"offline-test"}):
                provider = Provider(fallback_model="qwen/qwen3.8-27b")
            await provider.client.close()
            request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
            error = RateLimitError("Daily quota", response=httpx.Response(429,request=request),body={"message":"Rate limit reached on "+phrase})
            good = SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop",message=SimpleNamespace(content="OK"))],
                                   usage=SimpleNamespace(prompt_tokens=100,completion_tokens=5,total_tokens=105))
            create = AsyncMock(side_effect=[error,good])
            provider.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
            with self.assertLogs("assistant", level="WARNING"):
                await provider.complete([])
            self.assertEqual([c.kwargs["model"] for c in create.await_args_list], [provider.model,provider.fallback_model])
            self.assertEqual(provider.model, "openai/gpt-oss-120b")
            self.assertEqual(provider.usage[provider.fallback_model]["total_tokens"],105)
            self.assertEqual(create.await_args_list[1].kwargs["max_completion_tokens"],600)
            self.assertEqual(create.await_args_list[1].kwargs["reasoning_effort"], "none")
            self.assertEqual(create.await_args_list[1].kwargs["extra_body"], {"reasoning_format": "hidden"})

    async def test_natural_confirmations_and_cancellations_without_llm(self):
        for phrase in ("yes please", "yes!", "ok", "okay", "sure", "perfect", "correct", "go ahead", "confirm", "oui", "d'accord", "c'est bon", "eh", "ah", "akeed", "tamam", "ok tamam", "eh akid", "اي", "إيه", "أكيد", "تمام", "نعم", "موافق"):
            bot = Assistant(None, dry_run=True)
            bot.execute("save_booking",booking())
            await bot.reply(phrase + "!! ✅")
            self.assertIsNone(bot.pending, phrase)
            self.assertEqual(len(bot.saved),1,phrase)
        for phrase in ("no", "cancel", "la2", "non", "لا", "إلغاء"):
            bot = Assistant(None,dry_run=True)
            bot.execute("save_booking",booking())
            await bot.reply(phrase + " ❌")
            self.assertIsNone(bot.pending,phrase)
            self.assertEqual(len(bot.saved),0,phrase)

    async def test_unclear_reply_keeps_pending_and_change_keeps_fields(self):
        bot = Assistant(None,dry_run=True)
        bot.execute("save_booking",booking())
        await bot.reply("hmm 🤔")
        self.assertEqual(bot.pending,booking())
        captured = []
        class Capture:
            async def complete(self,messages,tools=None):
                captured.extend(messages)
                return ChatCompletionMessage(role="assistant",content="Which time?")
        bot.provider = Capture()
        await bot.reply("Actually change the time")
        self.assertIsNone(bot.pending)
        self.assertEqual(bot.booking_draft,booking())
        self.assertTrue(any("Demo Guest" in m.get("content","") and "Previously collected" in m.get("content","") for m in captured))

    async def test_review_is_localized_and_readable(self):
        bot = Assistant(None,dry_run=True)
        bot.pending = {**booking(),"date":"2026-10-30","time":"19:00"}
        english = bot.notice()
        self.assertIn("Friday 30 October",english)
        self.assertIn("7:00 pm",english)
        self.assertNotIn("Notes:",english)
        self.assertEqual(bot.button_labels(),("Confirm","Cancel"))
        bot.language = "ar"
        arabic = bot.notice()
        self.assertIn("الجمعة 30/10",arabic)
        self.assertIn("٧:٠٠ مساءً",arabic)
        self.assertNotIn("terrace",arabic)
        self.assertEqual(bot.button_labels(),("تأكيد","إلغاء"))

    async def test_missing_contact_not_saved(self):
        bot = Assistant(None, dry_run=True)
        self.assertIn("error",bot.execute("request_handoff",dict(name="",phone="",reason="Complaint")))
        self.assertIn("error",bot.execute("unknown",{}))

if __name__ == "__main__":
    unittest.main()
