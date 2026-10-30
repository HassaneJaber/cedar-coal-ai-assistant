import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from openai.types.chat import ChatCompletionMessage
from assistant import Assistant, Provider, strip_reasoning
from booking_guard import is_large_group_booking, LARGE_GROUP_REPLIES
from storage import Storage


class GuardTests(unittest.IsolatedAsyncioTestCase):
    async def test_multilingual_guard_never_calls_provider(self):
        examples = {
            "en": ["Can I book a table for 11 people?", "Book for eleven", "Reserve for fifteen guests on Friday", "Book a table for twenty-one people", "Book one hundred seats"],
            "ar": ["بدي احجز لـ١١ شخص", "بدي حجز لخمستعش شخص", "أريد حجز طاولة لأحد عشر شخصا", "بدي احجز لحدعش", "بدي احجز لواحد وعشرين شخص", "حجز لخمسة عشر شخص", "احجز لاثني عشر شخص"],
            "az": ["Baddi ehjoz la 11 ashkhas", "fiyye ehjoz la hda3ash ashkhas?", "badde e7joz la khamsta3ash", "baddi hajez la 7da3ash", "nehna 3eshreen badna tawle"],
            "fr": ["Je voudrais réserver pour 11 personnes", "Une table pour onze", "Je veux réserver pour quinze personnes", "Réserver pour vingt et un", "Une table pour quatre-vingt-dix-neuf personnes"],
        }
        for language, phrases in examples.items():
            for phrase in phrases:
                with self.subTest(phrase=phrase), tempfile.TemporaryDirectory() as directory:
                    provider = SimpleNamespace(complete=AsyncMock(side_effect=AssertionError("Guard must not call LLM")))
                    bot = Assistant(provider, Storage(directory, local_only=True))
                    reply = await bot.reply(phrase)
                    self.assertEqual(reply, LARGE_GROUP_REPLIES[language])
                    self.assertEqual(reply.count("?") + reply.count("؟"), 1)
                    provider.complete.assert_not_awaited()
                    self.assertEqual(bot.history[-1]["content"], reply)
                    self.assertEqual(bot.captured_tools, [])
                    self.assertEqual(list(Path(directory).iterdir()), [])

    async def test_small_groups_times_dates_and_faqs_do_not_trigger(self):
        for phrase in (
            "Book for 10 people", "Book for ten", "Book four seats at 15:00", "Reserve for 10 at 11 pm",
            "Book for 30 October", "Book for 30 octobre", "Book for 2026-11-15", "Book for 30/11",
            "Do you deliver for 15 dollars?", "Does the family meal feed fifteen people?", "How much for eleven people?",
            "Je voudrais réserver pour dix personnes", "حجز لعشرة اشخاص", "بدي احجز لـ١٠ اشخاص", "baddi ehjoz la 3ashra ashkhas",
            "Book for four, not fifteen people", "Book not for fifteen people", "Une table pour dix, pas quinze personnes",
        ):
            with self.subTest(phrase=phrase):
                self.assertFalse(is_large_group_booking(phrase))
        provider = SimpleNamespace(complete=AsyncMock(return_value=ChatCompletionMessage(role="assistant", content="What is your name?")))
        bot = Assistant(provider, dry_run=True)
        self.assertEqual(await bot.reply("Book for ten people"), "Sure! Under what name?")
        provider.complete.assert_not_awaited()

    async def test_name_followup_retains_group_context_and_handoff(self):
        provider = SimpleNamespace(complete=AsyncMock(side_effect=[
            ChatCompletionMessage(role="assistant", content="What phone number can the team reach you on?"),
            ChatCompletionMessage(role="assistant", tool_calls=[{"id":"handoff", "type":"function", "function":{
                "name":"request_handoff", "arguments":json.dumps({"name":"Demo Guest", "phone":"70000002", "reason":"Group of 15 on Friday"})}}]),
            ChatCompletionMessage(role="assistant", content="Your demo callback request has been recorded."),
        ]))
        with tempfile.TemporaryDirectory() as directory:
            bot = Assistant(provider, Storage(directory, local_only=True))
            await bot.reply("Can I book for 15 people on Friday?")
            reply = await bot.reply("Demo Guest")
            self.assertIn("phone", reply)
            self.assertNotIn("name", reply)
            messages = provider.complete.await_args.args[0]
            self.assertTrue(any("15 people on Friday" in m.get("content", "") for m in messages))
            self.assertTrue(any("limited to 10" in m.get("content", "") for m in messages))
            await bot.reply("70000002")
            self.assertEqual(bot.captured_tools[-1]["name"], "request_handoff")
            self.assertTrue((Path(directory)/"handoffs.csv").exists())
            self.assertFalse((Path(directory)/"bookings.csv").exists())


class ReasoningTests(unittest.IsolatedAsyncioTestCase):
    async def test_strip_reasoning_variants(self):
        for content, expected in [
            ("<think>private\nanalysis</think>Final answer", "Final answer"),
            ("<THINK>private</THINK>مرحبا", "مرحبا"),
            ("<think>a</think>Hello <think>b</think>there", "Hello there"),
            ("<think>a<think>b</think>c</think>Bonjour", "Bonjour"),
            ("<think>incomplete private analysis", ""),
            ("private analysis</think>Answer", "Answer"),
            ("Normal answer", "Normal answer"),
            (None, ""),
        ]:
            with self.subTest(content=content):
                self.assertEqual(strip_reasoning(content), expected)

    async def test_qwen_api_options_and_sanitized_output(self):
        # No real client/key/network: exercise Provider.complete with a fake SDK boundary.
        for tool_mode in (False, True):
            provider = Provider.__new__(Provider)
            provider.model = "qwen/qwen3.8-27b"
            provider.interval_seconds = 0
            provider.usage = {}
            provider.reasoning_effort = "low"
            message = ChatCompletionMessage(role="assistant", content="<think>private</think>Here is the answer.")
            if tool_mode:
                message.tool_calls = [{"id":"test", "type":"function", "function":{"name":"request_handoff", "arguments":"{}"}}]
            create = AsyncMock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")]))
            provider.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
            result = await provider.complete([], tools=[{"type":"function"}] if tool_mode else None)
            self.assertEqual(create.await_args.kwargs["reasoning_effort"], "none")
            self.assertEqual(create.await_args.kwargs["extra_body"], {"reasoning_format":"hidden"})
            self.assertEqual(create.await_args.kwargs["max_completion_tokens"], 600)
            self.assertEqual(result.content, "Here is the answer.")
            self.assertEqual(bool(result.tool_calls), tool_mode)

    async def test_assistant_never_displays_or_stores_think_blocks(self):
        for raw in ("<think>private</think>Hello!", "<think>unfinished"):
            provider = SimpleNamespace(complete=AsyncMock(return_value=ChatCompletionMessage(role="assistant", content=raw)))
            bot = Assistant(provider, dry_run=True)
            reply = await bot.reply("Hello")
            self.assertEqual(reply, "Hello!" if "</think>" in raw else "Please try again.")
            self.assertNotIn("<think>", bot.history[-1]["content"])


if __name__ == "__main__":
    unittest.main()
