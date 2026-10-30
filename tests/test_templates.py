import unittest
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from pathlib import Path
from openai.types.chat import ChatCompletionMessage
from assistant import Assistant, now
from booking_state import REQUIRED

EXPECTED = {
    "en": ["What's your name?", "What's your phone number?", "Which day would you like to come?", "What time?", "How many people?", "Inside or on the terrace?"],
    "ar": ["شو اسمك؟", "شو رقم التلفون؟", "أي يوم بدك تجي؟", "أي ساعة؟", "قديش شخص؟", "جوّا ولا عالتراس؟"],
    "az": ["Shu esmak?", "Shu ra2m el telefone?", "Ayya yom baddak teje?", "Ayya se3a?", "Addesh shakhes?", "Jowwa wala 3al terrace?"],
    "fr": ["Quel est votre nom ?", "Quel est votre numéro de téléphone ?", "Pour quel jour ?", "À quelle heure ?", "Pour combien de personnes ?", "À l'intérieur ou en terrasse ?"],
}
SAVED = {
    "en": "Your booking is confirmed! Thank you.",
    "ar": "تم تأكيد حجزك! شكراً.",
    "az": "Tam te2kid hajzak! Shukran.",
    "fr": "Votre réservation est confirmée ! Merci.",
}


def draft():
    return dict(name="Demo Guest", phone="70000001", date=(now().date()+timedelta(days=1)).isoformat(),
                time="20:00", party_size=2, seating="terrace", notes="")


class TemplateTests(unittest.IsolatedAsyncioTestCase):
    async def test_exact_questions_without_any_provider_calls(self):
        for language, prompts in EXPECTED.items():
            for index, expected in enumerate(prompts):
                with self.subTest(language=language, field=REQUIRED[index]):
                    provider = SimpleNamespace(complete=AsyncMock(side_effect=AssertionError("Routine step called the model")))
                    bot = Assistant(provider, dry_run=True)
                    bot.language = language
                    values = draft()
                    bot.execute("update_booking_draft", {field: values[field] for field in REQUIRED[:index]})
                    self.assertEqual(await bot.reply("ok"), expected)
                    provider.complete.assert_not_awaited()

    async def test_saved_message_and_review_labels(self):
        for language in EXPECTED:
            bot = Assistant(None, dry_run=True)
            bot.language = language
            bot.execute("save_booking", draft())
            review = bot.notice()
            if language == "az":
                self.assertIn("Makan: Terrace", review)
                self.assertNotIn("Terrasse", review)
                self.assertTrue(review.endswith("Mna2kid el hajez?"))
            if language == "ar":
                self.assertIn("التلفون: 70000001", review)
                self.assertIn("المكان: التراس", review)
                self.assertNotIn("الهاتف", review)
                self.assertTrue(review.endswith("منأكّد الحجز؟"))
            self.assertEqual(await bot.confirm(), SAVED[language])
            bot.execute("save_booking", {**draft(), "seating":"indoor"})
            if language == "ar":
                self.assertIn("المكان: جوّا", bot.notice())
            if language == "az":
                self.assertIn("Makan: Jowwa", bot.notice())

    async def test_off_script_question_still_uses_model(self):
        answer = "Valet parking is available from 6 pm for $3. What's your phone number?"
        provider = SimpleNamespace(complete=AsyncMock(return_value=ChatCompletionMessage(role="assistant", content=answer)))
        bot = Assistant(provider, dry_run=True)
        bot.execute("update_booking_draft", {"name":"Demo Guest"})
        self.assertEqual(await bot.reply("Is there parking?"), answer)
        provider.complete.assert_awaited_once()

    async def test_unknown_detail_uses_model_then_fixed_question(self):
        call = ChatCompletionMessage(role="assistant", tool_calls=[{
            "id":"name", "type":"function",
            "function":{"name":"update_booking_draft","arguments":'{"name":"Demo Guest"}'}}])
        provider = SimpleNamespace(complete=AsyncMock(side_effect=[call,
            ChatCompletionMessage(role="assistant", content="What phone number can we reach you on?")]))
        bot = Assistant(provider, dry_run=True)
        bot.execute("update_booking_draft", {})
        self.assertEqual(await bot.reply("My name is Demo Guest"), EXPECTED["en"][1])
        self.assertEqual(bot.booking_draft["name"], "Demo Guest")

    async def test_bare_guest_count_advances_to_seating(self):
        bot = Assistant(None, dry_run=True)
        bot.execute("update_booking_draft", {k:v for k,v in draft().items() if k not in {"party_size","seating"}})
        self.assertEqual(await bot.reply("5"), EXPECTED["en"][5])
        self.assertEqual(bot.booking_draft["party_size"], 5)

    async def test_invalid_field_still_has_model_context(self):
        provider = SimpleNamespace(complete=AsyncMock(return_value=ChatCompletionMessage(role="assistant", content="Please provide a valid phone number.")))
        bot = Assistant(provider, dry_run=True)
        bot.execute("update_booking_draft", {"name":"Demo Guest","phone":"invalid"})
        await bot.reply("ok")
        provider.complete.assert_awaited_once()
        self.assertIn("phone", bot.draft_errors)

    def test_arabic_welcome(self):
        source = (Path(__file__).resolve().parents[1]/"app.py").read_text("utf-8")
        self.assertIn("أهلا وسهلا! كيف فيني ساعدك؟ فيك تسأل عن المنيو، الدوام، التوصيل، أو تحجز طاولة. هيدا مطعم تجريبي.", source)
