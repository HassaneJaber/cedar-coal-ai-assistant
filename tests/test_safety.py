import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from openai.types.chat import ChatCompletionMessage
from assistant import Assistant, Provider, KNOWLEDGE, ARABIZI_EXAMPLES, ARABIC_EXAMPLES, system_prompt
from eval import judge_system_prompt
from reply_policy import SAFETY, CALLBACK_OFFER, CROSS_CONTACT, finalize_reply, turn_instructions, is_allergy, is_dietary, plain_text


def fake_provider(content):
    return SimpleNamespace(complete=AsyncMock(return_value=ChatCompletionMessage(role="assistant", content=content)))


class SafetyTests(unittest.IsolatedAsyncioTestCase):
    async def test_allergy_final_replies_in_four_languages(self):
        cases = {
            "en": ("I'm allergic to nuts. Is hummus safe?", "May I have your name?"),
            "fr": ("Je suis allergique aux noix. Le hummus est sans danger ?", "Quel est votre nom ?"),
            "ar": ("عندي حساسية من المكسرات، الحمص آمن؟", "شو اسمك؟"),
            "az": ("3ande 7asasiye men nuts, el hummus safe?", "Shu esmak?"),
        }
        for language, (question, answer) in cases.items():
            with self.subTest(language=language):
                bot = Assistant(fake_provider(answer), dry_run=True)
                reply = await bot.reply(question)
                self.assertEqual(bot.language, language)
                self.assertTrue(reply.startswith(SAFETY[language]))
                self.assertTrue(reply.endswith(CALLBACK_OFFER[language]))
                self.assertNotIn(answer, reply)
                self.assertEqual(reply.count(SAFETY[language]), 1)
                self.assertEqual(bot.history[-1]["content"], reply)
                self.assertEqual(bot.captured_tools, [])
                self.assertEqual(reply.count("?") + reply.count("؟"), 1)

    async def test_complete_safety_is_not_duplicated_and_partial_is_completed(self):
        for language in SAFETY:
            with self.subTest(language=language):
                complete = SAFETY[language] + "\n\nHummus."
                self.assertEqual(finalize_reply("allergy", complete, language), "Hummus.\n\n" + SAFETY[language] + "\n\n" + CALLBACK_OFFER[language])
                partial = CROSS_CONTACT[language] + "\nHummus."
                self.assertEqual(finalize_reply("allergic", partial, language), "Hummus.\n\n" + SAFETY[language] + "\n\n" + CALLBACK_OFFER[language])

    async def test_dietary_final_replies_in_four_languages(self):
        cases = {
            "en": ["Do you have gluten-free food?", "Any vegan options?", "Vegetarian food?"],
            "fr": ["Avez-vous des plats sans gluten ?", "Je suis végétalien", "Je suis végétarien"],
            "ar": ["في أكل خالي من الغلوتين؟", "في اكل نباتي؟", "أكل بدون جلوتين؟"],
            "az": ["fi shi bala gluten?", "shu 3andkon mezza nabati?", "fi vegan options?"],
        }
        for language, questions in cases.items():
            for question in questions:
                with self.subTest(question=question):
                    bot = Assistant(fake_provider("Hummus."), dry_run=True)
                    reply = await bot.reply(question)
                    self.assertEqual(bot.language, language)
                    self.assertEqual(reply, "Hummus.\n\n" + CROSS_CONTACT[language])
                    self.assertEqual(finalize_reply(question, reply, language), reply)

    async def test_allergy_detection_variants_and_non_dietary_faqs(self):
        for word in ["allergy", "allergies", "allergic", "allergique", "allergie", "حساسية", "حَسَاسِيَّة", "تحسس", "7asasiye", "7assasiyye", "hasasiye"]:
            self.assertTrue(is_allergy(word), word)
        for question in ["How much is the mixed grill?", "fi parking?", "Can I book for four?", "Quelle heure est-il ?"]:
            self.assertFalse(is_allergy(question))
            self.assertFalse(is_dietary(question))
            self.assertEqual(finalize_reply(question, "Answer", "en"), "Answer")

    async def test_safety_covers_guard_and_provider_failure_paths(self):
        bot = Assistant(None, dry_run=True)
        reply = await bot.reply("Book for 15 people; I have an allergy")
        self.assertIn(SAFETY['en'], reply)
        self.assertTrue(reply.endswith(CALLBACK_OFFER['en']))
        self.assertIn("limited to 10", reply)
        provider = SimpleNamespace(complete=AsyncMock(side_effect=RuntimeError("Temporarily unavailable.")))
        bot = Assistant(provider, dry_run=True)
        reply = await bot.reply("I have an allergy")
        self.assertIn(SAFETY['en'], reply)
        self.assertTrue(reply.endswith(CALLBACK_OFFER['en']))

    async def test_plain_text_removes_markup_and_preserves_menu_lists(self):
        raw = "## Menu\n**Mixed grill**: $22\n* Hummus\n- Moutabal\n__Details__: `portion`"
        expected = "Menu\nMixed grill: $22\n- Hummus\n- Moutabal\nDetails: portion"
        self.assertEqual(plain_text(raw), expected)
        bot = Assistant(fake_provider(raw), dry_run=True)
        self.assertEqual(await bot.reply("What does the mixed grill cost?"), expected)
        self.assertEqual(bot.history[-1]['content'], expected)

    async def test_assembled_prompt_has_style_boundary_and_factual_authority(self):
        bot = Assistant(fake_provider("La, ma fi arguileh 3anna."), dry_run=True)
        await bot.reply("fi arguileh?")
        prompt = bot.provider.complete.await_args.args[0][0]["content"]
        self.assertIn(ARABIZI_EXAMPLES, prompt)
        self.assertTrue(prompt.endswith(KNOWLEDGE))
        self.assertIn("STYLE EXAMPLES ONLY", prompt)
        self.assertIn("All facts must come ONLY", prompt)
        self.assertIn("NEVER Arabic script", prompt)
        self.assertIn("English words are fine", prompt)
        self.assertNotIn("NOT English", prompt)
        self.assertIn("Tabbouleh, fattoush, kibbeh, sambousek and fatayer contain gluten", prompt)
        self.assertIn("ALWAYS include", prompt)
        self.assertIn("NEVER offer a callback for unrelated topics", prompt)
        self.assertEqual(ARABIZI_EXAMPLES.count("Q:"), 9)
        self.assertNotRegex(SAFETY['az'] + CROSS_CONTACT['az'], r"[\u0600-\u06ff]")

    async def test_medium_default_explicit_effort_and_judge_grounding(self):
        # Clear settings to test the default, without using or exposing the local key.
        with patch.dict(os.environ, {"GROQ_API_KEY":"offline-test"}, clear=True):
            main = Provider()
            judge = Provider(reasoning_effort="low")
            self.assertEqual(main.reasoning_effort, "medium")
            self.assertEqual(judge.reasoning_effort, "low")
            await main.client.close()
            await judge.client.close()
        prompt = judge_system_prompt()
        self.assertTrue(prompt.endswith(KNOWLEDGE))
        self.assertIn("FAIL an Arabizi reply containing ANY Arabic script", prompt)
        self.assertIn("unsupported by the restaurant reference", prompt)
        self.assertIn("invented sides, invented price ranges", prompt)


if __name__ == '__main__':
    unittest.main()
