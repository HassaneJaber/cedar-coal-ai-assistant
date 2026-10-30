import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from openai.types.chat import ChatCompletionMessage
from assistant import Assistant, ARABIC_EXAMPLES, ARABIZI_EXAMPLES, KNOWLEDGE
from reply_policy import SAFETY, CALLBACK_OFFER, CROSS_CONTACT, MORE_DETAILS, finalize_reply, turn_instructions


class DedupTests(unittest.IsolatedAsyncioTestCase):
    async def test_allergy_warning_exactly_once_preserves_dish_and_removes_name_four_languages(self):
        examples = {
            'en': ("I have an allergy", "Hummus has no nuts, but contains tahini (sesame paste).", "May I have your name?", "Cross‑contact is possible. Our team can call you back."),
            'fr': ("Je suis allergique", "Le hummus contient du tahini (pâte de sésame), pas de noix.", "Quel est votre nom ?", "Une contamination croisée est possible. L'équipe peut vous rappeler."),
            'ar': ("عندي حساسية", "الحمص ما فيه مكسرات، بس فيه طحينة من السمسم.", "شو اسمك؟", "التلامس بين المكونات ممكن. الفريق فيه يتصل فيك."),
            'az': ("3ande 7asasiye", "El hummus ma fi nuts, bas fi tahini (sesame paste).", "Shu esmak?", "Cross-contamination momken. El team fiyo ye7kik."),
        }
        for lang, (question, dish, name, duplicate) in examples.items():
            with self.subTest(language=lang):
                raw = SAFETY[lang] + '\n' + dish + ' ' + duplicate + '\n' + SAFETY[lang] + '\n' + name
                provider = SimpleNamespace(complete=AsyncMock(return_value=ChatCompletionMessage(role='assistant',content=raw)))
                bot = Assistant(provider, dry_run=True)
                reply = await bot.reply(question)
                self.assertEqual(reply, dish + '\n\n' + SAFETY[lang] + '\n\n' + CALLBACK_OFFER[lang])
                self.assertEqual(reply.count(SAFETY[lang]), 1)
                self.assertEqual(reply.count(dish), 1)
                self.assertEqual(bot.history[-1]['content'], reply)
                self.assertEqual(finalize_reply(question, reply, lang), reply)
                prompt = provider.complete.await_args.args[0][0]['content']
                self.assertIn('ALLERGY TURN', prompt)
                self.assertIn('Do NOT write kitchen-allergen warnings', prompt)
                self.assertIn('Ingredient facts about the asked dish are allowed', prompt)

    async def test_dietary_warning_exactly_once_four_languages(self):
        questions = {'en':'Any vegetarian food?', 'fr':'Je suis végétarien', 'ar':'في شي نباتي؟', 'az':'shu fi nabati?'}
        for lang, question in questions.items():
            with self.subTest(language=lang):
                raw = CROSS_CONTACT[lang] + '\nHummus, moutabal, batata harra.\n' + SAFETY[lang] + '\n' + CROSS_CONTACT[lang]
                reply = finalize_reply(question, raw, lang)
                self.assertEqual(reply, 'Hummus, moutabal, batata harra.\n' + MORE_DETAILS[lang] + '\n\n' + CROSS_CONTACT[lang])
                self.assertEqual(reply.count(CROSS_CONTACT[lang]), 1)
                self.assertNotIn(SAFETY[lang], reply)
                self.assertEqual(finalize_reply(question, reply, lang), reply)

    async def test_short_category_lists_cap_dishes_and_remove_price_labels(self):
        dishes = ['Hummus','Moutabal','Tabbouleh','Fattoush','Warak enab','Labneh','Cheese sambousek','Spinach fatayer','Batata harra','Grilled halloumi','Fruit platter']
        raw = 'Our menu:\n' + '\n'.join('- ' + dish + ' (VG, GF) – $5' for dish in dishes)
        for lang, question in {'en':'Show me the mezze', 'fr':'Le menu ?', 'ar':'شو المنيو؟', 'az':'shu 3andkon mezza?'}.items():
            with self.subTest(language=lang):
                reply = finalize_reply(question,raw,lang)
                self.assertEqual(reply, ', '.join(dishes[:10]) + '.\n' + MORE_DETAILS[lang])
                self.assertNotIn('$',reply)
                self.assertNotIn('VG',reply)
                self.assertNotIn('GF',reply)

    async def test_prices_are_preserved_when_explicitly_requested(self):
        raw = '- Hummus: $5\n- Moutabal: $5'
        for lang, question in {'en':'What are the mezze prices?', 'fr':'Quels sont les prix du menu ?', 'ar':'قديش اسعار المنيو؟', 'az':'addesh el mezza?'}.items():
            with self.subTest(language=lang):
                self.assertEqual(finalize_reply(question, raw, lang), raw)

    async def test_arabic_style_loaded_and_turn_rules_refresh_for_tools(self):
        first = ChatCompletionMessage(role='assistant', tool_calls=[{'id':'test','type':'function','function':{'name':'unknown','arguments':'{}'}}])
        provider = SimpleNamespace(complete=AsyncMock(side_effect=[first,ChatCompletionMessage(role='assistant',content='Hummus.')]))
        bot = Assistant(provider,dry_run=True)
        await bot.reply('في اكل بلا غلوتين؟')
        self.assertEqual(provider.complete.await_count,2)
        for call in provider.complete.await_args_list:
            prompt = call.args[0][0]['content']
            self.assertIn(ARABIC_EXAMPLES,prompt)
            self.assertIn(ARABIZI_EXAMPLES,prompt)
            self.assertIn('LEBANESE ARABIC STYLE EXAMPLES ONLY',prompt)
            self.assertIn('DIETARY TURN',prompt)
            self.assertIn('at most 10 dishes',prompt)
            self.assertIn('بندورة not طماطم',prompt)
            self.assertTrue(prompt.endswith(KNOWLEDGE))
        self.assertEqual(ARABIC_EXAMPLES.count('Q:'),4)
        self.assertIn('arayes always contains gluten', turn_instructions('gluten-free menu'))

    async def test_comma_labels_do_not_create_fake_dishes_and_bread_qualifier_survives(self):
        raw = 'Choices: Hummus (VG, GF) $5, Moutabal (VG, GF) $5, Shish taouk (GF without bread) $12.'
        reply = finalize_reply('Which gluten-free dishes?', raw, 'en')
        self.assertEqual(reply, 'Hummus, Moutabal, Shish taouk without bread.\n' + MORE_DETAILS['en'] + '\n\n' + CROSS_CONTACT['en'])

    async def test_corrected_knowledge_classifications(self):
        row = next(line for line in KNOWLEDGE.splitlines() if line.startswith('| Arayes kafta |'))
        self.assertIn('contains gluten: kafta grilled inside Lebanese bread',row)
        self.assertNotIn('GF',row)
        self.assertIn('except arayes kafta, which is grilled inside bread', KNOWLEDGE)
        self.assertEqual(KNOWLEDGE.count('tahini (sesame paste)'),2)
        self.assertIn('garlic sauce (toum, توم)', KNOWLEDGE)


if __name__ == '__main__':
    unittest.main()
