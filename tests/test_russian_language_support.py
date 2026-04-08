"""
Тесты поддержки русского языка в EverMemOS.

Проверяет:
1. SmartTextParser — кириллица группируется в слова (score 1.5)
2. _detect_language — корректно определяет ru/zh/en
3. _tokenize_text — русский стеммер и стоп-слова
"""

import pytest
import sys
import os

from common_utils.text_utils import SmartTextParser, TokenType


class TestCyrillicTextParser:
    """Кириллица в SmartTextParser"""

    def setup_method(self):
        self.parser = SmartTextParser()

    def test_cyrillic_word_grouped(self):
        """Кириллические символы группируются в одно слово"""
        tokens = self.parser.parse_tokens("Привет")
        assert len(tokens) == 1
        assert tokens[0].type == TokenType.ENGLISH_WORD
        assert tokens[0].content == "Привет"

    def test_cyrillic_word_score(self):
        """Кириллическое слово получает score 1.5 (как английское)"""
        tokens = self.parser.parse_tokens("Пользователь")
        assert tokens[0].score == 1.5

    def test_cyrillic_not_other(self):
        """Кириллица НЕ попадает в OTHER"""
        tokens = self.parser.parse_tokens("Тест")
        assert tokens[0].type != TokenType.OTHER

    def test_mixed_russian_english(self):
        """Смешанный русско-английский текст"""
        tokens = self.parser.parse_tokens("Hello Мир")
        # "Hello", " ", "Мир"
        words = [t for t in tokens if t.type == TokenType.ENGLISH_WORD]
        assert len(words) == 2
        assert words[0].content == "Hello"
        assert words[1].content == "Мир"

    def test_russian_sentence_score(self):
        """Score русского предложения — по словам, не по буквам"""
        tokens = self.parser.parse_tokens("Привет мир")
        words = [t for t in tokens if t.type == TokenType.ENGLISH_WORD]
        # 2 слова × 1.5 = 3.0
        total_word_score = sum(t.score for t in words)
        assert total_word_score == 3.0

    def test_long_russian_word_still_one_token(self):
        """Длинное русское слово — один токен, не 12 отдельных"""
        tokens = self.parser.parse_tokens("пользователей")
        words = [t for t in tokens if t.type == TokenType.ENGLISH_WORD]
        assert len(words) == 1
        assert words[0].score == 1.5

    def test_russian_with_cjk(self):
        """Русский + китайский — разные типы токенов"""
        tokens = self.parser.parse_tokens("Привет 你好")
        types = {t.type for t in tokens if t.type != TokenType.WHITESPACE}
        assert TokenType.ENGLISH_WORD in types  # Привет
        assert TokenType.CJK_CHAR in types  # 你, 好

    def test_russian_with_numbers(self):
        """Русский с числами"""
        tokens = self.parser.parse_tokens("Версия 3.12")
        words = [t for t in tokens if t.type == TokenType.ENGLISH_WORD]
        numbers = [t for t in tokens if t.type == TokenType.CONTINUOUS_NUMBER]
        assert len(words) == 1  # "Версия"
        assert len(numbers) == 1  # "3.12"


class TestDetectLanguage:
    """_detect_language определяет язык текста"""

    def test_detect_russian(self):
        from agentic_layer.retrieval_utils import _detect_language
        assert _detect_language("Привет, как дела?") == "ru"

    def test_detect_chinese(self):
        from agentic_layer.retrieval_utils import _detect_language
        assert _detect_language("你好世界") == "zh"

    def test_detect_english(self):
        from agentic_layer.retrieval_utils import _detect_language
        assert _detect_language("Hello world") == "en"

    def test_mixed_russian_english_prefers_russian(self):
        from agentic_layer.retrieval_utils import _detect_language
        assert _detect_language("Hello Привет") == "ru"

    def test_mixed_chinese_russian_prefers_chinese(self):
        from agentic_layer.retrieval_utils import _detect_language
        assert _detect_language("你好 Привет") == "zh"


class TestTokenizeText:
    """_tokenize_text с русским стеммером и стоп-словами"""

    @pytest.fixture(autouse=True)
    def setup(self):
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import PorterStemmer, SnowballStemmer

        for resource in ["punkt", "punkt_tab", "stopwords"]:
            try:
                nltk.data.find(f"tokenizers/{resource}" if resource != "stopwords" else f"corpora/{resource}")
            except LookupError:
                nltk.download(resource, quiet=True)

        self.stemmers = {
            "en": PorterStemmer(),
            "ru": SnowballStemmer("russian"),
        }
        self.stop_words_map = {
            "en": set(stopwords.words("english")),
            "ru": set(stopwords.words("russian")),
        }

    def test_russian_stopwords_filtered(self):
        """Русские стоп-слова фильтруются"""
        from agentic_layer.retrieval_utils import _tokenize_text
        # "и", "в", "на" — стоп-слова
        tokens = _tokenize_text("кот и собака в парке на улице", "ru", self.stemmers, self.stop_words_map)
        for stopword in ["и", "в", "на"]:
            assert stopword not in tokens

    def test_russian_stemming(self):
        """Русский стеммер работает"""
        from agentic_layer.retrieval_utils import _tokenize_text
        tokens = _tokenize_text("пользователи пользователей", "ru", self.stemmers, self.stop_words_map)
        # Обе формы должны стеммиться к одной основе
        assert len(set(tokens)) == 1

    def test_english_still_works(self):
        """Английский пайплайн не сломан"""
        from agentic_layer.retrieval_utils import _tokenize_text
        tokens = _tokenize_text("running users quickly", "en", self.stemmers, self.stop_words_map)
        assert len(tokens) > 0
        # "quickly" -> "quick" (стемминг)
        assert any("quick" in t for t in tokens)

    def test_chinese_still_works(self):
        """Китайский пайплайн не сломан"""
        from agentic_layer.retrieval_utils import _tokenize_text
        tokens = _tokenize_text("苹果很好吃", "zh", self.stemmers, self.stop_words_map)
        assert len(tokens) > 0
