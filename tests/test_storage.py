from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lumi.contracts import ChatAnswer, EntityReference, Source
from lumi.storage import ConversationNotFound, ConversationStore


class ConversationStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.store = ConversationStore(Path(self._temporary.name) / "lumi.sqlite3")

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def test_new_conversations_inherit_user_model_and_thinking_choice(self) -> None:
        self.store.set_user_preference("account-1", "qwen3.5:4b", True)

        first = self.store.create_conversation("account-1", "qwen3.5:2b")
        self.store.update_choice("account-1", first.id, "qwen3.5:0.8b", False)
        second = self.store.create_conversation("account-1", "qwen3.5:2b")

        self.assertEqual((first.model, first.thinking), ("qwen3.5:4b", True))
        self.assertEqual((second.model, second.thinking), ("qwen3.5:4b", True))
        preference = self.store.get_user_preference("account-1")
        self.assertIsNotNone(preference)
        self.assertEqual((preference.model, preference.thinking), ("qwen3.5:4b", True))
        snapshot = self.store.get_snapshot("account-1", first.id)
        self.assertEqual(
            (snapshot.conversation.model, snapshot.conversation.thinking),
            ("qwen3.5:0.8b", False),
        )

    def test_default_is_used_when_user_has_no_saved_choice(self) -> None:
        conversation = self.store.create_conversation(
            "account-1", "qwen3.5:2b", default_thinking=True
        )

        self.assertEqual(conversation.model, "qwen3.5:2b")
        self.assertTrue(conversation.thinking)

    def test_conversation_id_can_be_bound_to_an_orchestrator_delegation(self) -> None:
        conversation = self.store.create_conversation(
            "account-1",
            "qwen3.5:2b",
            conversation_id="conversation-bound-by-orchestrator",
        )

        self.assertEqual(conversation.id, "conversation-bound-by-orchestrator")

    def test_append_turn_persists_only_visible_messages_and_trusted_metadata(self) -> None:
        conversation = self.store.create_conversation("account-1", "qwen3.5:2b")
        reference = EntityReference("series", "series-9", "Frieren")
        source = Source("https://example.org/review", "Example", "A review")
        answer = ChatAnswer(
            markdown="Try :::zenstream{type=\"series\" id=\"series-9\"}.",
            references=(reference,),
            sources=(source,),
            tool_rounds=2,
            tool_calls=4,
        )

        updated = self.store.append_turn("account-1", conversation.id, "What next?", answer)
        snapshot = self.store.get_snapshot("account-1", conversation.id)

        self.assertEqual(updated.title, "What next?")
        self.assertEqual([message.role for message in snapshot.messages], ["user", "assistant"])
        self.assertEqual(snapshot.messages[0].content, "What next?")
        self.assertEqual(snapshot.messages[1].references, (reference,))
        self.assertEqual(snapshot.messages[1].sources, (source,))
        self.assertEqual(len(snapshot.messages), 2)

    def test_conversation_operations_are_scoped_to_the_owner(self) -> None:
        conversation = self.store.create_conversation("account-1", "qwen3.5:2b")

        self.assertEqual(self.store.list_conversations("account-2"), ())
        with self.assertRaises(ConversationNotFound):
            self.store.get_snapshot("account-2", conversation.id)
        with self.assertRaises(ConversationNotFound):
            self.store.update_choice("account-2", conversation.id, "qwen3.5:4b", False)
        with self.assertRaises(ConversationNotFound):
            self.store.append_turn(
                "account-2",
                conversation.id,
                "Steal this conversation",
                ChatAnswer("No.", (), (), 0, 0),
            )

    def test_message_and_conversation_list_limits_are_bounded(self) -> None:
        conversation = self.store.create_conversation("account-1", "qwen3.5:2b")
        for index in range(55):
            self.store.append_turn(
                "account-1",
                conversation.id,
                f"Question {index}",
                ChatAnswer(f"Answer {index}", (), (), 0, 0),
            )

        snapshot = self.store.get_snapshot("account-1", conversation.id, message_limit=7_000)

        self.assertEqual(len(snapshot.messages), 110)
        self.assertEqual(len(self.store.list_conversations("account-1", limit=7_000)), 1)


if __name__ == "__main__":
    unittest.main()
