from draft.gui.conversations import ConversationRepository


def test_conversation_history_restores_for_same_owner(tmp_path):
    repository = ConversationRepository(tmp_path / "conversations.sqlite")
    owner = "browser-session-a"
    conversation_id = repository.get_or_create(None, owner)
    repository.add_message(conversation_id, owner, "user", "Hello PA-LAW")
    repository.add_message(conversation_id, owner, "assistant", "Hello. How can I help?")

    restored_id = repository.get_or_create(conversation_id, owner)
    messages = repository.get_messages(restored_id, owner)

    assert restored_id == conversation_id
    assert [item["role"] for item in messages] == ["user", "assistant"]
    assert messages[0]["content"] == "Hello PA-LAW"


def test_conversation_id_cannot_be_reused_by_another_owner(tmp_path):
    repository = ConversationRepository(tmp_path / "conversations.sqlite")
    first_id = repository.get_or_create(None, "browser-session-a")
    repository.add_message(first_id, "browser-session-a", "user", "Private message")

    second_id = repository.get_or_create(first_id, "browser-session-b")

    assert second_id != first_id
    assert repository.get_messages(first_id, "browser-session-b") == []
    assert repository.owns(first_id, "browser-session-a")
    assert not repository.owns(first_id, "browser-session-b")


def test_session_signing_secret_is_persistent(tmp_path):
    repository = ConversationRepository(tmp_path / "conversations.sqlite")

    assert repository.get_or_create_secret() == repository.get_or_create_secret()
