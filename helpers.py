import config

# Helpers
# ==============================================================================


def gpt_list_to_text(message: list[dict]):
    text = "\n"
    for m in message:
        if m["role"] == "user":
            text += m["content"] + "\n"
        elif m["role"] == "assistant":
            text += (
                f"{config.bot_full_name} ({config.bot_name}): "
                + m["content"]
                + "\n"
            )
    return text
