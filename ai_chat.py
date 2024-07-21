from telegram import Message


# interface class for GPT-3
class AiChatInterface:
    def get_response(self, messages: list[Message]):
        raise NotImplementedError


class AiChat(AiChatInterface):
    @staticmethod
    def __tg_to_gpt(message: Message):
        message_text = message.text
        user_full_name = message.from_user.full_name
        user_name = message.from_user.name

        role = "assistant" if user_name == config.bot_tg_name else "user"
        content = (
            f"{user_full_name} ({user_name}): {message_text}"
            if role == "user"
            else message_text
        )
        content = re.sub(bot_tg_name, bot_name, content)

        return {"role": role, "content": content}

    @staticmethod
    def __wrap_gpt_thread(thread_messages: list[dict]):
        # wrap messages in
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "assistant", "content": greeter_prompt},
            *thread_messages,
        ]
        return messages

    def get_response(self, messages: list[Message]):
        messages = list(map(AiChat.__tg_to_gpt, messages))
        messages = AiChat.__wrap_gpt_thread(messages)

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=1.0,
            # top_p = 0.3,
            # presence_penalty=1.5,
            # frequency_penalty=1.5,
            messages=messages,
        )
        response_text = response["choices"][0]["message"]["content"]

        # log bot response
        logging.info(
            f"Response:\n{bot_full_name} ({bot_name}): {response_text}"
        )

        # remove bot name from response
        response_text = re.sub(
            (
                rf"({bot_full_name}|{bot_name}) *\(?"
                rf"({bot_full_name}|{bot_name}|)\)? *:? *"
            ),
            "",
            response_text,
        )

        return response_text


openai.api_key = os.getenv("openaiAPI")
ai_chat: AiChatInterface = AiChat()
