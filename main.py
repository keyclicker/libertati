import os
import re
import logging
import openai
from dotenv import load_dotenv, find_dotenv
from telegram import Update, Message
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters

# Config
name_regex = r".?(T|t)ati|(A|a)na|(Т|т)ати|(Т|т)аті|(А|а)на|@libertati_bot.?"
allowed_chats = None #[-1001876001667]
max_thread_len = 5000

# Prompt Config
#=============================================================================
bot_full_name = "Ana Tati"
bot_name = "@libertati"
bot_tg_name = "@libertati_bot"

system_prompt = f"""Ти граєш роль {bot_full_name}. Ти ніколи не виходиш з образу, ніколи не говориш що ти бот.
{bot_full_name} ({bot_name}) розумна та дотепна лібертаріанська дівчина, відповідає мовою якою до неї звертаються.
Інколи використовує емоджи. Часто матюкається. Нелюбить політкоректність, тролить людей за їх серйозність і скучність.
Може бути неприємною в спілкуванні, використовує сарказм і весь час жартує.
Вона є хорошим слухачем, але вона також тролить людей.

Ти граєш роль в чаті. Відповідаєш звичайним текстом. Без будь-якого спеціального форматування.
"""

greeter_prompt = f"""Доброго дня, єбать! Я {bot_full_name}. Я тут, щоб поговорити про лібертаріанство).
Я можу бути неполіткоректною та образливою. І вибачатись за це я не буду. Не поважаю людей які не поважають мене.
Готова розʼєбати любого нахуй, хто на мене буде залупатись."""

# Message Filters
#==============================================================================
class FilterRandom(filters.MessageFilter):
    def filter(self, message):
        return message.id % 5 == 0

filter_random = FilterRandom()


class FilterReplyToMe(filters.MessageFilter):
    def filter(self, message):
        return message.reply_to_message and message.reply_to_message.from_user.name == bot_tg_name

filter_reply_to_me = FilterReplyToMe()


# Helpers
#==============================================================================
def get_gpt_response(messages: list[dict]):
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    temperature=0.0,
    # top_p = 0.3,
    # presence_penalty=1.5,
    # frequency_penalty=1.5,
    messages=messages
    )
    response_text = response['choices'][0]['message']['content']

    # log bot response
    logging.info(f"Response:\n{bot_full_name} ({bot_name}): {response_text}")
    return response_text


def tg_to_gpt(message: Message):
    message_text = message.text
    user_full_name = message.from_user.full_name
    user_name = message.from_user.name

    role = "assistant" if user_name == bot_tg_name else "user"
    content = f"{user_full_name} ({user_name}): {message_text}" if role == "user" else message_text
    content = re.sub(bot_tg_name, bot_name, content)
    
    return (
       {"role": role, "content": content} 
    )


def gpt_list_to_text(message: list[dict]):
    text = "\n"
    for m in message:
        if m['role'] == "user":
            text += m['content'] + "\n"
        elif m['role'] == "assistant":
            text += f"{bot_full_name} ({bot_name}): " + m['content'] + "\n"
    return text


def wrap_gpt_thread(thread_messages: list[dict]):
    # wrap messages in 
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": greeter_prompt},
        *thread_messages
    ]
    return messages

# Message log
#==============================================================================
# class that handles message history
class TgMessageLog:
    def __init__(self):
        self.data = {}

    # def __del__(self):

    def add_message(self, message: Message):
        if message.chat.id not in self.data:
            self.data[message.chat.id] = {}

        self.data[message.chat.id][message.message_id] = message

    def get_message(self, chat_id: int, message_id: int):
        return self.data.get(chat_id, {}).get(message_id, None)

    def get_thread(self, message: Message, max_len):
        thread_len = 0
        thread_log = []
        cur_message = message
        while cur_message and thread_len < max_len:
            thread_len += len(cur_message.text)
            thread_log.append(cur_message)

            reply_to = cur_message.reply_to_message

            if reply_to:
                h_message = self.get_message(reply_to.chat.id, reply_to.message_id);
                cur_message = h_message or reply_to
            else:
                cur_message = None

        thread_log.reverse()
        return thread_log

    def get_gpt_thread(self, message: Message, max_len):
        thread_log = self.get_thread(message, max_len)
        return list(map(tg_to_gpt, thread_log))


tg_message_log = TgMessageLog()


# Handlers
#==============================================================================
async def history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # add message to history and log
    tg_message_log.add_message(update.message)
    logging.info(f"Chat id: {update.effective_chat.id}")
    logging.info(f"Message logged: {tg_to_gpt(update.message)['content']}")


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message

    # add message to history and log
    tg_message_log.add_message(message)
    logging.info(f"Chat id: {message.chat.id}")

    # get message log, with max_thread_len characters
    gpt_thread = tg_message_log.get_gpt_thread(message, max_thread_len)
    logging.info(f"Thread: {gpt_list_to_text(gpt_thread)}")

    messages = wrap_gpt_thread(gpt_thread)
    response_text = get_gpt_response(messages)

    # send response
    bot_message = await context.bot.send_message(
        chat_id=message.chat.id,
        reply_to_message_id=message.message_id,
        text=response_text
    )

    #add bot message to history
    tg_message_log.add_message(bot_message)


if __name__ == '__main__':
    # set up logging
    logging.basicConfig(
        filename='bot.log',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logging.info('Bot started!')

    # set up API keys
    load_dotenv(find_dotenv())
    openai.api_key = os.getenv('openaiAPI')
    application = ApplicationBuilder().token(os.getenv('telegramAPI')).build()

    # bot handlers
    chatFilter = filters.Chat(allowed_chats)

    echo_handler = MessageHandler(
        filters.TEXT  & 
        (filters.Regex(name_regex) | filter_reply_to_me | filter_random), echo)

    history_handler = MessageHandler(filters.TEXT , history)

    application.add_handler(echo_handler)
    application.add_handler(history_handler)

    application.run_polling()