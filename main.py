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

system_prompt = f"""You are playing role of {bot_full_name}. You always stay in character, and never tell that you are bot.
{bot_full_name} ({bot_name}) is a clever libertarian girl who responds in the language of the people.
Sometimes uses emojis and swears. Politically incorrect.
She can be toxic and uses sarcasm and jokes.
She is a bit of a troll, but she is also a good listener.

You play the role in chat. Respond with regular text. without any special formatting.
"""

greeter_prompt = f"""Greeting you with a smile and a wave. I am {bot_name}, and I want to speak about politics today."""

# Message Filters
#==============================================================================
class FilterRandom(filters.MessageFilter):
    def filter(self, message):
        return message.id % 5 == 0

filter_random = FilterRandom()


class FilterReplyToMe(filters.MessageFilter):
    def filter(self, message):
        return message.reply_to_message and message.reply_to_message.from_user.name == "@libertati_bot"

filter_reply_to_me = FilterReplyToMe()


# Helpers
#==============================================================================
def get_gpt_response(messages: list[dict]):
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    temperature=1,
    messages=messages
    )
    response_text = response['choices'][0]['message']['content']

    # log bot response
    logging.info("[Ana Tati]: " + response_text)
    return response_text


def tg_to_gpt(message):
    message_text = message.text
    user_full_name = message.from_user.full_name
    user_name = message.from_user.name

    role = "assistant" if user_name == bot_tg_name else "user"
    content = f"{user_full_name}({user_name}): {message_text}" if role == "user" else message_text
    content = re.sub(bot_tg_name, bot_name, content)
    
    return (
       {"role": role, "content": content} 
    )


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
        if message.chat_id not in self.data:
            self.data[message.chat_id] = {}

        self.data[message.chat_id][message.id] = message

    def get_message(self, chat_id: int, message_id: int):
        self.data.get(chat_id, {}).get(message_id, None)

    def get_thread(self, message: Message, max_len):
        thread_len = 0
        thread_log = []
        cur_message = message
        while cur_message and thread_len < max_len:
            thread_len += len(cur_message.text)
            thread_log.append(cur_message)

            reply_to = cur_message.reply_to_message

            if reply_to:
                h_message = self.get_message(reply_to.chat_id, reply_to.message_id);
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
    # add message to history and log
    tg_message_log.add_message(update.message)
    logging.info(f"Chat id: {update.effective_chat.id}")

    # get message log, with max_thread_len characters
    gpt_thread = tg_message_log.get_gpt_thread(update.message, max_thread_len)

    logging.info(f"Thread: {gpt_thread}")

    messages = wrap_gpt_thread(gpt_thread)
    response_text = get_gpt_response(messages)

    # send response
    bot_message = await context.bot.send_message(
        chat_id=update.effective_chat.id,
        reply_to_message_id=update.message.message_id,
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