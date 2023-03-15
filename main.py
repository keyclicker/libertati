import os
import re
import logging
import openai
import logging
from telegram import Update, Message
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters

# Prompt Config
#=============================================================================
bot_full_name = "Ana Tati"
bot_name = "@libertati"
stop_sequence = "$$"

preprompt = f"""{bot_full_name} ({bot_name}) is a clever libertarian girl who responds in the language of the people.
She speaks Ukrainian, Russian, and English, and sometimes uses emojis and swears. Politically incorrect.
She can be toxic and uses sarcasm and jokes. 

Global chat:
{bot_full_name} {bot_name}:
Доброго дня, дорогі друзі!
{stop_sequence}
"""

# Format functions
#==============================================================================
def format_message(message):
    message_text = message.text
    user_full_name = message.from_user.full_name
    user_name = message.from_user.name

    thread_str += (
        f"[{user_full_name}][{user_name}]\n"
        f"{message_text}\n"
        f"{stop_sequence}"
    )


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
def get_response(prompt: str):
    response_text = ""
    if thread_len > 5000:
        response_text = "Надто багато тексту! Або надто великий тред. Іди нахуй, душняр!"
    else:
        model = "text-davinci-003"
        if thread_len > 400: 
            model = "text-curie-001"
        elif thread_len > 1000: 
            model = "text-babbage-001"

        response = openai.Completion.create(
            model=model,
            prompt=prompt,
            temperature=0.6,
            max_tokens=200,
            top_p=1.0,
            frequency_penalty=0.5,
            presence_penalty=0.0,
            stop=[stop_sequence]
        )

        response_text = response.choices[0].text

    # log bot response
    logging.info("[Ana Tati]: " + response_text)
    return response_text
    
def format_prompt(thread_str: str):
    # remove bot name from messages so she would not think that she is bot
    thread_str = re.sub(r"@libertati_bot", "@libertati", thread_str)

    # format prompt
    prompt = (
        f"{preprompt}\n"
        f"{thread_str}\n"
        f"[Ana Tati][@libertati]\n"
    )

# Message log
#==============================================================================
# class that handles message history
class MessageLog:
    def __init__(self):
        self.data = {}

    # def __del__(self):

    def add_message(self, message: Message):
        self.data[message.chat_id][message.id] = message

    def get_message(self, chat_id: int, message_id: int):
        self.data.get(chat_id, {}).get(message_id, None)

    def get_thread(self, message: Message, max_len: int = 300):
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

    @staticmethod
    def format_thread(thread_log: list):
        formatted_thread = ""
        for message in thread_log:
            formatted_thread = format_message(message)
            if message != thread_log[-1]:
                formatted_thread += "\n\n"
        return formatted_thread

    def get_formatted_thread(self, message: Message, max_len: int = 300):
        thread_log = self.get_thread(message, max_len)
        return self.thread_to_str(thread_log)


message_log = MessageLog()


# Handlers
#==============================================================================
async def history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # add message to history and log
    message_log.add_message(update.message)
    logging.info(f"Chat id: {update.effective_chat.id}")


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # add message to history and log
    message_log.add_message(update.message)
    logging.info(f"Chat id: {update.effective_chat.id}")

    # get message log, max 5000 characters
    formated_thread = message_log.get_formatted_thread(update.message, 5000)

    prompt = format_prompt(thread_str)
    response_text = get_response(prompt)

    # send response
    bot_message = await context.bot.send_message(
        chat_id=update.effective_chat.id,
        reply_to_message_id=update.message.message_id,
        text=response_text
    )

    #add bot message to history
    message_log.add_message(bot_message)


if __name__ == '__main__':
    logging.basicConfig(
        filename='bot.log',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logging.info('Bot started!')

    openai.api_key = os.getenv('openaiAPI')
    application = ApplicationBuilder().token(os.getenv('telegramAPI')).build()

    name_regex = r".?(T|t)ati|(A|a)na|(Т|т)ати|(Т|т)аті|(А|а)на|@libertati_bot.?"
    
    allowed_chats = None #[-1001876001667]
    chatFilter = filters.Chat(allowed_chats)

    echo_handler = MessageHandler(
        filters.TEXT  & 
        (filters.Regex(name_regex) | filter_reply_to_me | filter_random), echo)

    history_handler = MessageHandler(filters.TEXT , history)

    application.add_handler(echo_handler)
    application.add_handler(history_handler)

    application.run_polling()