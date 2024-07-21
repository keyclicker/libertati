import os
import logging
from dotenv import load_dotenv, find_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    MessageHandler,
    filters,
)
import config
from message_log import message_log
from ai_chat import ai_chat


# Message Filters
# ==============================================================================
class FilterRandom(filters.MessageFilter):
    def filter(self, message):
        return message.id % 5 == 0


filter_random = FilterRandom()


class FilterReplyToMe(filters.MessageFilter):
    def filter(self, message):
        return (
            message.reply_to_message
            and message.reply_to_message.from_user.name == config.bot_tg_name
        )


filter_reply_to_me = FilterReplyToMe()


# Handlers
# ==============================================================================
async def history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # add message to history and log
    message = update.message
    if message is None:
        logging.info("Message is None")
        return

    message_log.add_message(message)
    logging.info(f"Chat id: {message.chat.id}")
    logging.info(f"Message logged: {message.text}")


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    if message is None:
        logging.info("Message is None")
        return

    # add message to history and log
    message_log.add_message(message)
    logging.info(f"Chat id: {message.chat.id=}")

    # get message log, with max_thread_len characters
    gpt_thread = message_log.get_thread(message)

    # get response from GPT-3
    response = ai_chat.get_response(gpt_thread)

    # send response
    bot_message = await context.bot.send_message(
        chat_id=message.chat.id,
        reply_to_message_id=message.message_id,
        text=response,
    )

    # add bot message to history
    message_log.add_message(bot_message)


if __name__ == "__main__":
    # load environment variables
    load_dotenv(find_dotenv())

    # set up logging
    logging.basicConfig(
        filename="./log/bot.log",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    logging.info("Bot started!")

    # set up bot API
    application = (
        ApplicationBuilder().token(os.getenv("telegramAPI") or "").build()
    )

    # defining bot handlers
    history_handler = MessageHandler(filters.TEXT, history)
    echo_handler = MessageHandler(
        filters.TEXT
        & (
            filters.Regex(config.name_regex)
            | filter_reply_to_me
            | filter_random
        ),
        echo,
    )

    # add handlers to bot
    application.add_handler(history_handler)
    application.add_handler(echo_handler)

    # start bot
    application.run_polling()
