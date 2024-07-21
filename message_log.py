import pickle
from telegram import Message
import config


# Message log
# ==============================================================================
# class that handles message history
class TgMessageLog:
    def __init__(self):
        try:
            with open("message_log.pickle", "rb") as f:
                self.data = pickle.load(f)
        except FileNotFoundError:
            self.data = {}

    def __del__(self):
        with open("message_log.pickle", "wb") as f:
            pickle.dump(self.data, f)

    def add_message(self, message: Message):
        if message.chat.id not in self.data:
            self.data[message.chat.id] = {}

        self.data[message.chat.id][message.message_id] = message

    def get_message(self, chat_id: int, message_id: int):
        return self.data.get(chat_id, {}).get(message_id, None)

    def get_thread(
        self, message: Message, max_len: int = config.max_thread_len
    ):
        thread_len = 0
        thread_log = []
        cur_message: Message | None = message
        while cur_message and thread_len < max_len:
            thread_len += len(cur_message.text or "")
            thread_log.append(cur_message)

            reply_to = cur_message.reply_to_message

            if reply_to:
                h_message = self.get_message(
                    reply_to.chat.id, reply_to.message_id
                )
                cur_message = h_message or reply_to
            else:
                cur_message = None

        thread_log.reverse()
        return thread_log


message_log = TgMessageLog()
