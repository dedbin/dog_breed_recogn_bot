import os
import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import telebot
from token_for_bot import token, main_id
from model.model import load_and_preprocess_test_image
import numpy as np
from model.setup import breeds, translations
from keras.models import load_model

logfile = str(datetime.date.today()) + '.log'

bot = telebot.TeleBot(token=token)
bot.set_my_commands([
    telebot.types.BotCommand("/start", "в самое начало"),
    telebot.types.BotCommand("/help", "а че собственно делать"),
])

model = load_model(r'C:\Users\red1c\ dog_breed_recogn_bot\model\model.h5')


@bot.message_handler(commands=['start'])
def start(message: telebot.types.Message):
    text = f'приветствую тебя, {message.from_user.first_name}! ' \
           f'я бот который создан для определения породы собаки. пришли мне фото, а я отправлю тебе что это за порода. ' \
           f'извини, это моя ранняя версия, поэтому могут быть баги или ложные предсказания, но я учучь!' \
           f' чтобы ускорить мое обучения о всех подобных случаях пиши в телеграм моему создателю @math_is_ez'
    bot.send_message(message.chat.id, text)


@bot.message_handler(commands=['help'])
def help(message: telebot.types.Message):
    bot.send_message(message.chat.id, 'пришли мне фото, а я отправлю тебе что это за порода. '
                                      'одна из основных проблем которая была выяснена - модель плохо определяет, '
                                      'если на фото видна только мордочка или какая-то определенная часть тела. '
                                      'поэтому прошу тебя отправлять тело песы целиком')


@bot.message_handler(content_types=['photo'])
def predict_tg(message: telebot.types.Message):
    fileID = message.photo[-1].file_id
    try:
        file_info = bot.get_file(fileID)
        downloaded_file = bot.download_file(file_info.file_path)
        with open(f"image{fileID}.jpg", 'wb') as new_file:
            new_file.write(downloaded_file)
        img = load_and_preprocess_test_image(f"image{fileID}.jpg")
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img)
        try:
            bot.send_message(message.chat.id, f'я думаю что это {translations[breeds[np.argmax(prediction)]]}. если я не прав, пиши @math_is_ez')

        except KeyError as e:
            bot.send_message(message.chat.id,  f'я думаю что это {breeds[np.argmax(prediction)]}. если я не прав, пиши @math_is_ez')
    except Exception as e:
        bot.send_message(message.chat.id, "что-то пошло через жопу")
        with open(logfile, 'a', encoding='utf-8') as f:
            log = f'{datetime.datetime.today().strftime("%H:%M:%S")} id: {message.from_user.id} from: {message.from_user.first_name}\t{message.from_user.last_name}\t{message.from_user.username}: {e} \n'
            f.write(log)
            bot.send_message(main_id, log)
    finally:
        try:
        # удаление фото
            os.remove(f"image{fileID}.jpg")
        except Exception as e:
            pass

bot.polling(non_stop=True)
