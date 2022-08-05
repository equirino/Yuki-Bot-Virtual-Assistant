import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"
import webbrowser
import tensorflow as tf
import transformer_model.hparams as hparams
import webbrowser
import wikipedia as wiki
import requests
import operator
import subprocess
import urllib.parse
import urllib.request
import re
import pafy
import vlc
import time
import pyjokes
import smtplib
import ssl
import pwinput
import pyttsx3
from bs4 import BeautifulSoup
from datetime import date, datetime
from PyDictionary import PyDictionary
from nltk.corpus import wordnet
from transformer_model.model import transformer_model
from data.process_dataset import process_data
from data.create_dataset import filter_sentence
from train import loss_function, accuracy, CustomLearningRate

strategy = hparams.tpu_strategy()

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                         'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36'}


class RecipeScraper:
    links = []
    names = []

    def __init__(self, query_dict):
        self.query_dict = query_dict

    def get_url(self, url):
        url = requests.get(url).text
        self.soup = BeautifulSoup(url, 'lxml')

    def print_info(self):
        base_url = "https://allrecipes.com/search/results/?"
        query_url = urllib.parse.urlencode(self.query_dict, doseq=True)
        url = base_url + query_url
        self.get_url(url)
        name = self.query_dict['search']

        resp = str(self.soup.find('span', class_="search-results-total-results")).strip(' ')
        resp = ''.join(re.findall('\d+', resp))

        if resp == '0':
            print(f'No recipes found for {name}')
            return

        articles = self.soup.find_all('div', class_="card__detailsContainer")

        texts = []
        for article in articles:
            txt = article.find('div', class_='card__detailsContainer-left')
            if txt:
                if len(texts) < 5:
                    texts.append(txt)
                else:
                    break
        self.links = [txt.a['href'] for txt in texts]
        self.names = [txt.h3.text for txt in texts]
        self.get_data()

    def get_data(self):
        self.ingredientsList = []
        self.instructionsList = []
        for i, link in enumerate(self.links):
            self.get_url(link)
            print('-' * 4 + self.names[i] + '-' * 4)
            info_names = [div.text.strip() for div in self.soup.find_all(
                'div', class_='recipe-meta-item-header')]
            ingredient_spans = self.soup.find_all(
                'span', class_='ingredients-item-name')
            instructions_spans = self.soup.find_all('div', class_='paragraph')
            ingredients = [span.text.strip() for span in ingredient_spans]
            instructions = [span.text.strip() for span in instructions_spans]
            for i, div in enumerate(self.soup.find_all('div',
                                                       class_='recipe-meta-item-body')):
                print(info_names[i].capitalize(), div.text.strip())
            print()
            print('Ingredients'.center(len(ingredients[0]), ' '))
            print('\n'.join(ingredients))
            print()
            print('Instructions'.center(len(instructions[0]), ' '))
            print('\n'.join(instructions))
            print()
            print('*' * 50)
            self.ingredientsList.append(ingredients)
            self.instructionsList.append(instructions)


def speak(audio):
    engine = pyttsx3.init()
    voice = engine.getProperty('voices')
    engine.setProperty('voice', voice[1].id)

    engine.say(audio)
    engine.runAndWait()


def date_time():
    current_datetime = datetime.now()

    current_date = current_datetime.strftime("%B %d, %Y")
    current_time = current_datetime.strftime("%I:%M %p")

    return current_date, current_time


def weatherrequest(city):
    city = city.replace(" ", "+")
    city_url = "https://www.google.com/search?q=weather+{}&rlz=1C1CHBF_enUS922US922&oq=weather+{}&aqs=" \
               "chrome.0.0i512l10.2972j1j1&sourceid=chrome&ie=UTF-8".format(city, city)

    scrape = requests.get(city_url, headers=headers)
    soup = BeautifulSoup(scrape.content, 'html.parser')

    weather = soup.select('#wob_tm')[0].getText().strip()
    precipitation = soup.select('#wob_pp')[0].getText().strip()
    humidity = soup.select('#wob_hm')[0].getText().strip()
    winds = soup.select('#wob_ws')[0].getText().strip()
    skies = soup.select('#wob_dc')[0].getText().strip()
    location = soup.select('#wob_loc')[0].getText().strip()
    current_date, current_time = date_time()[0:2]

    print("Yuki Bot: The weather in {} for {} {} is {}℃ with a sky condition of {}, {} chance of precipitation, "
          "{} humidity, and {} winds.".format(location, current_date, current_time,
                                              weather, skies, precipitation, humidity, winds))
    speak("The weather in {} for {} {} is {}℃ with a sky condition of {}, {} chance of precipitation, "
          "{} humidity, and {} winds.".format(location, current_date, current_time,
                                              weather, skies, precipitation, humidity, winds))


def get_operator_fn(op):
    return {
        '+': operator.add,
        'plus': operator.add,
        '-': operator.sub,
        'minus': operator.sub,
        'x': operator.mul,
        '*': operator.mul,
        'times': operator.mul,
        'divided': operator.__truediv__,
        'Mod': operator.mod,
        'mod': operator.mod,
        '^': operator.xor,
        '**': operator.pow
    }[op]


def eval_binary_expr(op1, oper, op2):
    op1, op2 = int(op1), int(op2)
    return get_operator_fn(oper)(op1, op2)


def wikisummarysearch(wikikeyword):
    try:
        wikisummary = wiki.summary(wikikeyword, sentences=3)
    except wiki.DisambiguationError as err1:
        wikikeyword = err1.options[0]
        wikisummary = wiki.summary(wikikeyword)
    except wiki.PageError:
        print("Yuki Bot: Could not find any information on{} .".format(wikikeyword))
        speak("Could not find any information on{} .".format(wikikeyword))
        return
    wikiurl = wiki.page(wikikeyword).url
    print("Yuki Bot: {} For more information, please refer to {}".format(wikisummary, wikiurl))
    speak("{}".format(wikisummary))


def webbrowsersearch(search_topic):
    try:
        webbrowser.open("https://www.google.com/search?q={}".format(search_topic))
    except ValueError:
        print("Yuki Bot: Sorry . I Could not find what you were searching for .")
        speak("Sorry . I Could not find what you were searching for .")


def note(subject, body):
    username = os.getlogin()
    current_date, current_time = date_time()[0:2]
    current_time = current_time.replace(":", " ")
    notes_folder = r"C:\Users\{}\Desktop\Yuki Bot Notes".format(username)
    if not os.path.exists(notes_folder):
        os.mkdir(notes_folder)
    else:
        pass
    note_name = "{} - created on {} {}".format(subject, current_date, current_time)
    complete_dir = os.path.join(notes_folder, note_name + ".txt")
    with open(complete_dir, "w") as f:
        f.write(body)

    print("Yuki Bot: Note has been created !")
    speak("Note has been created !")

    subprocess.Popen(["notepad.exe", complete_dir])


def play_music(music_name):
    try:
        query_string = urllib.parse.urlencode({"search_query": music_name})
        formaturl = urllib.request.urlopen("https://www.youtube.com/results?" + query_string)
        search_results_linknum = re.findall(r"watch\?v=(\S{11})", formaturl.read().decode())
        first_video_link = "https://www.youtube.com/watch?v={}".format(search_results_linknum[0])

        music_link = pafy.new(first_video_link)
        music_title = music_link.title

        print("Yuki Bot: Did you want to play {} ? Please say yes or no .".format(music_title))
        speak("Did you want to play {} ?".format(music_title))
        confirm = input("You: ")

        if "yes" in confirm:
            print("Yuki Bot: Okay ! Now playing {} .".format(music_title))
            speak("Okay ! Now playing {} .".format(music_title))

            music = music_link.getbestaudio()
            media = vlc.MediaPlayer(music.url)

            media.play()
        elif "no" in confirm:
            pass
    except (Exception,):
        print("Yuki Bot: Invalid input , sorry .")
        speak("Invalid input , sorry .")


def inference(model, tokenizer, sentence):
    sentence = filter_sentence(sentence)
    start_token, end_token = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

    sentence = tf.expand_dims(start_token + tokenizer.encode(sentence) + end_token, axis=0)

    output = tf.expand_dims(start_token, 0)

    for i in range(hparams.max_length):
        predictions = model(inputs=[sentence, output], training=False)

        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        if tf.equal(predicted_id, end_token[0]):
            break

        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0)


def predict(model, tokenizer, sentence):
    prediction = inference(model, tokenizer, sentence)

    predicted_sentence = tokenizer.decode(
        [i for i in prediction if i < tokenizer.vocab_size]
    )

    return predicted_sentence


def main():
    print("loading chatbot...")

    vocab_size, tokenizer = process_data()[2:4]

    optimizer = tf.keras.optimizers.Adam(
        CustomLearningRate(d_model=hparams.d_model),
        beta_1=0.9, beta_2=0.98, epsilon=1e-9
    )

    with strategy.scope():
        model_main = transformer_model(
            vocab_size=vocab_size,
            num_layers=hparams.num_layers,
            units=hparams.units,
            d_model=hparams.d_model,
            num_heads=hparams.num_heads,
            dropout=hparams.dropout
        )

        model_main.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])

    loaded_model = tf.keras.models.load_model(hparams.output_dir, compile=False)

    print("done loading!")
    print("-----WELCOME TO YUKI BOT VIRTUAL AI SOFTWARE! STILL IN "
          "DEVELOPMENT PHASE (REQUIRES INTERNET CONNECTION)-----")

    chatbot_status = True

    greeting_keywords = ["hello", "hello there", "hi there"]
    greeting_keywords2 = ["nice to meet you", "the name", "my name"]
    active_keywords = ["give", "what", "find", "tell", "look"]
    weather_keywords = ["weather", "forecast"]
    wiki_keywords = ["tell me more about", "tell me about", "give me a summary on", "give me a quick summary on",
                     "give me a summary about", "give me a quick summary about",
                     "I want to know more about", "I want to know about", "what do you know about",
                     "I would like to know more about", "I would like to know about",
                     "give me more information", "give me information", "give me some information"]
    note_keywords = ["write", "make", "put down"]
    personal_keywords = ["for me", "for something"]
    ending_keywords = ["bye", "goodbye", "see you later", "see you next time", "until next time", "good bye"]
    thank_keywords = ["thank you", "thanks"]
    location_keywords = ["address", "location"]
    music_keywords = ["music", "song"]

    while chatbot_status:
        chat_input = input("You: ")

        if any([i in chat_input for i in greeting_keywords]):
            current_hour = int(datetime.now().hour)
            if 0 <= current_hour < 12:
                print("Yuki Bot: Hello there ! Good morning !")
                speak("Hello there ! Good morning !")
            elif 12 <= current_hour < 18:
                print("Yuki Bot: Hi there ! Good Afternoon !")
                speak("Hi there ! Good  Afternoon !")
            else:
                print("Yuki Bot: Hey There ! Good Evening !")
                speak("Hey There ! Good Evening !")

        elif any([i in chat_input for i in greeting_keywords2]):
            print("Yuki Bot: It is nice to meet you !")
            speak("It is nice to meet you !")

        elif any([i in chat_input for i in active_keywords]) and "date" in chat_input:
            current_date = date_time()[0]
            print("Yuki Bot: Today is {} .".format(current_date))
            speak("Today is {} .".format(current_date))

        elif any([i in chat_input for i in active_keywords]) and "time" in chat_input:
            current_time = date_time()[1]
            print("Yuki Bot: It is currently {} .".format(current_time))
            speak("It is currently {} .".format(current_time))

        elif "what" and "your name" in chat_input:
            print("Yuki Bot: My name is Yuki Bot !")
            speak("My name is Yuki Bot !")

        elif any([i in chat_input for i in thank_keywords]):
            print("Yuki Bot: You are most welcome !")
            speak("You are most welcome !")

        elif "search" in chat_input and any([i in chat_input for i in personal_keywords]):
            print("Yuki Bot: What would you like for me to search ?")
            speak("What would you like for me to search ?")
            search_input = input("You: ")
            print("Yuki Bot: Searching for '{}' . . .".format(search_input))
            speak("Searching for '{}' . . .".format(search_input))

            webbrowsersearch(search_input)

        elif any([i in chat_input for i in wiki_keywords]):
            wikikeyword = chat_input.replace("tell me more about", "").replace("can you tell me more about", "")\
                .replace("give me a summary on", "").replace("give me a quick summary on", "")\
                .replace("can you give me a quick summary on", "").replace("can you give me a summary on", "")\
                .replace("give me a summary about", "").replace("give me a quick summary about", "")\
                .replace("can you give me a quick summary about", "").replace("can you give me a summary about", "")\
                .replace("can you give me a summary on", "").replace("I want to know more about", "")\
                .replace("what do you know about", "").replace("I would like to know more about", "")\
                .replace("I would like to know about", "").replace("I want to know about", "")\
                .replace("can you give me information on", "")\
                .replace("can you give me more information on", "").replace("can you give me some information on", "")\
                .replace("give me information on", "")\
                .replace("give me more information on", "").replace("give me some information on", "")\
                .replace("can you give me information about", "")\
                .replace("can you give me more information about", "")\
                .replace("can you give me some information about", "")\
                .replace("give me information about", "")\
                .replace("give me more information about", "").replace("give me some information about", "")

            wikisummarysearch(wikikeyword)

        elif "calculate" in chat_input:
            calculate_input = chat_input.replace("calculate", "")
            if "to the power of" in calculate_input:
                new_calculate_input = calculate_input.replace("to the power of", "**")
            elif "divided by" in calculate_input:
                new_calculate_input = calculate_input.replace("divided by", "divided")
            else:
                new_calculate_input = calculate_input
            try:
                print("Yuki Bot: {}".format(eval_binary_expr(*(new_calculate_input.split()))))
                speak("{}".format(eval_binary_expr(*(new_calculate_input.split()))))
            except (ValueError, KeyError):
                print("Yuki Bot: I cannot compute that at the moment , sorry .")
                speak("I cannot compute that at the moment , sorry .")
            except TypeError:
                print("Yuki Bot: Improper format ! Please only use two numbers and / or add "
                      "a space between the numbers and the operator .")
                speak("Improper format ! Please only use two numbers and or add "
                      "a space between the numbers and the operator .")

        elif any([i in chat_input for i in active_keywords]) and any([i in chat_input for i in weather_keywords]):
            print("Yuki Bot: In what area ?")
            speak("In what area ?")
            area_input = input("You: ")
            weatherrequest(area_input)

        elif any([i in chat_input for i in active_keywords]) and any([i in chat_input for i in location_keywords]):
            print("Yuki Bot: What address would you like to look for ?")
            speak("What address would you like to look for ?")
            location = input("You: ")
            print("Yuki Bot: Looking for the location of {} .".format(location))
            speak("Looking for the location of {} .".format(location))
            webbrowser.open("https://www.google.com/maps/place/" + location)

        elif any([i in chat_input for i in note_keywords]) and "note" in chat_input:
            print("Yuki Bot: What would you like to write in the note ?")
            speak("What would you like to write in the note ?")
            text = input("You: ")
            print("Yuki Bot: What would you like the note to be called ?")
            speak("What would you like the note to be called ?")
            name = input("You: ")
            note(name, text)

        elif any([i in chat_input for i in music_keywords]) and "play" in chat_input:
            print("Yuki Bot: What music would you like to hear ?")
            speak("What music would you like to hear ?")
            music_input = input("You: ")

            play_music(music_input)

        elif any([i in chat_input for i in active_keywords]) and "joke" in chat_input:
            joke = pyjokes.get_joke(language="en", category="all")
            print("Yuki Bot: {}".format(joke))
            speak("{}".format(joke))

        elif any([i in chat_input for i in active_keywords]) and "recipe" in chat_input:
            recipe_name = input("What recipe do you want to search for:")
            if recipe_name == '':
                print("Yuki Bot: Error ! No recipe provided .")
                speak("Error ! No recipe provided .")
                return

            else:
                ing_incl = input("What ingredients do you want to include (leave blank if you have no preference): ")
                ing_excl = input("What ingredients do you want to exclude (leave blank if you have no preference): ")

                sort_input = input("Press/say '1' to sort for relevance, "
                                   "'2' to sort for rating, or '3' to sort for most popular: ")

                if sort_input == '1':
                    sort_num = 're'
                elif sort_input == '2':
                    sort_num = 'ra'
                elif sort_input == '3':
                    sort_num = 'p'
                else:
                    print("Yuki Bot: Error sorting recipes .")
                    speak("Error encountered when sorting recipes .")
                    return

                query_dict = {
                    "search": recipe_name,
                    "ingIncl": ing_incl,
                    "ingExcl": ing_excl,
                    "sort": sort_num
                }

                scrap = RecipeScraper(query_dict)

                print("Yuki Bot: Here are some recipes I found for {} !".format(recipe_name))
                speak("Here are some recipes I found for {} !".format(recipe_name))

                scrap.print_info()

        elif "dictionary definition" in chat_input:
            dictionary = PyDictionary()

            print("Yuki Bot: What word do you want to find the definition of ?")
            speak("What word do you want to find the definition of ?")
            definition_input = input("You: ")

            word_definition = dictionary.meaning(definition_input, disable_errors=True)

            print("Yuki Bot: {} : {}".format(definition_input, word_definition))
            speak("{} : {}".format(definition_input, word_definition))

        elif "synonym" in chat_input:
            synonyms = []

            print("Yuki Bot: What word do you want to find the synonym of ?")
            speak("What word do you want to find the synonym of ?")

            synonym_input = input("You: ")

            for syn in wordnet.synsets(synonym_input):
                for lm in syn.lemmas():
                    synonyms.append(lm.name())

            print("Yuki Bot: Synonyms for {} : {}".format(synonym_input, synonyms))
            speak("Synonyms for {} : {}".format(synonym_input, synonyms))

        elif "antonym" in chat_input:
            antonyms = []

            print("Yuki Bot: What word do you want to find the antonym of ?")
            speak("What word do you want to find the antonym of ?")

            antonym_input = input("You: ")

            for syn in wordnet.synsets(antonym_input):
                for lm in syn.lemmas():
                    if lm.antonyms():
                        antonyms.append(lm.antonyms()[0].name())

            print("Yuki Bot: Antonyms for {} : {}".format(antonym_input, antonyms))
            speak("Antonyms for {} : {}".format(antonym_input, antonyms))

        elif any([i in chat_input for i in ending_keywords]):
            print("Yuki Bot: Goodbye ! Have a nice day !")
            speak("Goodbye ! Have a nice day !")
            quit()

        else:
            bot_response = predict(loaded_model, tokenizer, chat_input)
            print("Yuki Bot: {}".format(bot_response))
            speak("{}".format(bot_response))


if __name__ == '__main__':
    main()
