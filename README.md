# Yuki-Bot-Virtual-Assistant

Chatbot based Virtual Assistant developed in Python.

Used Keras/Tensorflow to create a transformer-based machine learning model to handle general non-keyword based conversations. Data used to train the model came from a month of reddit comments in 2015.

Capabilities:
- Can say different greetings based on the time of day.
- Can tell date and time
- can search for anything on google
- Uses Wikipedia API to provide basic summary on any topic within the Wikipedia database
- Can do calculations up to exponents
- Uses requests and beautifulsoup4 to extract and tell the weather forecast and provide cooking recipes
- Can play music
- Can tell jokes
- Can create notes
- Can provide you with the location of a given address
- Dictionary and thesaurus
- any non-keyword based inputs are generated from the machine learning model
- Every response is also spoken in audio.

Future updates:
- (MAIN PRIORITY): re-train the model with a better dataset for more coherent conversations
- (MAIN PRIORITY): have inputs also be speech-based
- Execute basic computer commands (e.g. shut down, restart, sleep)
- Translate any languages
- Send emails
- Open apps in the computer
- Schedule events in a calendar
