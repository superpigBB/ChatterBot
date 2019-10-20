from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer

# Create a new chat bot named Charlie
chatbot = ChatBot('Piggy',
                  database_url = r'mysql://localhost:8889/test')
# print(chatbot.name)

trainer = ListTrainer(chatbot)

#
# trainer.train([
#     "Hi, can I help you?",
#     "Sure, I'd like to book a flight to Iceland.",
#     "Your flight has been booked."
# ])
#
# # Get a response to the input text 'I would like to book a flight.'
# # response = chatbot.get_response('Hi, I want to book a flight')
# trainer.train([
#     "Hi there!",
#     "Hello",
# ])
#
trainer.train([
    "aRE You a bot",
    "I am a data py-ilot here",
])
response = chatbot.get_response('Are you a bot')
print(response)
