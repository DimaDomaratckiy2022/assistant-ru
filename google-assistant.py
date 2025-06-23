import multiprocessing

import os

import sys

import time



from telegram import Update

from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

from sentence_transformers import SentenceTransformer, util



DB_PATH = "db.txt"

TOKEN = "7882568330:AAF6hYR62drjbFsxdu8JpnEiDFgDgVI5E6g"



model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')





def load_qa_from_file(filepath):

    questions = []

    answers = []

    with open(filepath, 'r', encoding='utf-8') as f:

        for line in f:

            line = line.strip()

            if not line or "?" not in line:

                continue

            q, a = line.split("?", 1)

            questions.append(q.strip())

            answers.append(a.strip())

    return questions, answers





def ask_bot_semantic(user_input, questions, answers, question_embeddings):

    user_embedding = model.encode(user_input, convert_to_tensor=True)

    hits = util.semantic_search(user_embedding, question_embeddings, top_k=1)

    hit = hits[0][0]

    score = hit['score']

    idx = hit['corpus_id']

    if score > 0.6:

        return answers[idx]

    else:

        return "Извини, я не знаю ответа на этот вопрос."





async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):

    await update.message.reply_text("Привет! Задай мне вопрос.")





async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):

    user_text = update.message.text

    questions = context.bot_data["questions"]

    answers = context.bot_data["answers"]

    embeddings = context.bot_data["embeddings"]

    response = ask_bot_semantic(user_text, questions, answers, embeddings)

    await update.message.reply_text(response)





def bot_process():

    if not os.path.exists(DB_PATH):

        print(f"Файл {DB_PATH} не найден!")

        return



    questions, answers = load_qa_from_file(DB_PATH)

    embeddings = model.encode(questions, convert_to_tensor=True)



    app = ApplicationBuilder().token(TOKEN).build()

    app.bot_data["questions"] = questions

    app.bot_data["answers"] = answers

    app.bot_data["embeddings"] = embeddings



    app.add_handler(CommandHandler("start", start))

    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))



    print("Бот работает...")

    app.run_polling()





def main():

    multiprocessing.set_start_method('spawn', force=True)



    while True:

        p = multiprocessing.Process(target=bot_process)

        p.start()



        print("\nБот запущен. Введите команду:")

        print("  r — перезапустить")

        print("  e — завершить")



        while p.is_alive():

            cmd = input(">> ").strip().lower()

            if cmd == 'e':

                print("Завершение...")

                p.terminate()

                p.join()

                sys.exit(0)

            elif cmd == 'r':

                print("Перезапуск...")

                p.terminate()

                p.join()

                break

            elif cmd != '':

                print("Неизвестная команда. Используй 'r' или 'e'.")





if __name__ == '__main__':

    main()
