from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from sentence_transformers import SentenceTransformer, util
import os

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

DB_PATH = "db.txt"
if not os.path.exists(DB_PATH):
    print(f"Файл {DB_PATH} не найден!")
    exit()

questions, answers = load_qa_from_file(DB_PATH)
question_embeddings = model.encode(questions, convert_to_tensor=True)

def ask_bot_semantic(user_input):
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
    response = ask_bot_semantic(user_text)
    await update.message.reply_text(response)

def main():
    TOKEN = "ТВОЙ_ТОКЕН"

    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))

    print("Бот запущен...")
    app.run_polling()

if __name__ == '__main__':
    main()