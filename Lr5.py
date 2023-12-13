from docx import Document

# Открываем файл
doc = Document(r"C:\Users\ars4z\Downloads\файл для анализа.docx")

# Получаем текст из файла
text = []
for paragraph in doc.paragraphs:
    text.append(paragraph.text)

# Выводим содержимое файла
# print('\n'.join(text))
dct = {}
for i in set(text):
    if text.count(i) > 1:
        indexes = [j for j, x in enumerate(text) if x == i]
        print(*indexes, ':', i)
