import pandas as pd

# Задание 1
mydf = pd.DataFrame({'Экзамен/Зачёт':['Экзамен', 'Зачёт', 'Зачёт', 'Зачёт', 'Зачёт', 'Экзамен', 'Зачёт', 'Экзамен', 'Зачёт', 'Экзамен', 'Зачёт', 'Экзамен', 'Экзамен', 'Зачёт', 'Экзамен', 'Зачёт'],
                     'Оценка':['3', '', '', '', '', '4', '', '5', '', '4/3', '', '3', '3', '', '3', ''],
                     'Семестр':['1/2', '2', '2', '2', '2', '2', '1/2', '2', '1/2', '1/2', '2', '1', '1', '1', '1', '1'],
                     'ФИО преподавателя':['Макарян В.Г.', 'Теряева', 'Шопин', '', '', 'Рощупкина И.Ю.', 'Решетин А.А.', 'Матвеева И.А.', 'Слобожанина', 'Ширяева Л.К.', '', 'Парамонова', 'Коновалова', 'Решетин А.А.', 'Алкеев', '']},
                     index = ['Физика', 'Электротехника', 'Элементная база электроники', 'Основы биологии', 'Основы формирования инклюзивного взаимодействия', 'Химия',
                              'Элективные курсы по физической культуре и спорту', 'Язык программирования Python', 'Иностранный язык', 'Математика', 'Ознакомительная практика',
                              'История (история России, всеобщая история)', 'Линейная алгебра', 'Физическая культура и спорт', 'Инженерная и компьютерная графика', 'Информатика и программирование'])
# print(mydf)
mydf.to_csv('ЖукАА.csv')

only_exams_df = mydf.loc[mydf['Экзамен/Зачёт'] == 'Экзамен']
# print(only_exams_df)


# Задание 2
import matplotlib.pyplot as plt
def more_than_50(row):
  if row.Age > 50:
    return 'yes'
  else:
    return 'no'


df_1 = pd.read_excel(r"C:\Users\ars4z\Downloads\1 DF.xlsx")
print(df_1['Insurance Provider'].unique())
df_2 = pd.read_excel(r"C:\Users\ars4z\Downloads\2 DF.xlsx")

ax = df_1.groupby('Insurance Provider').Name.count().plot.bar(x='Insurance Provider', y='Name', rot=0)
plt.show()

# Объединение таблиц по Name
df_3 = df_1.join(df_2.set_index('Name'), on='Name', lsuffix='_caller', rsuffix='_other')

# Редактирование: удаление повторяющихся столбцов и переименование
df_3 = df_3.drop(['Age_other', 'Gender_other'], axis=1).rename(columns={"Age_caller": "Age", "Gender_caller": "Gender"})

# Добавление нового столбца
df_3['more_than_50'] = df_3.apply(more_than_50, axis=1)

# Серия из названий пустых столбцов и количества в них пустых значений
nan_columns = pd.Series([df_3[i].isna().sum() for i in df_3.columns[df_3.isna().any()].tolist()], index = df_3.columns[df_3.isna().any()]).sort_values(ascending = False)

# Замена пустых значений на 'Unknown'
df_3 = df_3.fillna('Unknown')

print(df_3.groupby('Test Results').Age.agg([len, min, max]))

fig, ax = plt.subplots()
ax_1 = df_3[df_3['Test Results'] == 'Normal'].groupby('Gender').Name.count().plot.bar(title='Normal')
ax_2 = df_3[df_3['Test Results'] == 'Abnormal'].groupby('Gender').Name.count().plot.bar(title='Abnormal')

ax_3 = df_3[df_3['Test Results'] == 'Normal'].plot.hist(x='Test Results', y='Age', title='Normal', bins=25)
ax_4 = df_3[df_3['Test Results'] == 'Abnormal'].plot.hist(x='Test Results', y='Age', title='Abnormal', bins=25)

plt.show()

# Работа с возрастными группами
gr_1 = df_3[df_3.Age <= 45]
gr_2 = df_3.loc[((df_3.Age > 45) & (df_3.Age <= 60))]
gr_3 = df_3[df_3.Age > 60]

print('До 45')
print(gr_1.groupby('Gender').Name.count(), '\n')
print(gr_1.groupby('Test Results').Name.count(), '\n')
print(gr_1.groupby('Medical Condition').Name.count(), '\n')

print('От 46 до 60')
print(gr_2.groupby('Gender').Name.count(), '\n')
print(gr_2.groupby('Test Results').Name.count(), '\n')
print(gr_2.groupby('Medical Condition').Name.count(), '\n')

print('Старше 61')
print(gr_3.groupby('Gender').Name.count(), '\n')
print(gr_3.groupby('Test Results').Name.count(), '\n')
print(gr_3.groupby('Medical Condition').Name.count(), '\n')