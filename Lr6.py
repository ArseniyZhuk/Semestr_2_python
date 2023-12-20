# Задание 1
class WaterContainer:
    def __init__(self, capacity, current_amount):
        self.capacity = capacity  # Объем емкости
        self.current_amount = current_amount  # Текущее количество жидкости

    def pour_out(self, amount):
        if self.current_amount >= amount:
            self.current_amount -= amount
            return amount
        else:
            print('В кувшине нет столько жидкости!')
            return 0
            # poured_out = self.current_amount
            # self.current_amount = 0
            # return poured_out

    def fill_glass(self, amount):
        space_available = self.capacity - self.current_amount
        if amount <= space_available:
            self.current_amount += amount
            return amount
        else:
            print('В стакан не поместится столько!')
            return '0'
            # filled = space_available
            # self.current_amount = self.capacity
            # return filled

    def fill_jug(self, amount):
        space_available = self.capacity - self.current_amount
        if amount <= space_available:
            self.current_amount += amount
        else:
            print('Не хватает места в кувшине.')

class Jug(WaterContainer):
    def __init__(self, capacity, current_amount):
        super().__init__(capacity, current_amount)

class Glass(WaterContainer):
    def __init__(self, capacity, current_amount):
        super().__init__(capacity, current_amount)


jug = Jug(1000, 500)  # Создание кувшина
glass = Glass(250, 100)  # Создание стакана


def transfer_water(source, destination, amount):
    transferred = source.pour_out(amount)  # Переливаем воду из исходного контейнера
    actual_amount_poured = destination.fill_glass(transferred)  # Наполняем второй контейнер полученным объемом
    if actual_amount_poured == '0':
        source.fill_jug(amount)
    print(f"Перелито {actual_amount_poured} мл воды из кувшина в стакан")


print(f"В кувшине осталось {jug.current_amount} мл воды и в стакане {glass.current_amount} мл воды")
while True:
    print(f"Вы хотите перелить жидкость из кувшина в стакан или долить ещё в стакан?\nНажмите 1 или 0, или любой другой символ если Вы хотите закончить.")
    n = input()
    if n == '1':
        amount = int(input('Введите сколько Вы хотите перелить: '))
        transfer_water(jug, glass, amount)
        print(f"В кувшине осталось {jug.current_amount} мл воды и в стакане {glass.current_amount} мл воды\n")
    elif n == '0':
        amount = int(input('Введите сколько Вы хотите долить: '))
        jug.fill_jug(amount)
        print(f"В кувшине осталось {jug.current_amount} мл воды и в стакане {glass.current_amount} мл воды\n")
    else:
        break

# transfer_water(jug, glass, 100)
# print(f"В кувшине осталось {jug.current_amount} мл воды и в стакане {glass.current_amount} мл воды")
# transfer_water(jug, glass, 200)
# print(f"В кувшине осталось {jug.current_amount} мл воды и в стакане {glass.current_amount} мл воды")
# transfer_water(jug, glass, 300)
# print(f"В кувшине осталось {jug.current_amount} мл воды и в стакане {glass.current_amount} мл воды")
# transfer_water(jug, glass, 400)
# print(f"В кувшине осталось {jug.current_amount} мл воды и в стакане {glass.current_amount} мл воды")
# transfer_water(jug, glass, 500)
# print(f"В кувшине осталось {jug.current_amount} мл воды и в стакане {glass.current_amount} мл воды")


# Задание 2
import random
import math

class Circle:
    def __init__(self, radius):
        self.radius = radius

    def calculate_area(self):
        return round(math.pi, 2) * (self.radius ** 2)

    def calculate_perimeter(self):
        return 2 * round(math.pi, 2) * self.radius

class Rectangular:
    def __init__(self, width, length):
        self.width = width
        self.length = length

    def calculate_area(self):
        return self.width * self.length

    def calculate_perimeter(self):
        return 2 * (self.width + self.length)

class Square(Rectangular):
    def __init__(self, side):
        super().__init__(side, side)

class Game:
    def __init__(self):
        pass

    def play(self):
        start_game = input("Готовы ли вы начать игру? (да/нет)\n")
        if start_game.lower() == "да":
            self.run()
        else:
            print("Хорошо, спасибо! До новых встреч!")

    def get_shape(self):
        shapes = [Circle, Rectangular, Square]
        random_shape = random.choice(shapes)

        if random_shape == Circle:
            radius = random.randint(1, 10)
            return random_shape(radius)
        elif random_shape == Rectangular:
            width = random.randint(1, 10)
            length = random.randint(1, 10)
            return random_shape(width, length)
        elif random_shape == Square:
            side = random.randint(1, 10)
            return random_shape(side)

    def run(self):
        shape = self.get_shape()
        if isinstance(shape, Circle):
            print("Фигура: Круг")
            print(f"Радиус круга: {shape.radius}")
            user_area = float(input("Введите значение площади круга: "))
            user_perimeter = float(input("Введите значение периметра круга: "))
            if (user_area == shape.calculate_area()) and (user_perimeter == shape.calculate_perimeter()):
                print("Все верно!")
            else:
                print("Ошибка")
        elif isinstance(shape, Rectangular):
            print("Фигура: Прямоугольник")
            print(f"Ширина: {shape.width}, Длина: {shape.length}")
            user_area = float(input("Введите значение площади прямоугольника: "))
            user_perimeter = float(input("Введите значение периметра прямоугольника: "))
            if (user_area == shape.calculate_area()) and (user_perimeter == shape.calculate_perimeter()):
                print("Все верно!")
            else:
                print("Ошибка")
        elif isinstance(shape, Square):
            print("Фигура: Квадрат")
            print(f"Сторона квадрата: {shape.width} (или {shape.length})")
            user_area = float(input("Введите значение площади квадрата: "))
            user_perimeter = float(input("Введите значение периметра квадрата: "))
            if (user_area == shape.calculate_area()) and (user_perimeter == shape.calculate_perimeter()):
                print("Все верно!")
            else:
                print("Ошибка")


# game = Game()
# game.play()
