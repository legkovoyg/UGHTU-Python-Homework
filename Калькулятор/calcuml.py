import tkinter as tk
import math

# МОЗГИ
# Калькулятор
def chet (operation):
    global fomula
    if operation == 'C':
        fomula = ''
    elif operation == "del":
        if fomula ==('Деление на ноль'):
            fomula =  fomula[0:-20]
        else:fomula = fomula[0:-1]
    elif operation == 'x^2':
        fomula = str((eval(fomula))**2)
    elif operation == 'log10':
        fomula=math.log10(int(fomula))
        fomula = str(int(fomula))
    elif operation == 'e^':
        fomula = str(2.71828182845905**(eval(fomula)))
    elif operation == 'sqrt':
        fomula = str((eval(fomula))**0.5)
    elif operation == '!':
        fomula = math.factorial(int(fomula))
        fomula = str(int(fomula))
    elif operation == '=':
        if fomula == '1488+228':
            fomula = 'Дима лох'
        elif len(fomula)>20:
            fomula = 'Max 20 simbl'
        else:
            try:
                fomula = str(eval(fomula))
            except ZeroDivisionError:
                fomula = str('Деление на ноль')
    elif operation == '+/-':
        fomula = str(-eval(fomula))
    else:
        if fomula =="0":
            fomula =''
        fomula += operation
    label_text.configure (text=fomula)
#Ловит нажатия клавиш
def press_key(event):
    if event.char.isdigit():
        add_digit(event.char)
#Передает цифры в калькулятор с клавы
def add_digit (digit):
    chet(digit)

# Окно
window = tk.Tk()
window.title('Калькулятор')
window.geometry('490x625')
window.resizable(width=False, height=False)
window.configure(bg='#202020', )
window.bind('<Key>',press_key)

# Окно для вывода
fomula='0'
label_text = tk.Label(text = fomula, font = ('Arial',30,'bold'), bg= '#202020', fg= '#33CC66')
label_text.place(x=20,y=50)


# Кнопки
buttons = ['sqrt','e^','!','log10',
           'x^2','del','*','C',
           '1','2','3','/',
           '4','5','6','+',
           '7','8','9','-',
           '+/-','0','.','=']
x = 18
y = 140
for button in buttons:
    get_rslt = lambda x=button: chet(x)
    (tk.Button(text=button,
                    foreground ="#FFFFFF",
                    background ="#33CC66",
                    font =('Arial',22),
                    command =get_rslt).
                    place(x=x, y=y, width = "105", height = "70"))
    x +=117
    if x>400:
        x=18
        y += 81
window.mainloop()
