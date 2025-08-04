import PySimpleGUI as sg

from scipy.optimize import newton, bisect, fsolve, root

import sympy as sp
from sympy import diff, symbols, lambdify
from sympy.abc import x

import numpy as np

#define a flag constant for error handling
FLAG = -1
x = symbols("x")

#newton raphson method
def NewtonRaphson(f,df,x0,epsilon,iteraciones):
    xn=x0 #initialize var
    
    try:
        df(xn)
        for n in range(0,iteraciones):

            #evaluate in the function
            fxn=f(xn)

            #check for the tolerance and end iterations (solution found)
            if abs(fxn) < epsilon:

                print("Found solution after",n,"iterations. ")
                return xn, n

            #evaluate in the derivate
            dfxn=df(xn)

            #if the derivate equals 0 the solution can't be found
            if dfxn == 0:
                layout = [
                    [sg.Text("Zero derivative, solution not found. ")],
                    [sg.Button("OK")]
                ]
                DisplayNoneResult(layout)
                return None, FLAG

            newton_step = fxn/dfxn
            xn = xn - newton_step

        #if the solution was not found after max iterations: return nothing (no solution found)
        layout = [
            [sg.Text("Exceed maximum iterations. No solution found. ")],
            [sg.Button("OK")]
        ]
        DisplayNoneResult(layout) 
        return None, FLAG 
        
    except Exception as e:
        SecantMethod(f, round(newton(f,xn),2), epsilon, iteraciones)                   


#newton raphson modified function operations
def NewtonRaphsonModified_Operations(f, df, ddf, x):
    numerator = f*df
    denominator = df**2 - f*ddf

    #error handling
    if denominator == 0:
        return FLAG
    return x - numerator/denominator


#newton raphson modified method main function
def NewtonRaphsonModified(f, df, ddf, x0, epsilon, iteraciones):
    xn = float(x0)

    try:
        df(xn)

        for n in range(iteraciones):
            fxn = f(xn)
            dfxn = df(xn)
            ddfxn = ddf(xn)

            if abs(fxn) < epsilon:
                print("Solution found after",n,"iterations.")
                return xn, n

            xn = NewtonRaphsonModified_Operations(fxn, dfxn, ddfxn, xn)

            if xn == FLAG:
                layout = [
                    [sg.Text("Zero derivative, solution not found. ")],
                    [sg.Button("OK")]
                ]
                DisplayNoneResult(layout)
                return None, FLAG
        
        #if the solution was not found after max iterations: return nothing (no solution found)
        layout = [
            [sg.Text("Exceed maximum iterations. No solution found. ")],
            [sg.Button("OK")]
        ]
        DisplayNoneResult(layout)
        return None, FLAG
    
    except Exception as e:
        return SecantMethod(f, round(newton(f,xn),2), epsilon, iteraciones)                   


def SecantMethod(f, x0, epsilon, iteraciones):
    sg.popup("Couldn't evaluate in provided derivative. Using secant method.")
            
    x1 = x0 + epsilon
                
    for n in range(0,iteraciones):

        fx0 = f(x0)
        fx1 = f(x1)

        xn = x0 - (fx0 / ((fx0 - fx1) / (x0 - x1)))

        if abs(fx1 - fx0) < epsilon:
            return xn, n   

        x0 = x1
        x1 = xn

    layout = [
        [sg.Text("No solution found within the specified iterations.")],
        [sg.Button("OK")]
    ]
    DisplayNoneResult(layout)
    
    return xn, iteraciones

    

#bisection method function
def BisectionMethod(f, a, b, epsilon):
    
    if np.sign(f(a)) == np.sign(f(b)):
        layout = [
            [sg.Text("Scalars do not bound a root. ")],
            [sg.Button("OK")]
        ]
        DisplayNoneResult(layout)
        return FLAG

    m = (a + b) / 2

    if np.abs(f(m)) < epsilon:
        return m
    elif np.sign(f(a)) == np.sign(f(m)):
        return BisectionMethod(f, m, b, epsilon)
    elif np.sign(f(b)) == np.sign(f(m)):
        return BisectionMethod(f, a, m, epsilon)
    

#returns first and second derivates    
def GetDerivatives(f):
    try:
        df, ddf = lambdify(x, diff(f, x)), lambdify(x, diff(diff(f, x), x))
    except Exception as e:
        layout = [
           [sg.Text("Exception: can't compute derivatives.")],
           [sg.Text("\n\n")],
           [sg.Text("Please enter the derivatives for the provided functions.")],
           [sg.Text("Enter the first derivatives:")], [sg.InputText(key="-DF-")],
           [sg.Text("Enter the second derivative:")], [sg.InputText(key="-DDF-")],
           [sg.Text("\n")],
           [sg.Text("Step can be omitted but answers may vary. ")],
           [sg.Text("\n\n")],
           [sg.Button("Submit"), sg.Button("Omit"), sg.Button("Cancel")]
        ]
        window = sg.Window("Exception", layout)
        while True:
            event, values = window.read()
            if event == sg.WINDOW_CLOSED or event == "Cancel" or event == "Omit":
                window.close()
                return lambda: None, lambda: None
                
            elif event == "Submit":
                df = values["-DF-"]
                ddf = values["-DDF-"]
                window.close()
                return lambdify(x, df, 'numpy'), lambdify(x, ddf, 'numpy')

    return df, ddf


### ### ### ### ### ### ### ### ### ### ###

     # User interface code section #

### ### ### ### ### ### ### ### ### ### ###


def InputFunction():
    layout = [
        [sg.Text("Enter the new function:")],
        [sg.InputText(key="-FUNCTION-")],
        [sg.Button("Submit"), sg.Button("Cancel")]
    ]
    window = sg.Window("Input Function", layout)
    while True:
        event, values = window.read()
        if event == sg.WINDOW_CLOSED or event == "Cancel":
            window.close()
            return None
        elif event == "Submit":
            function_str = values["-FUNCTION-"]
            window.close()
            return function_str

def ChooseInput():
    layout = [
        [sg.Text("For this method you can input multiple estimate values.")],
        [sg.Text("How many values would you like to input?"), sg.InputText(key="-LENGTH-")],
        [sg.Text("\n\n")]
        [sg.Button("Submit")]
    ]

    window = sg.Window("Choose Input", layout)

    while True:
        event, values = window.read()
        if event == sg.WINDOW_CLOSED:
            window.close()
            return None
        elif event == "Submit":
            try:
                length = int(values["-LENGTH-"])
                if length <= 0:
                    sg.popup("You must input at least 1 estimate value.")
                elif length > 16:
                    sg.popup("Array too big to compute. Maximum estimate values: 16")
                else:
                    window.close()
                    return ArrayOfInputsEstimateValue(length)
            except ValueError:
                sg.popup("Please enter a valid integer value.")

def InputEstimateValue():
    layout = [
        [sg.Text("Introduce the estimate value:")],
        [sg.InputText(key="-VALUE-")],
        [sg.Text("\n")],
        [sg.Button("Submit")]
    ]

    while True:
        window = sg.Window("Input Estimate Value", layout)
        event, values = window.read()
        if event == sg.WINDOW_CLOSED:
            window.close()
            return None
        elif event == "Submit":
            try:
                estimate_value = float(values["-VALUE-"])
                window.close()
                return estimate_value
            except ValueError:
                sg.popup("Please enter a valid float value.")

def ArrayOfInputsEstimateValue(length):
    input_texts = [[sg.InputText(key=f"-VALUE{i}-{j}-") for j in range(4*i, min(4*(i+1), length))] for i in range((length + 3) // 4)]
    layout = [
        [sg.Text(f"Enter {length} values:")],
        *input_texts,
        [sg.Text("\n")],
        [sg.Button("Submit")]
    ]
    window = sg.Window("Input Multiple Values", layout)

    while True:
        event, values = window.read()
        if event == sg.WINDOW_CLOSED:
            window.close()
            return None
        elif event == "Submit":
            try:
                x_array = [float(values[f"-VALUE{i}-{j}-"]) for i in range((length + 3) // 4) for j in range(min(4, length - 4*i))]
                window.close()
                return x_array
            except ValueError:
                sg.popup("Please enter valid float values.")

def InputPoints(f):
    layout = [
        [sg.Text("Enter point A:")],
        [sg.InputText(key="-A-")],
        [sg.Text("Enter point B:")],
        [sg.InputText(key="-B-")],
        [sg.Text("\n")],
        [sg.Button("Submit"), sg.Button("Cancel")]
    ]
    window = sg.Window("Input Points", layout)

    while True:
        event, values = window.read()
        if event == sg.WINDOW_CLOSED or event == "Cancel":
            window.close()
            return None, None
        elif event == "Submit":
            try:
                point_a = float(values["-A-"])
                point_b = float(values["-B-"])
                if np.sign(f(point_a)) != np.sign(f(point_b)):
                    window.close()
                    return point_a, point_b
                else:
                    sg.popup("The evaluation of inputs must return different signs; try again.")
            except ValueError:
                sg.popup("Please enter valid float values for point A and point B.")

def InputEpsilon():
    layout = [
        [sg.Text("Enter the tolerance:")],
        [sg.InputText(key="-EPSILON-")],
        [sg.Text("\n")],
        [sg.Button("Submit"), sg.Button("Cancel")]
    ]
    window = sg.Window("Input Epsilon", layout)

    while True:
        event, values = window.read()
        if event == sg.WINDOW_CLOSED or event == "Cancel":
            window.close()
            return None
        elif event == "Submit":
            try:
                epsilon = float(values["-EPSILON-"])
                window.close()
                return epsilon
            except ValueError:
                sg.popup("Please enter a valid float value for the tolerance.")

def InputIterations():
    layout = [
        [sg.Text("Enter the number of iterations:")],
        [sg.InputText(key="-ITERATIONS-")],
        [sg.Text("\n")],
        [sg.Button("Submit"), sg.Button("Cancel")]
    ]
    window = sg.Window("Input Iterations", layout)

    while True:
        event, values = window.read()
        if event == sg.WINDOW_CLOSED or event == "Cancel":
            window.close()
            return None
        elif event == "Submit":
            try:
                iterations = int(values["-ITERATIONS-"])
                window.close()
                return iterations
            except ValueError:
                sg.popup("Please enter a valid integer value for the number of iterations.")

def OutputResult(result):
    
    layout = [
        [sg.Text("Solution found: " + str(result))],
        [sg.Button("OK")]
    ]
    window = sg.Window("Result", layout)
    event, _ = window.read()
    window.close()

def ExceptionWindow(e):
    layout = [
        [sg.Text("An exception ocurred: no solution found, try inputting another estimate value. " + str(e))],
        [sg.Button("OK")]
    ]
    window = sg.Window("Exception", layout)
    event, _ = window.read()
    window.close()

def DisplayNoneResult(layout):
    window = sg.Window("Result", layout)
    event, _ = window.read()
    window.close()
    return

def PopFunctionInfo(message):
    layout = [
        [sg.Text("Solving with " + message )],
        [sg.Text("Parameters are indicated in the function.")],
        [sg.Button("OK")]
    ]

    window = sg.Window("Function info", layout)
    event, _ = window.read()
    window.close()

### ### ### ### ### ### ### ### ### ### 

        # Start of main code #

### ### ### ### ### ### ### ### ### ### 

def MainLoop():

    # define a function
    function_str = 'x'
    f = lambdify(x, function_str, 'numpy')
    df, ddf = GetDerivatives(function_str)

    while True:
        function_text = sg.Text("Current Function: " + str(function_str), key="-FUNCTION-")
        layout = [
            [function_text],
            [sg.Text("\n")],
            [sg.Text("Choose an option:")],
            [sg.Text("Scipy menu: ")],
            [sg.Button("Newton"), sg.Button("Bisection"),
             sg.Button("FSolve"), sg.Button("Root")],
            [sg.Text("Redefined functions: ")],
            [sg.Button("BisectionMethod"), sg.Button("NewtonRaphson"),
             sg.Button("NewtonRaphsonModified")], 
            [sg.Text("Additional options: ")],
            [sg.Button("Input Function"), sg.Button("Quit")]
        ]
        window = sg.Window("Main Menu (Numerical methods)", layout)

        event, values = window.read()

        try:
            if event == sg.WINDOW_CLOSED or event == "Quit":
                break
            elif event == "Input Function":
                function_str = InputFunction()
                if function_str:
                    f = lambdify(x, function_str, 'numpy')
                    sg.popup("Function f(x) was updated; new function: ", function_str)
                    df, ddf = GetDerivatives(sp.sympify(function_str))
            elif event == "Newton":
                PopFunctionInfo("newton(f, x0)")
                estimate_value = InputEstimateValue()
                if estimate_value:
                    try:
                        OutputResult(newton(f, estimate_value))
                    except Exception as e:
                        ExceptionWindow(e)
                    ans = True
            elif event == "Bisection":
                PopFunctionInfo("bisct(f, point_a, point_b)")
                point_a, point_b = InputPoints(f)
                try:
                    OutputResult(bisect(f, point_a, point_b))
                except Exception as e:
                    ExceptionWindow(e)
                ans = True
            elif event == "FSolve":
                PopFunctionInfo("fsolve(f, estimate_value)")
                estimate_value = ChooseInput() 
                if estimate_value:            
                    try:
                        OutputResult(fsolve(f, estimate_value))
                    except Exception as e:
                        ExceptionWindow(e)
                    ans = True
            elif event == "Root":
                PopFunctionInfo("root(f, estimate_value).x")
                estimate_value = ChooseInput()
                if estimate_value:
                    try:
                        OutputResult(root(f, estimate_value).x)
                    except Exception as e:
                        ExceptionWindow(e)
                    ans = True
            elif event == "BisectionMethod":
                PopFunctionInfo("BisectionMethod(f, point_a, point_b, epsilon)")
                point_a, point_b = InputPoints(f)
                try:
                    OutputResult(BisectionMethod(f, point_a, point_b, InputEpsilon()))
                except Exception as e:
                    ExceptionWindow(e)
                ans = True
            elif event == "NewtonRaphson":
                PopFunctionInfo("NewtonRaphson(f, df, estimate_value, epsilon, iterations)")
                estimate_value = InputEstimateValue()
                if estimate_value:
                    try:
                        OutputResult(NewtonRaphson(f, df, estimate_value, InputEpsilon(), InputIterations()))
                    except Exception as e:
                        ExceptionWindow(e)
                    ans = True
            elif event == "NewtonRaphsonModified":
                PopFunctionInfo("NewtonRaphsonModified(f, df, ddf, estimate_value, epsilon, iterations)")
                estimate_value = InputEstimateValue()
                if estimate_value:
                    try:
                        OutputResult(NewtonRaphsonModified(f, df, ddf, estimate_value, InputEpsilon(), InputIterations()))                
                    except Exception as e:
                        ExceptionWindow(e)
                    ans = True
        
        except Exception as e:
            print("Something went wrong. ")

        window.close()

#initialize main loop
MainLoop()